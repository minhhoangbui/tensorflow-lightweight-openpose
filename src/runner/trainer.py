import os
import tensorflow as tf
import numpy as np
import datetime
import sys
import logging
from tensorflow.keras.utils import Progbar
from src.utils.loss import get_loss
from src import models
logging.basicConfig(format='[ %(levelname)s ] %(message)s', level=logging.INFO, stream=sys.stdout)


class Trainer(object):
    def __init__(self, cfg, strategy):
        self.cfg = cfg
        self.strategy = strategy
        self.saved_dir = os.path.join(cfg['COMMON']['saved_dir'],
                                      f"{cfg['DATASET']['name']}_lw_pose_{cfg['MODEL']['mobile']}")
        if cfg['DATASET']['name'] == 'coco':
            num_joints = 19
        elif cfg['DATASET']['name'] == 'kinect':
            num_joints = 33
        else:
            raise NotImplementedError
        num_pafs = num_joints * 2

        with self.strategy.scope():
            lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=cfg['TRAIN']['learning_rate'],
                decay_steps=cfg['TRAIN']['decay_step'],
                decay_rate=cfg['TRAIN']['decay_rate'],
                staircase=True
            )
            self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr_schedule)
            self.model = models.__dict__[cfg['MODEL']['name']](num_channels=cfg['MODEL']['num_channels'],
                                                               num_refinement_stages=cfg['MODEL']['num_stages'],
                                                               num_joints=num_joints, num_pafs=num_pafs,
                                                               mobile=cfg['MODEL']['mobile'])
            self.model.build((1, cfg['MODEL']['input_size'], cfg['MODEL']['input_size'], 3))
            self.model.summary()

            # initialize model
            self.model(np.zeros((1, cfg['MODEL']['input_size'], cfg['MODEL']['input_size'], 3),
                                dtype=np.float32))

        self.checkpoint = tf.train.Checkpoint(epoch=tf.Variable(0), model=self.model)
        self.manager = tf.train.CheckpointManager(checkpoint=self.checkpoint,
                                                  directory=os.path.join(self.saved_dir, 'ckpts'),
                                                  max_to_keep=5)
        if cfg['COMMON']['retrain']:
            self._restore_weight()
        self._setup_logger()

    def _restore_weight(self):
        if self.manager.latest_checkpoint:
            self.checkpoint.restore(self.manager.latest_checkpoint).assert_consumed()
            logging.info(f"Restored from {self.manager.latest_checkpoint}")
        else:
            logging.info("Initializing from scratch.")

    def _setup_logger(self):
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = os.path.join(self.saved_dir, 'logs', current_time)
        self.writer = tf.summary.create_file_writer(log_dir)

    @tf.function
    def train_step(self, inputs):
        images, target, mask = inputs
        with tf.GradientTape() as tape:
            outputs = self.model(images, training=True)
            loss = get_loss(target=target, outputs=outputs, mask=mask)
        grads = tape.gradient(loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        return loss

    @tf.function
    def eval_step(self, inputs):
        images, target, mask = inputs
        outputs = self.model(images, training=False)
        unscaled_loss = get_loss(target=target, outputs=outputs, mask=mask)
        return unscaled_loss

    @tf.function
    def distributed_train_step(self, dist_inputs):
        def _step_fn(inputs):
            images, target, mask = inputs
            with tf.GradientTape() as tape:
                outputs = self.model(images, training=True)
                per_batch_loss = get_loss(target=target, outputs=outputs, mask=mask)
            grads = tape.gradient(per_batch_loss, self.model.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
            return per_batch_loss

        per_replica_loss = self.strategy.run(_step_fn, args=(dist_inputs,))
        mean_loss = self.strategy.reduce(
            tf.distribute.ReduceOp.MEAN, per_replica_loss, axis=None
        )
        return mean_loss

    @tf.function
    def distributed_eval_step(self, dist_inputs):
        def _step_fn(inputs):
            images, target, mask = inputs
            outputs = self.model(images, training=True)
            per_batch_loss = get_loss(target=target, outputs=outputs, mask=mask)
            return per_batch_loss

        per_replica_loss = self.strategy.run(_step_fn, args=(dist_inputs,))
        mean_loss = self.strategy.reduce(
            tf.distribute.ReduceOp.MEAN, per_replica_loss, axis=None
        )
        return mean_loss

    def custom_loop(self, train_dataset, val_dataset, num_train_batch, num_val_batch):
        epoch = last_epoch = int(self.checkpoint.epoch)
        if self.cfg['TRAIN']['num_epochs'] <= int(self.checkpoint.epoch):
            logging.info("Already reached this epoch")
            return
        for ep in range(self.cfg['TRAIN']['num_epochs'] - last_epoch):
            epoch = ep + 1 + last_epoch
            logging.info(f'Start of epoch {epoch}')
            self.checkpoint.epoch.assign_add(1)
            train_loss = 0
            val_loss = 0
            step = None
            train_progbar = Progbar(target=num_train_batch, stateful_metrics=['loss'])
            val_progbar = Progbar(target=num_val_batch, stateful_metrics=['loss'])

            for step, dist_inputs in enumerate(train_dataset):
                current_loss = self.train_step(dist_inputs)
                train_progbar.update(step + 1, [('loss', current_loss)])
                train_loss += current_loss
            train_loss /= step + 1

            for step, dist_inputs in enumerate(val_dataset):
                current_loss = self.eval_step(dist_inputs)
                val_progbar.update(step + 1, [('loss', current_loss)])
                val_loss += current_loss
            val_loss /= step + 1

            with self.writer.as_default():
                tf.summary.scalar('Training loss', train_loss, step=epoch)
                tf.summary.scalar('Val loss', val_loss, step=epoch)
                self.writer.flush()
            logging.info(f'Epoch {epoch}, Loss: {train_loss}, Test Loss: {val_loss}')
            if epoch % self.cfg['COMMON']['saved_epochs'] == 0:
                saved_path = self.manager.save()
                logging.info(f"Saved checkpoint for epoch {epoch}: {saved_path}")
        logging.info(f"Finish training at {epoch}")
        self.writer.close()

    def distributed_custom_loop(self, train_dist_dataset, val_dist_dataset, num_train_batch, num_val_batch):
        epoch = last_epoch = int(self.checkpoint.epoch)
        if self.cfg['TRAIN']['num_epochs'] <= int(self.checkpoint.epoch):
            logging.info("Already reached this epoch")
            return
        for ep in range(self.cfg['TRAIN']['num_epochs'] - last_epoch):
            epoch = ep + 1 + last_epoch
            logging.info(f'Start of epoch {epoch}')
            self.checkpoint.epoch.assign_add(1)
            train_loss = 0
            val_loss = 0
            step = None
            train_progbar = Progbar(target=num_train_batch, stateful_metrics=['loss'])
            val_progbar = Progbar(target=num_val_batch, stateful_metrics=['loss'])

            for step, dist_inputs in enumerate(train_dist_dataset):
                current_loss = self.distributed_train_step(dist_inputs)
                train_progbar.update(step + 1, [('loss', current_loss)])
                train_loss += current_loss
            train_loss /= step + 1

            for step, dist_inputs in enumerate(val_dist_dataset):
                current_loss = self.distributed_eval_step(dist_inputs)
                val_progbar.update(step + 1, [('loss', current_loss)])
                val_loss += current_loss
            val_loss /= step + 1

            with self.writer.as_default():
                tf.summary.scalar('Training loss', train_loss, step=epoch)
                tf.summary.scalar('Val loss', val_loss, step=epoch)
                self.writer.flush()
            logging.info(f'Epoch {epoch}, Loss: {train_loss}, Test Loss: {val_loss}')
            if epoch % self.cfg['COMMON']['saved_epochs'] == 0:
                saved_path = self.manager.save()
                print("Saved checkpoint for epoch {}: {}".format(epoch, saved_path))
        print("Finish training at %d" % epoch)
        self.writer.close()
