import sys
import yaml
import os
import tensorflow as tf
import shutil
import numpy as np
from tensorflow.python.ops import summary_ops_v2
from src import models
from tensorflow.lite.python.util import run_graph_optimizations
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2


def export_saved_model(cfg):
    if cfg['MODEL']['dataset'] == 'coco':
        num_joints = 19
    elif cfg['MODEL']['dataset'] == 'kinect':
        num_joints = 33
    else:
        raise NotImplementedError
    num_pafs = num_joints * 2
    format_ = cfg['EXPORT']['format']
    assert format_ in ['sm', 'tf'], "Not supported format"
    model = models.__dict__[cfg['MODEL']['name']](num_channels=cfg['MODEL']['num_channels'],
                                                  num_refinement_stages=cfg['MODEL']['num_stages'],
                                                  num_joints=num_joints, num_pafs=num_pafs,
                                                  mobile=cfg['MODEL']['mobile'])
    model(np.zeros((1, cfg['MODEL']['input_size'], cfg['MODEL']['input_size'], 3),
                    dtype=np.float32))
    model.summary()
    checkpoint = tf.train.Checkpoint(epoch=tf.Variable(0), model=model)
    status = checkpoint.restore(tf.train.latest_checkpoint(cfg['EXPORT']['checkpoint']))
    status.assert_existing_objects_matched()
    print("Convert checkpoint at epoch %d" % int(checkpoint.epoch))
    if os.path.exists(cfg['EXPORT']['saved_model']):
        shutil.rmtree(cfg['EXPORT']['saved_model'])
    model._set_inputs(inputs=tf.ones((1, cfg['MODEL']['input_size'], cfg['MODEL']['input_size'], 3),
                                     dtype=tf.float32))
    if format_ == 'sm':
        tf.saved_model.save(model, cfg['EXPORT']['saved_model'])
    else:
        model.save(cfg['EXPORT']['saved_model'], save_format='tf')


def export_tflite(cfg):
    # Method 1: Using Keras Model
    model = tf.keras.models.load_model(cfg['EXPORT']['saved_model'], compile=False)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # Method 2: Using Concrete Function
    # func = tf.function(model).get_concrete_function(
    #     tf.TensorSpec(shape=(1, cfg['MODEL']['input_size'], cfg['MODEL']['input_size'], 3),
    #                   dtype=tf.float32))
    # converter = tf.lite.TFLiteConverter.from_concrete_functions([func])

    # Method 3: Using Saved Model
    # converter = tf.lite.TFLiteConverter.from_saved_model(cfg['EXPORT']['saved_model'])

    if cfg['EXPORT']['quantized']:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                                tf.lite.OpsSet.SELECT_TF_OPS]

    converter.experimental_new_converter = True
    tflite_model = converter.convert()
    with tf.io.gfile.GFile(cfg['EXPORT']["tf_lite"], 'wb') as fp:
        fp.write(tflite_model)


def export_frozen_graph(cfg):
    model = tf.keras.models.load_model(cfg['EXPORT']['saved_model'], compile=False)
    func = tf.function(model).get_concrete_function(
        tf.TensorSpec(shape=(1, cfg['MODEL']['input_size'], cfg['MODEL']['input_size'], 3),
                      dtype=tf.float32))
    # graph_writer = tf.summary.create_file_writer(logdir=cfg['EXPORT']['frozen_pb'])
    # with graph_writer.as_default():
    #     graph = func.graph
    #     summary_ops_v2.graph(graph.as_graph_def())
    # graph_writer.close()
    log_dir, filename = os.path.split(cfg['EXPORT']['frozen_pb'])
    frozen_func = convert_variables_to_constants_v2(func)
    frozen_func.graph.as_graph_def()
    tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                      logdir=log_dir,
                      name=filename, as_text=False)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = "5"
    available_gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in available_gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    config = sys.argv[1]
    with open(config, 'r') as fp:
        cfg = yaml.full_load(fp)
    export_saved_model(cfg)
    # export_tflite(cfg)
    # export_frozen_graph(cfg)