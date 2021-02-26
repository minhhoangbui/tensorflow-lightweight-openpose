import os
import sys
sys.path.append('/home/hoangbm/lightweight_openpose_tensorflow')
import yaml
import tensorflow as tf

from src.runner.trainer import Trainer
from src import datasets


def main(cfg):
    assert len(cfg['COMMON']['GPU'].split(',')) != 0, "No GPU is chosen"
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg['COMMON']['GPU']
    available_gpus = tf.config.experimental.list_physical_devices('GPU')
    print("There are {} gpus running".format(len(available_gpus)))
    for gpu in available_gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    mirrored_strategy = tf.distribute.MirroredStrategy()
    train_dataset = datasets.__dict__[cfg['DATASET']['name']](
        annotations_dir=cfg['DATASET']['annotation_dir'],
        images_dir=cfg['DATASET']['image_dir'],
        input_size=cfg['MODEL']['input_size'],
        stride=cfg['DATASET']['stride'],
        sigma=cfg['DATASET']['sigma'],
        paf_thickness=cfg['DATASET']['paf_thickness'],
        use_aid=cfg['DATASET']['use_aid'],
        is_training=True)
    num_train_batch = len(train_dataset) // cfg['TRAIN']['batch_size']
    train_dataset = train_dataset.get_dataset(cfg['TRAIN']['batch_size'])
    train_dist_dataset = mirrored_strategy.experimental_distribute_dataset(train_dataset)
    val_dataset = datasets.__dict__[cfg['DATASET']['name']](
        annotations_dir=cfg['DATASET']['annotation_dir'],
        images_dir=cfg['DATASET']['image_dir'],
        input_size=cfg['MODEL']['input_size'],
        stride=cfg['DATASET']['stride'],
        sigma=cfg['DATASET']['sigma'],
        paf_thickness=cfg['DATASET']['paf_thickness'],
        use_aid=False,
        is_training=False)
    num_val_batch = len(val_dataset) // cfg['VAL']['batch_size']
    val_dataset = val_dataset.get_dataset(cfg['VAL']['batch_size'])
    val_dist_dataset = mirrored_strategy.experimental_distribute_dataset(val_dataset)
    trainer = Trainer(cfg, mirrored_strategy)
    trainer.distributed_custom_loop(train_dist_dataset, val_dist_dataset, num_train_batch, num_val_batch)


if __name__ == '__main__':
    config = sys.argv[1]
    with open(config, 'r') as fp:
        cfg = yaml.full_load(fp)
    main(cfg)

