import os
import sys
import yaml
import tensorflow as tf

from src.runner.evaluator import TFEvaluator, OpenVinoEvaluator


def main(cfg):
    assert len(cfg['COMMON']['GPU'].split(',')) == 1, "Only one GPU should be chosen"
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg['COMMON']['GPU']
    available_gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in available_gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    evaluator = OpenVinoEvaluator(cfg)
    evaluator.evaluate()


if __name__ == '__main__':
    config = sys.argv[1]
    with open(config, 'r') as fp:
        cfg = yaml.full_load(fp)
    main(cfg)
