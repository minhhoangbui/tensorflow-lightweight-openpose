import numpy as np
import yaml
import sys
import os
from src import models
from src.models.backbones import ResNet50, ShuffleNetV2, MobileNetV2


def inspect_architecture(cfg):
    model = models.__dict__[cfg['MODEL']['name']](num_channels=cfg['MODEL']['num_channels'],
                                                  num_refinement_stages=cfg['MODEL']['num_stages'],
                                                  num_joints=cfg['MODEL']['num_joints'],
                                                  num_pafs=cfg['MODEL']['num_joints'] * 2,
                                                  mobile=cfg['MODEL']['mobile'])
    model(np.zeros((1, cfg['MODEL']['input_size'], cfg['MODEL']['input_size'], 3),
                   dtype=np.float32))
    model.summary()


def inspect_backbones(cfg):
    backbone = ResNet50()
    out = backbone(np.zeros((1, cfg['MODEL']['input_size'], cfg['MODEL']['input_size'], 3),
                   dtype=np.float32))
    print(out.shape)


if __name__ == '__main__':
    cfg = sys.argv[1]
    with open(cfg, 'r') as fp:
        cfg = yaml.full_load(fp)

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    # inspect_backbones(cfg)
    inspect_architecture(cfg)
