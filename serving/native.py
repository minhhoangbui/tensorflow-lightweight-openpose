import tensorflow as tf
import cv2
import numpy as np
import sys
import yaml
import os
from serving.base import BaseServing


class Serving(BaseServing):
    def __init__(self, cfg):
        super(Serving, self).__init__(cfg)
        # self.model = tf.saved_model.load(cfg['MODEL']['directory']).signatures["serving_default"]
        self.model = tf.keras.models.load_model(cfg['MODEL']['saved_model_dir'], compile=False)

    def infer(self, image):
        height, width, _ = image.shape
        scale = (self.input_size / width, self.input_size / height)

        scaled_image = cv2.resize(image, (0, 0), fx=scale[0], fy=scale[1],
                                  interpolation=cv2.INTER_CUBIC)

        scaled_image = (scaled_image - 128) / 255.0
        scaled_image = np.expand_dims(scaled_image, axis=0)

        tensor_input = tf.convert_to_tensor(scaled_image, dtype=tf.float32)

        stages_output = self.model(tensor_input)

        heatmaps = np.squeeze(stages_output[-1][0].numpy())

        heatmaps = cv2.resize(heatmaps, (0, 0),
                              fx=self.stride, fy=self.stride,
                              interpolation=cv2.INTER_CUBIC)

        pafs = np.squeeze(stages_output[-1][1].numpy())
        pafs = cv2.resize(pafs, (0, 0),
                          fx=self.stride, fy=self.stride,
                          interpolation=cv2.INTER_CUBIC)

        return heatmaps, pafs, scale


if __name__ == '__main__':
    config = sys.argv[1]
    with open(config, 'r') as fp:
        cfg = yaml.full_load(fp)
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg['COMMON']['GPU']
    available_gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in available_gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    tester = Serving(cfg)
    tester.predict(cfg['DATASET']['image'])
