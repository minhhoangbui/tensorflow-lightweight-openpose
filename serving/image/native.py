import tensorflow as tf
import cv2
import numpy as np
import logging
import sys
import yaml
import os
from time import perf_counter
from serving.image.base import BaseServing
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)


class Serving(BaseServing):
    def __init__(self, cfg):
        super(Serving, self).__init__(cfg)
        self.model = tf.keras.models.load_model(cfg['MODEL']['saved_model_dir'],
                                                compile=False)
        logging.info(f"Finished loading model from {cfg['MODEL']['saved_model_dir']}")

    def infer(self, image):
        scaled_image, scale = self.preprocess_image(image)

        tensor_input = tf.convert_to_tensor(scaled_image, dtype=tf.float32)
        t_start = perf_counter()
        for _ in range(1000):
            stages_output = self.model(tensor_input)
        t_end = perf_counter()
        print(f'Elapsed time: {(t_end - t_start)/ 1000}')

        heatmaps = np.squeeze(stages_output[-1][0].numpy())
        pafs = np.squeeze(stages_output[-1][1].numpy())

        heatmaps = cv2.resize(heatmaps, (0, 0),
                              fx=self.stride, fy=self.stride,
                              interpolation=cv2.INTER_CUBIC)
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
