import numpy as np
import tensorflow as tf
import yaml
import sys
import cv2
from serving.base import BaseServing


class TFLiteServing(BaseServing):
    def __init__(self, cfg):
        super(TFLiteServing, self).__init__(cfg)
        self.interpreter = tf.lite.Interpreter(cfg['MODEL']['tflite'])
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def infer(self, image):
        height, width, _ = image.shape
        scale = (self.input_size / width, self.input_size / height)
        scaled_image = cv2.resize(image, (0, 0), fx=scale[0], fy=scale[1],
                                  interpolation=cv2.INTER_CUBIC)
        scaled_image = np.float32(scaled_image)
        scaled_image = (scaled_image - 128) / 255.0
        scaled_image = np.expand_dims(scaled_image, axis=0)
        self.interpreter.set_tensor(self.input_details[0]['index'], scaled_image)
        self.interpreter.invoke()
        heatmaps = np.squeeze(self.interpreter.get_tensor(self.output_details[-2]['index']))
        print(np.sum(heatmaps[:, :, :-1]))
        heatmaps = cv2.resize(heatmaps, (0, 0),
                              fx=self.stride, fy=self.stride,
                              interpolation=cv2.INTER_CUBIC)

        pafs = np.squeeze(self.interpreter.get_tensor(self.output_details[-1]['index']))
        pafs = cv2.resize(pafs, (0, 0),
                          fx=self.stride, fy=self.stride,
                          interpolation=cv2.INTER_CUBIC)
        return heatmaps, pafs, scale


if __name__ == '__main__':
    config = sys.argv[1]
    with open(config, 'r') as fp:
        cfg = yaml.full_load(fp)
    serving = TFLiteServing(cfg)
    serving.predict(cfg['DATASET']['image'])
