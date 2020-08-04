import os
import yaml
import sys
import cv2
import numpy as np
from openvino.inference_engine import IECore
from serving.image.base import BaseServing


class OpenVinoServing(BaseServing):
    def __init__(self, cfg):
        super(OpenVinoServing, self).__init__(cfg)
        ie = IECore()
        model = ie.read_network(cfg['MODEL']['openvino'],
                                os.path.splitext(cfg['MODEL']['openvino'])[0] + '.bin')
        assert len(model.input_info) == 1, "Expected 1 input blob"

        self._input_layer_name = next(iter(model.input_info))
        self._output_layer_name = list(model.outputs.keys())
        self.exec_model = ie.load_network(model, cfg['COMMON']['device'])

    def infer(self, image):
        scaled_image, scale = self.preprocess_image(image)
        scaled_image = np.transpose(scaled_image, axes=(0, 3, 1, 2))

        output = self.exec_model.infer(inputs={self._input_layer_name: scaled_image})
        heatmaps = np.squeeze(output[self._output_layer_name[0]])
        pafs = np.squeeze(output[self._output_layer_name[1]])

        heatmaps = heatmaps.transpose((1, 2, 0))
        pafs = pafs.transpose((1, 2, 0))

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
    serving = OpenVinoServing(cfg)
    serving.predict(cfg['DATASET']['image'])
