import MNN
import cv2
import numpy as np
import yaml
import sys
from serving.base import BaseServing


class MNNServing(BaseServing):
    def __init__(self, cfg):
        super(MNNServing, self).__init__(cfg)
        self.interpreter = MNN.Interpreter(cfg['MODEL']['mnn'])
        self.session = self.interpreter.createSession()
        self.input_tensor = self.interpreter.getSessionInput(self.session)
        self.heatmaps_tensor = self.interpreter.getSessionOutput(self.session,
                                                                 'light_weight_open_pose/StatefulPartitionedCall/StatefulPartitionedCall/refinement_stage/sequential_11/conv_26/RefinementStage_heat_conv2d/BiasAdd')
        self.pafs_tensor = self.interpreter.getSessionOutput(self.session,
                                                             'light_weight_open_pose/StatefulPartitionedCall/StatefulPartitionedCall/refinement_stage/sequential_12/conv_28/RefinementStage_paf_conv2d/BiasAdd')

        self.heatmaps_output = MNN.Tensor(self.heatmaps_tensor.getShape(), MNN.Halide_Type_Float,
                                          np.zeros(self.heatmaps_tensor.getShape(), dtype=np.float32),
                                          MNN.Tensor_DimensionType_Caffe)
        self.pafs_output = MNN.Tensor(self.pafs_tensor.getShape(), MNN.Halide_Type_Float,
                                      np.zeros(self.pafs_tensor.getShape(), dtype=np.float32),
                                      MNN.Tensor_DimensionType_Caffe)

    def infer(self, image):
        height, width, _ = image.shape
        scale = (self.input_size / width, self.input_size / height)
        scaled_image = cv2.resize(image, (0, 0), fx=scale[0], fy=scale[1],
                                  interpolation=cv2.INTER_CUBIC)
        scaled_image = (scaled_image - 128) / 255.0
        scaled_image = np.float32(scaled_image)
        scaled_image = np.expand_dims(scaled_image, axis=0)

        tmp_input = MNN.Tensor((1, self.input_size, self.input_size, 3),
                               MNN.Halide_Type_Float, scaled_image,
                               MNN.Tensor_DimensionType_Tensorflow)

        self.input_tensor.copyFrom(tmp_input)
        self.interpreter.runSession(self.session)

        self.heatmaps_tensor.copyToHostTensor(self.heatmaps_output)
        self.pafs_tensor.copyToHostTensor(self.pafs_output)

        heatmaps = np.squeeze(self.heatmaps_output.getData())
        pafs = np.squeeze(self.pafs_output.getData())

        print(np.sum(heatmaps))
        print(np.sum(pafs))

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
    serving = MNNServing(cfg)
    serving.predict(cfg['DATASET']['image'])
