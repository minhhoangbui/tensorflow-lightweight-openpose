import tensorflow as tf
import os
import yaml
import sys
import cv2
import logging
from time import perf_counter
import numpy as np
from serving.image.base import BaseServing
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)


class FrozenServing(BaseServing):
    def __init__(self, cfg):
        super(FrozenServing, self).__init__(cfg)
        with tf.io.gfile.GFile(os.path.join(self.cfg['MODEL']['frozen_graph']), 'rb') as fp:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(fp.read())
        with tf.compat.v1.Graph().as_default() as graph:
            tf.compat.v1.import_graph_def(graph_def, name="")
        self.input = graph.get_tensor_by_name('self:0')
        self.heatmaps = graph.get_tensor_by_name('Identity_2:0')
        self.pafs = graph.get_tensor_by_name('Identity_3:0')
        self.sess = tf.compat.v1.Session(graph=graph)
        logging.info(f"Finished loading model from {cfg['MODEL']['frozen_graph']}")

    def infer(self, image):
        scaled_image, scale = self.preprocess_image(image)
        start = perf_counter()
        for _ in range(1000):
            [heatmaps, pafs] = self.sess.run(
                [self.heatmaps, self.pafs],
                feed_dict={self.input: scaled_image}
            )
        end = perf_counter()
        print(f'Elapsed time: {(end - start)/ 1000}')

        heatmaps = np.squeeze(heatmaps)
        pafs = np.squeeze(pafs)

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

    serving = FrozenServing(cfg)
    serving.predict(cfg['DATASET']['image'])