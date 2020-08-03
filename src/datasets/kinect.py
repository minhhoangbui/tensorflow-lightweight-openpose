import os
import pickle

from pathlib import Path
import sys
sys.path.append(str(Path(os.path.abspath(__file__)).parent.parent.parent))
from src.datasets.base import BaseDataset

BODY_PARTS_KPT_IDS = [[0, 22], [22, 23], [23, 24], [24, 25], [0, 18], [18, 19], [19, 20], [20, 21],
                       [0, 1], [1, 2], [2, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16],
                       [14, 17], [2, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [7, 10], [2, 3],
                       [3, 26], [26, 27], [27, 28], [27, 30], [28, 29], [30, 31], [26, 31], [26, 29]]

BODY_PARTS_PAF_IDS = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13],
                      [14, 15], [16, 17], [18, 19], [20, 21], [22, 23], [24, 25],
                      [26, 27], [28, 29], [30, 31], [32, 33], [34, 35], [36, 37],
                      [38, 39], [40, 41], [42, 43], [44, 45], [46, 47], [48, 49],
                      [50, 51], [52, 53], [54, 55], [56, 57], [58, 59], [60, 61],
                      [62, 63], [64, 65]]


class KinectDataset(BaseDataset):
    def __init__(self, annotations_dir, images_dir, input_size,
                 stride, sigma, paf_thickness, is_training):

        super(KinectDataset, self).__init__(input_size, stride, sigma, paf_thickness)
        if is_training:
            annotations_file = os.path.join(annotations_dir, 'annos.pkl')
            self._images_dir = os.path.join(images_dir, 'images')
        else:
            annotations_file = os.path.join(annotations_dir, 'val2017.pkl')
            self._images_dir = os.path.join(images_dir, 'val2017')
        with open(annotations_file, 'rb') as f:
            self._img_ids, self._annotations = pickle.load(f)
        self.n_keypoints = 32
        self.body_parts_kpt_ids = [[0, 22], [22, 23], [23, 24], [24, 25], [0, 18], [18, 19], [19, 20], [20, 21],
                                   [0, 1], [1, 2], [2, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16],
                                   [14, 17], [2, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [7, 10], [2, 3],
                                   [3, 26], [26, 27], [27, 28], [27, 30], [28, 29], [30, 31], [26, 31], [26, 29]]


def kinect(**kwargs):
    return KinectDataset(annotations_dir=kwargs['annotations_dir'], images_dir=kwargs['images_dir'],
                         input_size=kwargs['input_size'], stride=kwargs['stride'],
                         sigma=kwargs['sigma'], paf_thickness=kwargs['paf_thickness'],
                         is_training=kwargs['is_training'])


if __name__ == '__main__':
    dataset = KinectDataset(annotations_dir='/home/hoangbm/datasets/kinect/processed/',
                            images_dir='/home/hoangbm/datasets/kinect/processed/images',
                            stride=8, sigma=7, paf_thickness=1, input_size=368,
                            is_training=True)

