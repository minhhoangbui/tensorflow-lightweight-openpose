import os
import pickle
import tensorflow as tf
import json

from pathlib import Path
import sys
sys.path.append(str(Path(os.path.abspath(__file__)).parent.parent.parent))
from src.datasets.base import BaseDataset
from src.datasets.transformations import ConvertKeypoints

BODY_PARTS_KPT_IDS = [[1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10], [1, 11],
                      [11, 12], [12, 13], [1, 0], [0, 14], [14, 16], [0, 15], [15, 17], [2, 16], [5, 17]]

BODY_PARTS_PAF_IDS = ([12, 13], [20, 21], [14, 15], [16, 17], [22, 23], [24, 25], [0, 1], [2, 3], [4, 5],
                      [6, 7], [8, 9], [10, 11], [28, 29], [30, 31], [34, 35], [32, 33], [36, 37], [18, 19],
                      [26, 27])

# BODY_PARTS_KPT_IDS = [[1, 8], [8, 9], [9, 10], [1, 11], [11, 12], [12, 13], [1, 2],
#                        [2, 3], [3, 4], [2, 16], [1, 5], [5, 6], [6, 7], [5, 17], [1, 0],
#                        [0, 14], [0, 15], [14, 16], [15, 17]]
#
# BODY_PARTS_PAF_IDS = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13],
#                       [14, 15], [16, 17], [18, 19], [20, 21], [22, 23], [24, 25],
#                       [26, 27], [28, 29], [30, 31], [32, 33], [34, 35], [36, 37]]


def map_decorator(func):
    def wrapper(img_id):
        return tf.py_function(
            func,
            inp=[img_id],
            Tout=(tf.float32,)
        )
    return wrapper


class CocoDataset(BaseDataset):
    def __init__(self, annotations_dir, images_dir, input_size,
                 stride, sigma, paf_thickness, is_training):
        super(CocoDataset, self).__init__(input_size=input_size, stride=stride,
                                          sigma=sigma, paf_thickness=paf_thickness)

        if is_training:
            annotations_file = os.path.join(annotations_dir, 'train2017.pkl')
            self._images_dir = os.path.join(images_dir, 'train2017')
        else:
            annotations_file = os.path.join(annotations_dir, 'val2017.pkl')
            self._images_dir = os.path.join(images_dir, 'val2017')
        with open(annotations_file, 'rb') as f:
            self._img_ids, self._annotations = pickle.load(f)

        self.n_keypoints = 18
        self.body_parts_kpt_ids = [[1, 8], [8, 9], [9, 10], [1, 11], [11, 12], [12, 13], [1, 2],
                                   [2, 3], [3, 4], [2, 16], [1, 5], [5, 6], [6, 7], [5, 17], [1, 0],
                                   [0, 14], [0, 15], [14, 16], [15, 17]]

        self.transformations.insert(0, ConvertKeypoints())


class CocoValDataset:
    def __init__(self, annotations_dir, images_dir, dataset_type):
        if dataset_type == 'train':
            self.images_dir = os.path.join(images_dir, 'train2017')
            self.annotation_file = os.path.join(annotations_dir, 'person_keypoints_train2017.json')
        elif dataset_type == 'val':
            self.images_dir = os.path.join(images_dir, 'val2017')
            self.annotation_file = os.path.join(annotations_dir, 'person_keypoints_val2017.json')
        else:
            raise ValueError("Don't support other types")

        with open(self.annotation_file, 'r') as fp:
            annotations = json.load(fp)
        sample_size = len(annotations['images'])
        self.sample = [annotations['images'][idx]['file_name'] for idx in range(sample_size)]


def coco(**kwargs):
    return CocoDataset(annotations_dir=kwargs['annotations_dir'], images_dir=kwargs['images_dir'],
                       input_size=kwargs['input_size'], stride=kwargs['stride'],
                       sigma=kwargs['sigma'], paf_thickness=kwargs['paf_thickness'],
                       is_training=kwargs['is_training'])