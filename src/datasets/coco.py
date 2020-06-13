import os
import pickle
import tensorflow as tf
import numpy as np
import cv2
import copy
import json

from pathlib import Path
import sys
sys.path.append(str(Path(os.path.abspath(__file__)).parent.parent.parent))
from pycocotools import mask as msk
from src.datasets.utils import set_paf, set_gaussian
from src.datasets.transformations import Scale, Rotate, CropPad, Flip


BODY_PARTS_KPT_IDS = [[1, 8], [8, 9], [9, 10], [1, 11], [11, 12], [12, 13], [1, 2],
                      [2, 3], [3, 4], [2, 16], [1, 5], [5, 6], [6, 7], [5, 17], [1, 0],
                      [0, 14], [0, 15], [14, 16], [15, 17]]


def map_decorator(func):
    def wrapper(img_id):
        return tf.py_function(
            func,
            inp=[img_id],
            Tout=(tf.float32,)
        )
    return wrapper


def get_mask(segmentations, mask):
    for segmentation in segmentations:
        rle = msk.frPyObjects(segmentation, mask.shape[0], mask.shape[1])
        mask[msk.decode(rle) > 0.5] = 0
    return mask


class CocoDataset:
    def __init__(self, annotations_dir, images_dir, input_size,
                 stride, sigma, paf_thickness, is_training):
        if is_training:
            self._annotations_dir = os.path.join(annotations_dir, 'train2017.pkl')
            self._images_dir = os.path.join(images_dir, 'train2017')
        else:
            self._annotations_dir = os.path.join(annotations_dir, 'val2017.pkl')
            self._images_dir = os.path.join(images_dir, 'val2017')
        with open(self._annotations_dir, 'rb') as f:
            self._img_ids, self._annotations = pickle.load(f)
        self.input_size = input_size
        self._stride = stride
        self._sigma = sigma
        self._paf_thickness = paf_thickness

        self.transformations = [
            Scale(),
            Rotate(pad=(128, 128, 128)),
            CropPad(pad=(128, 128, 128)),
            Flip()
        ]

    def _generate_keypoint_maps(self, sample):
        n_keypoints = 18
        n_rows, n_cols, _ = sample['image'].shape
        keypoint_maps = np.zeros(shape=(n_keypoints + 1,
                                        n_rows // self._stride, n_cols // self._stride), dtype=np.float32)  # +1 for bg

        label = sample['label']
        for keypoint_idx in range(n_keypoints):
            keypoint = label['keypoints'][keypoint_idx]
            if keypoint[2] <= 1:
                set_gaussian(keypoint_maps[keypoint_idx], keypoint[0], keypoint[1], self._stride, self._sigma)
            for another_annotation in label['processed_other_annotations']:
                keypoint = another_annotation['keypoints'][keypoint_idx]
                if keypoint[2] <= 1:
                    set_gaussian(keypoint_maps[keypoint_idx], keypoint[0], keypoint[1], self._stride, self._sigma)

        keypoint_maps[-1] = 1 - keypoint_maps.max(axis=0)
        return keypoint_maps

    def _generate_paf_maps(self, sample):
        n_pafs = len(BODY_PARTS_KPT_IDS)
        n_rows, n_cols, _ = sample['image'].shape
        paf_maps = np.zeros(shape=(n_pafs * 2, n_rows // self._stride, n_cols // self._stride), dtype=np.float32)

        label = sample['label']
        for paf_idx in range(n_pafs):
            keypoint_a = label['keypoints'][BODY_PARTS_KPT_IDS[paf_idx][0]]
            keypoint_b = label['keypoints'][BODY_PARTS_KPT_IDS[paf_idx][1]]
            if keypoint_a[2] <= 1 and keypoint_b[2] <= 1:
                set_paf(paf_maps[paf_idx * 2:paf_idx * 2 + 2],
                        keypoint_a[0], keypoint_a[1], keypoint_b[0], keypoint_b[1],
                        self._stride, self._paf_thickness)
            for another_annotation in label['processed_other_annotations']:
                keypoint_a = another_annotation['keypoints'][BODY_PARTS_KPT_IDS[paf_idx][0]]
                keypoint_b = another_annotation['keypoints'][BODY_PARTS_KPT_IDS[paf_idx][1]]
                if keypoint_a[2] <= 1 and keypoint_b[2] <= 1:
                    set_paf(paf_maps[paf_idx * 2:paf_idx * 2 + 2],
                            keypoint_a[0], keypoint_a[1], keypoint_b[0], keypoint_b[1],
                            self._stride, self._paf_thickness)
        return paf_maps

    def parse_func(self, img_id):
        img_id = img_id.numpy()
        annotation = copy.deepcopy(self._annotations[img_id])
        img_path = os.path.join(self._images_dir, annotation['img_paths'])
        mask = np.ones(shape=(annotation['img_height'], annotation['img_width']), dtype=np.float32)
        mask = get_mask(annotation['segmentations'], mask)

        image = cv2.imread(img_path, cv2.IMREAD_COLOR)

        sample = {
            'label': annotation,
            'image': image,
            'mask': mask
        }
        for func in self.transformations:
            sample = func(sample)
        mask = cv2.resize(sample['mask'], dsize=None, fx=1 / self._stride, fy=1 / self._stride,
                          interpolation=cv2.INTER_AREA)

        keypoint_maps = self._generate_keypoint_maps(sample)
        keypoint_mask = np.zeros(shape=keypoint_maps.shape, dtype=np.float32)
        for idx in range(keypoint_mask.shape[0]):
            keypoint_mask[idx] = mask

        paf_maps = self._generate_paf_maps(sample)
        paf_mask = np.zeros(shape=paf_maps.shape, dtype=np.float32)
        for idx in range(paf_mask.shape[0]):
            paf_mask[idx] = mask

        image = (sample['image'] - 128) / 255.0

        keypoint_maps = keypoint_maps.transpose((1, 2, 0))
        keypoint_mask = keypoint_mask.transpose((1, 2, 0))
        paf_maps = paf_maps.transpose((1, 2, 0))
        paf_mask = paf_mask.transpose((1, 2, 0))

        return image, keypoint_maps, paf_maps, keypoint_mask, paf_mask

    def tf_parse_func(self, img_id):
        [img, kps_maps, paf_maps, kps_mask, paf_mask] = tf.py_function(self.parse_func,
                                                                       [img_id], [tf.float32, tf.float32, tf.float32,
                                                                                  tf.float32, tf.float32])
        return img, (kps_maps, paf_maps), (kps_mask, paf_mask)

    def __len__(self):
        return len(self._img_ids)

    def get_dataset(self, batch_size):
        dataset = tf.data.Dataset.from_tensor_slices(self._img_ids)
        dataset = dataset.shuffle(buffer_size=1000).repeat(1)
        dataset = dataset.map(self.tf_parse_func, num_parallel_calls=10)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return dataset


class CocoValDataset_:
    def __init__(self, annotations_dir, images_dir, dataset_type):
        if dataset_type == 'train':
            self._annotations_dir = os.path.join(annotations_dir, 'train2017.pkl')
            self._images_dir = os.path.join(images_dir, 'train2017')
        elif dataset_type == 'val':
            self._annotations_dir = os.path.join(annotations_dir, 'val2017.pkl')
            self._images_dir = os.path.join(images_dir, 'val2017')
        else:
            raise ValueError("Don't support other types")
        with open(self._annotations_dir, 'rb') as f:
            self._img_ids, self._annotations = pickle.load(f)

    def get_images_and_annotations(self):
        return self._images_dir, self._annotations

    def __len__(self):
        return len(self._img_ids)

    def get_index(self):
        dataset = tf.data.Dataset.from_tensor_slices(self._img_ids)
        dataset = dataset.shuffle(buffer_size=1000).repeat(1)
        dataset = dataset.batch(1)
        return dataset


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


