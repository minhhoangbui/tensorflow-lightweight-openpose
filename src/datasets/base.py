import tensorflow as tf
import numpy as np
import os
import cv2
import copy

from pathlib import Path
import sys
sys.path.append(str(Path(os.path.abspath(__file__)).parent.parent.parent))
from src.datasets.utils import set_paf, set_gaussian, get_mask
from src.datasets.transformations import Scale, Rotate, CropPad, Flip


class BaseDataset:
    def __init__(self, input_size, stride, sigma, paf_thickness):
        self.input_size = input_size
        self._stride = stride
        self._sigma = sigma
        self._paf_thickness = paf_thickness

        self.n_keypoints = None
        self.body_parts_kpt_ids = None
        self._img_ids, self._annotations, self._images_dir = None, None, None

        self.transformations = [
            Scale(),
            Rotate(pad=(128, 128, 128)),
            CropPad(pad=(128, 128, 128)),
            Flip()
        ]

    def _generate_keypoint_maps(self, sample):
        n_rows, n_cols, _ = sample['image'].shape
        keypoint_maps = np.zeros(shape=(self.n_keypoints + 1,
                                        n_rows // self._stride, n_cols // self._stride), dtype=np.float32)  # +1 for bg

        label = sample['label']
        for keypoint_idx in range(self.n_keypoints):
            keypoint = label['keypoints'][keypoint_idx]
            if keypoint[2] <= 1:
                set_gaussian(keypoint_maps[keypoint_idx], keypoint[0], keypoint[1], self._stride, self._sigma)
            for another_annotation in label['processed_other_annotations']:
                keypoint = another_annotation['keypoints'][keypoint_idx]
                if keypoint[2] <= 1:
                    set_gaussian(keypoint_maps[keypoint_idx], keypoint[0], keypoint[1], self._stride, self._sigma)

        keypoint_maps[-1] = 1 - keypoint_maps.max(axis=0, initial=0)
        return keypoint_maps

    def _generate_paf_maps(self, sample):
        n_pafs = len(self.body_parts_kpt_ids)
        n_rows, n_cols, _ = sample['image'].shape
        paf_maps = np.zeros(shape=(n_pafs * 2, n_rows // self._stride, n_cols // self._stride), dtype=np.float32)

        label = sample['label']
        for paf_idx in range(n_pafs):
            keypoint_a = label['keypoints'][self.body_parts_kpt_ids[paf_idx][0]]
            keypoint_b = label['keypoints'][self.body_parts_kpt_ids[paf_idx][1]]
            if keypoint_a[2] <= 1 and keypoint_b[2] <= 1:
                set_paf(paf_maps[paf_idx * 2:paf_idx * 2 + 2],
                        keypoint_a[0], keypoint_a[1], keypoint_b[0], keypoint_b[1],
                        self._stride, self._paf_thickness)
            for another_annotation in label['processed_other_annotations']:
                keypoint_a = another_annotation['keypoints'][self.body_parts_kpt_ids[paf_idx][0]]
                keypoint_b = another_annotation['keypoints'][self.body_parts_kpt_ids[paf_idx][1]]
                if keypoint_a[2] <= 1 and keypoint_b[2] <= 1:
                    set_paf(paf_maps[paf_idx * 2:paf_idx * 2 + 2],
                            keypoint_a[0], keypoint_a[1], keypoint_b[0], keypoint_b[1],
                            self._stride, self._paf_thickness)
        return paf_maps

    def parse_func(self, img_id):
        img_id = img_id.numpy()
        annotation = copy.deepcopy(self._annotations[img_id])
        image_path = os.path.join(self._images_dir, annotation['img_paths'])
        mask = np.ones(shape=(annotation['img_height'], annotation['img_width']), dtype=np.float32)
        mask = get_mask(annotation['segmentations'], mask)
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)

        sample = {
            'label': annotation,
            'image': image,
            'mask': mask
        }
        for func in self.transformations:
            sample = func(sample)

        mask = cv2.resize(sample['mask'], dsize=None, fx=1 / self._stride,
                          fy=1 / self._stride, interpolation=cv2.INTER_AREA)

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