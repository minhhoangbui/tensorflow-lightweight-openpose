import cv2
import json
import os
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from src.datasets.coco import CocoValDataset
import tensorflow as tf
from src.utils.keypoints_grouping import extract_keypoints, group_keypoints


def convert_to_coco_format(pose_entries, all_keypoints):
    coco_keypoints = []
    scores = []
    for n in range(len(pose_entries)):
        if len(pose_entries[n]) == 0:
            continue
        keypoints = [0] * 17 * 3
        to_coco_map = [0, -1, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3]
        person_score = pose_entries[n][-2]
        position_id = -1
        for keypoint_id in pose_entries[n][:-2]:
            position_id += 1
            if position_id == 1:  # no 'neck' in COCO
                continue

            cx, cy, score, visibility = 0, 0, 0, 0  # keypoint not found
            if keypoint_id != -1:
                cx, cy, score = all_keypoints[int(keypoint_id), 0:3]
                cx = cx + 0.5
                cy = cy + 0.5
                visibility = 1
            keypoints[to_coco_map[position_id] * 3 + 0] = cx
            keypoints[to_coco_map[position_id] * 3 + 1] = cy
            keypoints[to_coco_map[position_id] * 3 + 2] = visibility
        coco_keypoints.append(keypoints)
        scores.append(person_score * max(0, (pose_entries[n][-1] - 1)))  # -1 for 'neck'
    return coco_keypoints, scores


def run_coco_eval(gt_file_path, dt_file_path):
    annotation_type = 'keypoints'
    print('Running test for {} results.'.format(annotation_type))

    coco_gt = COCO(gt_file_path)
    coco_dt = coco_gt.loadRes(dt_file_path)

    result = COCOeval(coco_gt, coco_dt, annotation_type)
    result.evaluate()
    result.accumulate()
    result.summarize()


class Evaluator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.saved_dir = os.path.join(cfg['COMMON']['saved_dir'], 'lw_pose_tf')
        self.model = tf.keras.models.load_model(cfg['MODEL']['saved_model_dir'], compile=False)

        self.dataset = CocoValDataset(cfg['DATASET']['annotations'],
                                      cfg['DATASET']['images_dir'],
                                      cfg['DATASET']['dataset_type'])
        if cfg["DATASET"]['dataset_type'] == 'train':
            self.json = os.path.join(cfg['DATASET']['json_dir'], 'person_keypoints_train2017.json')
        elif cfg["DATASET"]['dataset_type'] == "val":
            self.json = os.path.join(cfg['DATASET']['json_dir'], 'person_keypoints_val2017.json')
        else:
            raise ValueError("Don't support other types")
        self.images_dir, self.annotations = self.dataset.get_images_and_annotations()
        self.input_size = cfg['MODEL']['input_size']
        self.stride = cfg['MODEL']['stride']
        self.output = cfg['COMMON']['output']

    def evaluate(self):
        coco_result = []
        for sample in self.dataset.get_index():
            idx = sample.numpy()
            annotation = self.annotations[idx[0]]
            image_path = os.path.join(self.images_dir, annotation['img_paths'])

            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            heatmaps, pafs, scale = self.infer(image)

            total_keypoints_num = 0
            all_keypoints_by_type = []

            for kp_idx in range(18):
                total_keypoints_num = extract_keypoints(heatmaps[:, :, kp_idx], all_keypoints_by_type,
                                                        total_keypoints_num)
            pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs)
            for kpt_id in range(all_keypoints.shape[0]):
                all_keypoints[kpt_id, 0] = all_keypoints[kpt_id, 0] / scale[0]
                all_keypoints[kpt_id, 1] = all_keypoints[kpt_id, 1] / scale[1]
            coco_keypoints, scores = convert_to_coco_format(pose_entries, all_keypoints)
            file_name = annotation['img_paths']
            image_id = int(file_name[0:file_name.rfind('.')])
            for _id in range(len(coco_keypoints)):
                coco_result.append({
                    'image_id': image_id,
                    'category_id': 1,  # person
                    'keypoints': coco_keypoints[_id],
                    'score': scores[_id]
                })

        with open(self.output, 'w') as f:
            json.dump(coco_result, f, indent=4)

        run_coco_eval(self.json, self.output)

    def infer(self, image):
        height, width, _ = image.shape
        scale = (self.input_size / width, self.input_size / height)

        scaled_img = cv2.resize(image, (0, 0), fx=scale[0], fy=scale[1],
                                interpolation=cv2.INTER_CUBIC)
        scaled_img = (scaled_img - 128) / 256.0
        scaled_img = np.expand_dims(scaled_img, axis=0)
        stages_output = self.model(scaled_img, training=False)

        heatmaps = np.squeeze(stages_output[-1][0].numpy())
        heatmaps = cv2.resize(heatmaps, (0, 0), fx=self.stride, fy=self.stride,
                              interpolation=cv2.INTER_CUBIC)

        pafs = np.squeeze(stages_output[-1][1].numpy())
        pafs = cv2.resize(pafs, (0, 0), fx=self.stride, fy=self.stride,
                          interpolation=cv2.INTER_CUBIC)

        return heatmaps, pafs, scale







