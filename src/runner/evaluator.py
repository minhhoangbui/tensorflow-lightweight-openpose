import cv2
import json
import os
from abc import abstractmethod
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import MNN
import tensorflow as tf
from src.utils.keypoints_grouping import extract_keypoints, group_keypoints
from src.datasets.coco import CocoValDataset


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


class BaseEvaluator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.saved_dir = os.path.join(cfg['COMMON']['saved_dir'], 'lw_pose_tf')
        self.dataset = CocoValDataset(cfg['DATASET']['annotations'],
                                      cfg['DATASET']['images_dir'],
                                      cfg['DATASET']['dataset_type'])
        self.images_dir = self.dataset.images_dir

        self.input_size = cfg['MODEL']['input_size']
        self.stride = cfg['MODEL']['stride']
        self.output = cfg['COMMON']['output']

    def preprocess_image(self, image):
        height, width, _ = image.shape
        scale = (self.input_size / width, self.input_size / height)

        scaled_image = cv2.resize(image, (0, 0), fx=scale[0], fy=scale[1],
                                  interpolation=cv2.INTER_CUBIC)
        scaled_image = (scaled_image - 128) / 255.0
        scaled_image = np.expand_dims(scaled_image, axis=0)
        return scaled_image, scale

    def evaluate(self):
        coco_result = []
        for file_name in self.dataset.sample:

            image_path = os.path.join(self.images_dir, file_name)
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

        run_coco_eval(self.dataset.annotation_file, self.output)

    @abstractmethod
    def infer(self, image):
        pass


class TFEvaluator(BaseEvaluator):
    def __init__(self, cfg):
        super(TFEvaluator, self).__init__(cfg)
        self.model = tf.keras.models.load_model(cfg['MODEL']['saved_model_dir'], compile=False)

    def infer(self, image):
        scaled_image, scale = self.preprocess_image(image)
        stages_output = self.model(scaled_image, training=False)

        heatmaps = np.squeeze(stages_output[-1][0].numpy())
        heatmaps = cv2.resize(heatmaps, (0, 0), fx=self.stride, fy=self.stride,
                              interpolation=cv2.INTER_CUBIC)

        pafs = np.squeeze(stages_output[-1][1].numpy())
        pafs = cv2.resize(pafs, (0, 0), fx=self.stride, fy=self.stride,
                          interpolation=cv2.INTER_CUBIC)

        return heatmaps, pafs, scale


class TFLiteEvaluator(BaseEvaluator):
    def __init__(self, cfg):
        super(TFLiteEvaluator, self).__init__(cfg)
        self.interpreter = tf.lite.Interpreter(cfg['MODEL']['tflite'])
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def infer(self, image):

        scaled_image, scale = self.preprocess_image(image)

        scaled_image = np.float32(scaled_image)
        self.interpreter.set_tensor(self.input_details[0]['index'], scaled_image)
        self.interpreter.invoke()
        heatmaps = np.squeeze(self.interpreter.get_tensor(self.output_details[-2]['index']))
        heatmaps = cv2.resize(heatmaps, (0, 0),
                              fx=self.stride, fy=self.stride,
                              interpolation=cv2.INTER_CUBIC)

        pafs = np.squeeze(self.interpreter.get_tensor(self.output_details[-1]['index']))
        pafs = cv2.resize(pafs, (0, 0),
                          fx=self.stride, fy=self.stride,
                          interpolation=cv2.INTER_CUBIC)
        return heatmaps, pafs, scale


class MNNEvaluator(BaseEvaluator):
    def __init__(self, cfg):
        super(MNNEvaluator, self).__init__(cfg)
        self.interpreter = MNN.Interpreter(cfg['MODEL']['mnn'])
        self.session = self.interpreter.createSession()
        self.input_tensor = self.interpreter.getSessionInput(self.session)
        if cfg['MODEL']['quantized']:
            self.heatmaps_tensor = self.interpreter.getSessionOutput(self.session, 'Int8ToFloat119')
            self.pafs_tensor = self.interpreter.getSessionOutput(self.session, 'Int8ToFloat124')
        else:
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
        scaled_image, scale = self.preprocess_image(image)

        scaled_image = np.float32(scaled_image)
        tmp_input = MNN.Tensor((1, self.input_size, self.input_size, 3),
                               MNN.Halide_Type_Float, scaled_image,
                               MNN.Tensor_DimensionType_Tensorflow)

        self.input_tensor.copyFrom(tmp_input)
        self.interpreter.runSession(self.session)

        self.heatmaps_tensor.copyToHostTensor(self.heatmaps_output)
        self.pafs_tensor.copyToHostTensor(self.pafs_output)

        heatmaps = np.squeeze(self.heatmaps_output.getData())
        pafs = np.squeeze(self.pafs_output.getData())

        heatmaps = heatmaps.transpose((1, 2, 0))
        pafs = pafs.transpose((1, 2, 0))

        heatmaps = cv2.resize(heatmaps, (0, 0),
                              fx=self.stride, fy=self.stride,
                              interpolation=cv2.INTER_CUBIC)

        pafs = cv2.resize(pafs, (0, 0),
                          fx=self.stride, fy=self.stride,
                          interpolation=cv2.INTER_CUBIC)

        return heatmaps, pafs, scale










