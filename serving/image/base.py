from abc import abstractmethod
import cv2
import numpy as np
from src.utils.keypoints_grouping import extract_keypoints, group_keypoints, Pose


class BaseServing:
    def __init__(self, cfg):
        self.cfg = cfg
        self.input_size = cfg['MODEL']['input_size']
        self.stride = cfg['MODEL']['stride']

    @abstractmethod
    def infer(self, image):
        pass

    def preprocess_image(self, image):
        height, width, _ = image.shape
        scale = (self.input_size / width, self.input_size / height)

        scaled_image = cv2.resize(image, (0, 0), fx=scale[0], fy=scale[1],
                                  interpolation=cv2.INTER_CUBIC)

        scaled_image = (scaled_image - 128) / 255.0
        scaled_image = np.expand_dims(scaled_image, axis=0)
        scaled_image = np.float32(scaled_image)
        return scaled_image, scale

    def predict(self, image_path):
        if self.cfg['DATASET']['name'] == 'coco':
            num_keypoints = 18
        elif self.cfg['DATASET']['name'] == 'kinect':
            num_keypoints = 32
        else:
            raise NotImplementedError
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        heatmaps, pafs, scale = self.infer(image)

        total_keypoints_num = 0
        all_keypoints_by_type = []

        for kpt_idx in range(num_keypoints):
            total_keypoints_num = extract_keypoints(heatmaps[:, :, kpt_idx], all_keypoints_by_type,
                                                    total_keypoints_num)

        pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs, pose_entry_size=num_keypoints+2,
                                                      demo=True, dataset=self.cfg['DATASET']['name'])
        for kpt_id in range(all_keypoints.shape[0]):
            all_keypoints[kpt_id, 0] = all_keypoints[kpt_id, 0] / scale[0]
            all_keypoints[kpt_id, 1] = all_keypoints[kpt_id, 1] / scale[1]

        current_poses = []
        for n in range(len(pose_entries)):
            if len(pose_entries[n]) == 0:
                continue
            pose_keypoints = np.ones((num_keypoints, 2), dtype=np.int32) * -1
            for kpt_id in range(num_keypoints):
                if pose_entries[n][kpt_id] != -1.0:  # keypoint was found
                    pose_keypoints[kpt_id, 0] = int(all_keypoints[int(pose_entries[n][kpt_id]), 0])
                    pose_keypoints[kpt_id, 1] = int(all_keypoints[int(pose_entries[n][kpt_id]), 1])

            pose = Pose(pose_keypoints, pose_entries[n][num_keypoints], dataset=self.cfg['DATASET']['name'])

            current_poses.append(pose)
        for pose in current_poses:
            pose.draw(image)
        cv2.imwrite(self.cfg['COMMON']['result'], image)
        # cv2.imshow('Result', image)
        # if cv2.waitKey(0) == ord('q'):
        #     cv2.destroyAllWindows()