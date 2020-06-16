from abc import abstractmethod
import cv2
import numpy as np
from src.utils.keypoints_grouping import extract_keypoints, group_keypoints, Pose


class BaseServing:
    def __init__(self, cfg):
        self.cfg = cfg
        self.input_size = cfg['MODEL']['input_size']
        self.stride = self.cfg['MODEL']['stride']
        self.num_keypoints = Pose.num_kpts

    @abstractmethod
    def infer(self, image):
        pass

    def predict(self, image_path):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        heatmaps, pafs, scale = self.infer(image)

        total_keypoints_num = 0
        all_keypoints_by_type = []

        for kpt_idx in range(self.num_keypoints):  # 19th for bg
            total_keypoints_num = extract_keypoints(heatmaps[:, :, kpt_idx], all_keypoints_by_type,
                                                    total_keypoints_num)

        pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs, demo=True)
        for kpt_id in range(all_keypoints.shape[0]):
            all_keypoints[kpt_id, 0] = all_keypoints[kpt_id, 0] / scale[0]
            all_keypoints[kpt_id, 1] = all_keypoints[kpt_id, 1] / scale[1]

        current_poses = []
        for n in range(len(pose_entries)):
            if len(pose_entries[n]) == 0:
                continue
            pose_keypoints = np.ones((self.num_keypoints, 2), dtype=np.int32) * -1
            for kpt_id in range(self.num_keypoints):
                if pose_entries[n][kpt_id] != -1.0:  # keypoint was found
                    pose_keypoints[kpt_id, 0] = int(all_keypoints[int(pose_entries[n][kpt_id]), 0])
                    pose_keypoints[kpt_id, 1] = int(all_keypoints[int(pose_entries[n][kpt_id]), 1])
            pose = Pose(pose_keypoints, pose_entries[n][18])
            current_poses.append(pose)
        for pose in current_poses:
            pose.draw(image)
        cv2.imwrite(self.cfg['COMMON']['result'], image)
        # cv2.imshow('Result', image)
        # if cv2.waitKey(0) == ord('q'):
        #     cv2.destroyAllWindows()