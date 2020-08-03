import math
import numpy as np
import cv2
from operator import itemgetter
from src.utils.filters import OneEuroFilter


def linspace2d(start, stop, n=10):
    points = 1 / (n - 1) * (stop - start)
    return points[:, None] * np.arange(n) + start[:, None]


def extract_keypoints(heatmap, all_keypoints, total_keypoint_num):
    heatmap[heatmap < 0.1] = 0
    heatmap_with_borders = np.pad(heatmap, [(2, 2), (2, 2)], mode='constant')
    heatmap_center = heatmap_with_borders[1:heatmap_with_borders.shape[0]-1, 1:heatmap_with_borders.shape[1]-1]
    heatmap_left = heatmap_with_borders[1:heatmap_with_borders.shape[0]-1, 2:heatmap_with_borders.shape[1]]
    heatmap_right = heatmap_with_borders[1:heatmap_with_borders.shape[0]-1, 0:heatmap_with_borders.shape[1]-2]
    heatmap_up = heatmap_with_borders[2:heatmap_with_borders.shape[0], 1:heatmap_with_borders.shape[1]-1]
    heatmap_down = heatmap_with_borders[0:heatmap_with_borders.shape[0]-2, 1:heatmap_with_borders.shape[1]-1]

    heatmap_peaks = (heatmap_center > heatmap_left) &\
                    (heatmap_center > heatmap_right) &\
                    (heatmap_center > heatmap_up) &\
                    (heatmap_center > heatmap_down)
    heatmap_peaks = heatmap_peaks[1:heatmap_center.shape[0]-1, 1:heatmap_center.shape[1]-1]
    keypoints = list(zip(np.nonzero(heatmap_peaks)[1], np.nonzero(heatmap_peaks)[0]))  # (w, h)
    keypoints = sorted(keypoints, key=itemgetter(0))

    suppressed = np.zeros(len(keypoints), np.uint8)
    keypoints_with_score_and_id = []
    keypoint_num = 0
    for i in range(len(keypoints)):
        if suppressed[i]:
            continue
        for j in range(i+1, len(keypoints)):
            if math.sqrt((keypoints[i][0] - keypoints[j][0]) ** 2 +
                         (keypoints[i][1] - keypoints[j][1]) ** 2) < 6:
                suppressed[j] = 1
        keypoint_with_score_and_id = (keypoints[i][0], keypoints[i][1], heatmap[keypoints[i][1], keypoints[i][0]],
                                      total_keypoint_num + keypoint_num)
        keypoints_with_score_and_id.append(keypoint_with_score_and_id)
        keypoint_num += 1
    all_keypoints.append(keypoints_with_score_and_id)
    return keypoint_num + total_keypoint_num


def group_keypoints(all_keypoints_by_type, pafs, pose_entry_size=20, min_paf_score=0.05, demo=False, dataset='coco'):
    if dataset == 'coco':
        from src.datasets.coco import BODY_PARTS_KPT_IDS, BODY_PARTS_PAF_IDS
        ignored = [17, 18]
    elif dataset == 'kinect':
        from src.datasets.kinect import BODY_PARTS_PAF_IDS, BODY_PARTS_KPT_IDS
        ignored = [31, 32]
    else:
        raise ModuleNotFoundError
    pose_entries = []
    all_keypoints = np.array([item for sublist in all_keypoints_by_type for item in sublist])

    for part_id in range(len(BODY_PARTS_PAF_IDS)):
        part_pafs = pafs[:, :, BODY_PARTS_PAF_IDS[part_id]]
        kpts_a = all_keypoints_by_type[BODY_PARTS_KPT_IDS[part_id][0]]
        kpts_b = all_keypoints_by_type[BODY_PARTS_KPT_IDS[part_id][1]]
        num_kpts_a = len(kpts_a)
        num_kpts_b = len(kpts_b)
        kpt_a_id = BODY_PARTS_KPT_IDS[part_id][0]
        kpt_b_id = BODY_PARTS_KPT_IDS[part_id][1]

        if num_kpts_a == 0 and num_kpts_b == 0:  # no keypoints for such body part
            continue
        elif num_kpts_a == 0:  # body part has just 'b' keypoints
            for i in range(num_kpts_b):
                num = 0
                for j in range(len(pose_entries)):  # check if already in some pose, was added by another body part
                    if pose_entries[j][kpt_b_id] == kpts_b[i][3]:
                        num += 1
                        break
                if num == 0:
                    pose_entry = np.ones(pose_entry_size) * -1
                    pose_entry[kpt_b_id] = kpts_b[i][3]  # keypoint idx
                    pose_entry[-1] = 1                   # num keypoints in pose
                    pose_entry[-2] = kpts_b[i][2]        # pose score
                    pose_entries.append(pose_entry)
            continue
        elif num_kpts_b == 0:  # body part has just 'a' keypoints
            for i in range(num_kpts_a):
                num = 0
                for j in range(len(pose_entries)):
                    if pose_entries[j][kpt_a_id] == kpts_a[i][3]:
                        num += 1
                        break
                if num == 0:
                    pose_entry = np.ones(pose_entry_size) * -1
                    pose_entry[kpt_a_id] = kpts_a[i][3]
                    pose_entry[-1] = 1
                    pose_entry[-2] = kpts_a[i][2]
                    pose_entries.append(pose_entry)
            continue

        connections = []
        for i in range(num_kpts_a):
            kpt_a = np.array(kpts_a[i][0:2])
            for j in range(num_kpts_b):
                kpt_b = np.array(kpts_b[j][0:2])
                mid_point = [(), ()]
                mid_point[0] = (int(round((kpt_a[0] + kpt_b[0]) * 0.5)),
                                int(round((kpt_a[1] + kpt_b[1]) * 0.5)))
                mid_point[1] = mid_point[0]

                vec = [kpt_b[0] - kpt_a[0], kpt_b[1] - kpt_a[1]]
                vec_norm = math.sqrt(vec[0] ** 2 + vec[1] ** 2)
                if vec_norm == 0:
                    continue
                vec[0] /= vec_norm
                vec[1] /= vec_norm
                cur_point_score = (vec[0] * part_pafs[mid_point[0][1], mid_point[0][0], 0] +
                                   vec[1] * part_pafs[mid_point[1][1], mid_point[1][0], 1])

                height_n = pafs.shape[0] // 2
                success_ratio = 0
                point_num = 10  # number of points to integration over paf
                if cur_point_score > -100:
                    passed_point_score = 0
                    passed_point_num = 0
                    x, y = linspace2d(kpt_a, kpt_b)
                    for point_idx in range(point_num):
                        if not demo:
                            px = int(round(x[point_idx]))
                            py = int(round(y[point_idx]))
                        else:
                            px = int(x[point_idx])
                            py = int(y[point_idx])
                        paf = part_pafs[py, px, 0:2]
                        cur_point_score = vec[0] * paf[0] + vec[1] * paf[1]
                        if cur_point_score > min_paf_score:
                            passed_point_score += cur_point_score
                            passed_point_num += 1
                    success_ratio = passed_point_num / point_num
                    ratio = 0
                    if passed_point_num > 0:
                        ratio = passed_point_score / passed_point_num
                    ratio += min(height_n / vec_norm - 1, 0)
                if ratio > 0 and success_ratio > 0.8:
                    score_all = ratio + kpts_a[i][2] + kpts_b[j][2]
                    connections.append([i, j, ratio, score_all])
        if len(connections) > 0:
            connections = sorted(connections, key=itemgetter(2), reverse=True)

        num_connections = min(num_kpts_a, num_kpts_b)
        has_kpt_a = np.zeros(num_kpts_a, dtype=np.int32)
        has_kpt_b = np.zeros(num_kpts_b, dtype=np.int32)
        filtered_connections = []
        for row in range(len(connections)):
            if len(filtered_connections) == num_connections:
                break
            i, j, cur_point_score = connections[row][0:3]
            if not has_kpt_a[i] and not has_kpt_b[j]:
                filtered_connections.append([kpts_a[i][3], kpts_b[j][3], cur_point_score])
                has_kpt_a[i] = 1
                has_kpt_b[j] = 1
        connections = filtered_connections
        if len(connections) == 0:
            continue

        if part_id == 0:
            pose_entries = [np.ones(pose_entry_size) * -1 for _ in range(len(connections))]
            for i in range(len(connections)):
                pose_entries[i][BODY_PARTS_KPT_IDS[0][0]] = connections[i][0]
                pose_entries[i][BODY_PARTS_KPT_IDS[0][1]] = connections[i][1]
                pose_entries[i][-1] = 2
                pose_entries[i][-2] = np.sum(all_keypoints[connections[i][0:2], 2]) + connections[i][2]
        elif part_id in ignored:
            kpt_a_id = BODY_PARTS_KPT_IDS[part_id][0]
            kpt_b_id = BODY_PARTS_KPT_IDS[part_id][1]
            for i in range(len(connections)):
                for j in range(len(pose_entries)):
                    if pose_entries[j][kpt_a_id] == connections[i][0] and pose_entries[j][kpt_b_id] == -1:
                        pose_entries[j][kpt_b_id] = connections[i][1]
                    elif pose_entries[j][kpt_b_id] == connections[i][1] and pose_entries[j][kpt_a_id] == -1:
                        pose_entries[j][kpt_a_id] = connections[i][0]
            continue
        else:
            kpt_a_id = BODY_PARTS_KPT_IDS[part_id][0]
            kpt_b_id = BODY_PARTS_KPT_IDS[part_id][1]
            for i in range(len(connections)):
                num = 0
                for j in range(len(pose_entries)):
                    if pose_entries[j][kpt_a_id] == connections[i][0]:
                        pose_entries[j][kpt_b_id] = connections[i][1]
                        num += 1
                        pose_entries[j][-1] += 1
                        pose_entries[j][-2] += all_keypoints[connections[i][1], 2] + connections[i][2]
                if num == 0:
                    pose_entry = np.ones(pose_entry_size) * -1
                    pose_entry[kpt_a_id] = connections[i][0]
                    pose_entry[kpt_b_id] = connections[i][1]
                    pose_entry[-1] = 2
                    pose_entry[-2] = np.sum(all_keypoints[connections[i][0:2], 2]) + connections[i][2]
                    pose_entries.append(pose_entry)

    filtered_entries = []
    for i in range(len(pose_entries)):
        if pose_entries[i][-1] < 3 or (pose_entries[i][-2] / pose_entries[i][-1] < 0.2):
            continue
        filtered_entries.append(pose_entries[i])
    pose_entries = np.asarray(filtered_entries)
    return pose_entries, all_keypoints


class Pose:
    last_id = -1
    color = [0, 0, 255]

    def __init__(self, keypoints, confidence, dataset='coco'):
        super().__init__()
        self.dataset = dataset
        if self.dataset == 'coco':
            self.num_kpts = 18
        elif self.dataset == 'kinect':
            self.num_kpts = 32
        self.keypoints = keypoints
        self.confidence = confidence
        self.bbox = self.get_bbox()
        self.id = None
        self.filters = [[OneEuroFilter(), OneEuroFilter()] for _ in range(self.num_kpts)]

    def get_bbox(self):
        found_keypoints = np.zeros((np.count_nonzero(self.keypoints[:, 0] != -1), 2), dtype=np.int32)
        found_kpt_id = 0
        for kpt_id in range(self.num_kpts):
            if self.keypoints[kpt_id, 0] == -1:
                continue
            found_keypoints[found_kpt_id] = self.keypoints[kpt_id]
            found_kpt_id += 1
        bbox = cv2.boundingRect(found_keypoints)
        return bbox

    def update_id(self, id=None):
        self.id = id
        if self.id is None:
            self.id = Pose.last_id + 1
            Pose.last_id += 1

    def draw(self, img):
        assert self.keypoints.shape == (self.num_kpts, 2)
        if self.dataset == 'coco':
            from src.datasets.coco import BODY_PARTS_PAF_IDS, BODY_PARTS_KPT_IDS
        elif self.dataset == 'kinect':
            from src.datasets.kinect import BODY_PARTS_PAF_IDS, BODY_PARTS_KPT_IDS
        else:
            raise ModuleNotFoundError

        for part_id in range(len(BODY_PARTS_PAF_IDS) - 2):
            kpt_a_id = BODY_PARTS_KPT_IDS[part_id][0]
            global_kpt_a_id = self.keypoints[kpt_a_id, 0]
            if global_kpt_a_id != -1:
                x_a, y_a = self.keypoints[kpt_a_id]
                cv2.circle(img, (int(x_a), int(y_a)), 3, Pose.color, -1)
            kpt_b_id = BODY_PARTS_KPT_IDS[part_id][1]
            global_kpt_b_id = self.keypoints[kpt_b_id, 0]
            if global_kpt_b_id != -1:
                x_b, y_b = self.keypoints[kpt_b_id]
                cv2.circle(img, (int(x_b), int(y_b)), 3, Pose.color, -1)
            if global_kpt_a_id != -1 and global_kpt_b_id != -1:
                cv2.line(img, (int(x_a), int(y_a)), (int(x_b), int(y_b)), Pose.color, 2)
