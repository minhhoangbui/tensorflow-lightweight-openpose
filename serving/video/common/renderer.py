from collections import defaultdict
from functools import partial

import cv2
import numpy as np

from src.utils.keypoints_grouping import extract_keypoints, group_keypoints, Pose
from .meters import WindowAverageMeter


def decode_output(outputs, scale, meta):
    if meta['dataset'] == 'coco':
        num_keypoints = 18
    elif meta['dataset'] == 'kinect':
        num_keypoints = 32
    else:
        raise NotImplementedError
    heatmaps = np.squeeze(outputs[meta['output_names'][0]].buffer)
    pafs = np.squeeze(outputs[meta['output_names'][1]].buffer)
    heatmaps = heatmaps.transpose((1, 2, 0))
    pafs = pafs.transpose((1, 2, 0))

    heatmaps = cv2.resize(heatmaps, (0, 0),
                          fx=meta['stride'], fy=meta['stride'],
                          interpolation=cv2.INTER_CUBIC)

    pafs = cv2.resize(pafs, (0, 0),
                      fx=meta['stride'], fy=meta['stride'],
                      interpolation=cv2.INTER_CUBIC)
    total_keypoints_num = 0
    all_keypoints_by_type = []

    for kpt_idx in range(num_keypoints):
        total_keypoints_num = extract_keypoints(heatmaps[:, :, kpt_idx], all_keypoints_by_type,
                                                total_keypoints_num)

    pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs, pose_entry_size=num_keypoints + 2,
                                                  demo=True, dataset=meta['dataset'])

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

        pose = Pose(pose_keypoints, pose_entries[n][num_keypoints], dataset=meta['dataset'])

        current_poses.append(pose)
    return current_poses


class ResultRenderer:
    def __init__(self, output_dir, meta, num_requests=8):
        self.output = output_dir
        self.meta = meta
        self.meters = defaultdict(partial(WindowAverageMeter, num_requests))

    def update_timers(self, timers):
        self.meters['estimation'].update(timers['estimation'])
        return self.meters['estimation'].avg

    def render_frame(self, frame, outputs, scale, timers, frame_id):
        inference_time = self.update_timers(timers)

        current_poses = decode_output(outputs, scale, self.meta)
        print(f'Frame {frame_id} -- {inference_time:.2f}')
        w, h, _ = frame.shape

        if not hasattr(self, 'writer') and self.output:
            self.writer = cv2.VideoWriter(self.output,
                                          cv2.VideoWriter_fourcc('M', 'J', 'P', 'Q'), 10, (w, h))
        for pose in current_poses:
            pose.draw(frame)
        if not self.output:
            cv2.imshow('Pose Estimation', frame)
            key = cv2.waitKey(1) & 0xFF
            if key in {ord('q'), ord('Q'), 27}:
                return -1
        else:
            self.writer.write(frame)


