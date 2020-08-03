import sys
import os
import cv2
import numpy as np

from src import datasets

from src.utils.keypoints_grouping import extract_keypoints, group_keypoints, Pose


def visualize(dataset, dataset_type):
    if dataset_type == 'coco':
        num_keypoints = 18
    elif dataset_type == 'kinect':
        num_keypoints = 32
    else:
        raise NotImplementedError
    dataset = dataset.take(1)
    stride = 8
    upsample_ratio = 4
    for batch in dataset:
        image = batch[0][0].numpy() * 255.0 + 128.0
        image = image.astype(np.uint8)
        heatmaps = batch[1][0][0].numpy()
        heatmaps = cv2.resize(heatmaps, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)
        pafs = batch[1][1][0].numpy()
        pafs = cv2.resize(pafs, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)
        total_keypoints_num = 0
        all_keypoints_by_type = []

        for kpt_idx in range(num_keypoints):  # 19th for bg
            total_keypoints_num = extract_keypoints(heatmaps[:, :, kpt_idx], all_keypoints_by_type,
                                                    total_keypoints_num)

        pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs, pose_entry_size=num_keypoints+2,
                                                      demo=True, dataset=dataset_type)

        for kpt_id in range(all_keypoints.shape[0]):
            all_keypoints[kpt_id, 0] = all_keypoints[kpt_id, 0] * stride / upsample_ratio
            all_keypoints[kpt_id, 1] = all_keypoints[kpt_id, 1] * stride / upsample_ratio
        current_poses = []
        for n in range(len(pose_entries)):
            if len(pose_entries[n]) == 0:
                continue
            pose_keypoints = np.ones((num_keypoints, 2), dtype=np.int32) * -1
            for kpt_id in range(num_keypoints):
                if pose_entries[n][kpt_id] != -1.0:  # keypoint was found
                    pose_keypoints[kpt_id, 0] = int(all_keypoints[int(pose_entries[n][kpt_id]), 0])
                    pose_keypoints[kpt_id, 1] = int(all_keypoints[int(pose_entries[n][kpt_id]), 1])

            pose = Pose(pose_keypoints, pose_entries[n][num_keypoints], dataset=dataset_type)

            current_poses.append(pose)

        for pose in current_poses:
            pose.draw(image)

        cv2.imwrite('/home/hoangbm/lightweight_openpose_tensorflow/tests/test.jpg', image)


if __name__ == '__main__':
    import yaml
    cfg = sys.argv[1]
    with open(cfg, 'r') as fp:
        cfg = yaml.full_load(fp)

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    train_dataset = datasets.__dict__[cfg['name']](annotations_dir=cfg['annotation_dir'],
                                                   images_dir=cfg['image_dir'],
                                                   input_size=cfg['input_size'],
                                                   stride=cfg['stride'], sigma=cfg['sigma'],
                                                   paf_thickness=cfg['paf_thickness'], is_training=True)

    ds = train_dataset.get_dataset(cfg['batch_size'])

    visualize(ds, cfg['name'])
