import numpy as np
import random
import copy
import imgaug as ia
from imgaug import augmenters as iaa
from imgaug import parameters as iap


def augmentate_data(image, all_keypoints):
    is_flip = [random.randint(0, 1), random.randint(0, 1)]

    transfomation_seq = iaa.Sequential([
        iaa.Multiply((0.7, 1.5)),
        iaa.Grayscale(iap.Choice(a=[0, 1], p=[0.8, 0.2]), from_colorspace='BGR'),
        iaa.Fliplr(is_flip[0]),
        iaa.Flipud(is_flip[1]),
        iaa.Affine(rotate=(-15, 15), scale=(0.8, 1.2), mode='constant')
    ])
    seq_set = transfomation_seq.to_deterministic()
    augmentated_image = seq_set.augment_image(image)

    # keypoints transformation
    all_keypoints = np.array(all_keypoints)
    kps = ia.KeypointsOnImage([], shape=image.shape)
    for i in range(all_keypoints.shape[0]):
        single_person_keypoints = all_keypoints[i]
        for j in range(all_keypoints.shape[1]):
            joint = single_person_keypoints[j]
            kps.keypoints.append(ia.Keypoint(x=joint[0], y=joint[1]))

    ori_kps = copy.copy(all_keypoints)
    aug_kps = seq_set.augment_keypoints(kps)
    keypoints = []

    for aug_kp in aug_kps:
        keypoints.append([aug_kp.x, aug_kp.y, 1])
    keypoints = np.reshape(np.asarray(keypoints), newshape=ori_kps.shape)

    for i in range(keypoints.shape[0]):
        for j in range(keypoints.shape[1]):
            keypoints[i, j, 2] = ori_kps[i, j, 2]

    # flip keypoints
    right = [2, 3, 4, 8, 9, 10, 14, 16]
    left = [5, 6, 7, 11, 12, 13, 15, 17]
    if is_flip[0]:
        for i in range(ori_kps.shape[0]):
            for l_idx, r_idx in zip(left, right):
                right_point = copy.copy(keypoints[i][r_idx])
                keypoints[i][r_idx] = keypoints[i][l_idx]
                keypoints[i][l_idx] = right_point
    return augmentated_image, keypoints
