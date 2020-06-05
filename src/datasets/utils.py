import math
from pycocotools import mask as msk


def set_paf(paf_map, x_a, y_a, x_b, y_b, stride, thickness):
    x_a /= stride
    y_a /= stride
    x_b /= stride
    y_b /= stride
    x_ba = x_b - x_a
    y_ba = y_b - y_a
    _, h_map, w_map = paf_map.shape
    x_min = int(max(min(x_a, x_b) - thickness, 0))
    x_max = int(min(max(x_a, x_b) + thickness, w_map))
    y_min = int(max(min(y_a, y_b) - thickness, 0))
    y_max = int(min(max(y_a, y_b) + thickness, h_map))
    norm_ba = (x_ba * x_ba + y_ba * y_ba) ** 0.5
    if norm_ba < 1e-7:  # Same points, no paf
        return
    x_ba /= norm_ba
    y_ba /= norm_ba

    for y in range(y_min, y_max):
        for x in range(x_min, x_max):
            x_ca = x - x_a
            y_ca = y - y_a
            d = math.fabs(x_ca * y_ba - y_ca * x_ba)
            if d <= thickness:
                paf_map[0, y, x] = x_ba
                paf_map[1, y, x] = y_ba


def set_gaussian(keypoint_map, x, y, stride, sigma):
    n_sigma = 4
    tl = [int(x - n_sigma * sigma), int(y - n_sigma * sigma)]
    tl[0] = max(tl[0], 0)
    tl[1] = max(tl[1], 0)

    br = [int(x + n_sigma * sigma), int(y + n_sigma * sigma)]
    map_h, map_w = keypoint_map.shape
    br[0] = min(br[0], map_w * stride)
    br[1] = min(br[1], map_h * stride)

    shift = stride / 2 - 0.5
    for map_y in range(tl[1] // stride, br[1] // stride):
        for map_x in range(tl[0] // stride, br[0] // stride):
            d2 = (map_x * stride + shift - x) * (map_x * stride + shift - x) + \
                 (map_y * stride + shift - y) * (map_y * stride + shift - y)
            exponent = d2 / 2 / sigma / sigma
            if exponent > 4.6052:  # threshold, ln(100), ~0.01
                continue
            keypoint_map[map_y, map_x] += math.exp(-exponent)
            if keypoint_map[map_y, map_x] > 1:
                keypoint_map[map_y, map_x] = 1


def get_mask(segmentations, mask):
    for segmentation in segmentations:
        rle = msk.frPyObjects(segmentation, mask.shape[0], mask.shape[1])
        mask[msk.decode(rle) > 0.5] = 0
    return mask


