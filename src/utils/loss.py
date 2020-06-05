import tensorflow as tf


def get_loss(target, outputs, mask):
    heatmap_loss = 0
    paf_loss = 0
    for output in outputs:
        heatmap_loss += tf.nn.l2_loss((output[0] - target[0]) * mask[0])
        paf_loss += tf.nn.l2_loss((output[1] - target[1]) * mask[1])
    return (heatmap_loss + paf_loss) / len(outputs) / len(target)