import sys
import yaml
import os
import tensorflow as tf
import shutil
from tensorflow.lite.python.util import run_graph_optimizations
from src.models.lightweight_openpose import LightWeightOpenPose


def export_saved_model(cfg):
    format_ = cfg['EXPORT']['format']
    assert format_ in ['sm', 'tf'], "Not supported format"
    model = LightWeightOpenPose(num_channels=cfg['MODEL']['num_channels'],
                                num_refinement_stages=cfg['MODEL']['num_stages'])
    model.build((1, cfg['MODEL']['input_size'], cfg['MODEL']['input_size'], 3))
    model.summary()
    checkpoint = tf.train.Checkpoint(epoch=tf.Variable(0), model=model)
    status = checkpoint.restore(tf.train.latest_checkpoint(cfg['EXPORT']['checkpoint']))
    status.assert_existing_objects_matched()
    print("Convert checkpoint at epoch %d" % int(checkpoint.epoch))
    if os.path.exists(cfg['EXPORT']['saved_model']):
        shutil.rmtree(cfg['EXPORT']['saved_model'])
    if format_ == 'sm':
        tf.saved_model.save(model, cfg['EXPORT']['saved_model'])
    else:
        model._set_inputs(inputs=tf.ones((1, cfg['MODEL']['input_size'], cfg['MODEL']['input_size'], 3)))
        model.save(cfg['EXPORT']['saved_model'], save_format='tf')


def load_saved_model(cfg):
    loaded = tf.saved_model.load(cfg['EXPORT']['saved_model'])
    infer = loaded.signatures["serving_default"]
    print(infer.structured_outputs)


def export_tflite(cfg):
    converter = tf.lite.TFLiteConverter.from_saved_model(cfg['EXPORT']['saved_model'])
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                            tf.lite.OpsSet.SELECT_TF_OPS]
    tflite_model = converter.convert()
    with tf.io.gfile.GFile(cfg['EXPORT']["tf_lite"], 'wb') as fp:
        fp.write(tflite_model)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = "4"
    available_gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in available_gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    config = sys.argv[1]
    with open(config, 'r') as fp:
        cfg = yaml.full_load(fp)
    # export_saved_model(cfg)
    # load_saved_model(cfg)
    export_tflite(cfg)