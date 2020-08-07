import os.path as osp
import threading
from collections import deque
from abc import abstractmethod
import cv2
import numpy as np
import monitors
import sys
import logging
from openvino.inference_engine import IECore

logging.basicConfig(format='[ %(levelname)s ] %(message)s', level=logging.INFO, stream=sys.stdout)
log = logging.getLogger()


class Model:
    def __init__(self, xml_file_path, device='CPU',
                 plugin_config={}, max_num_requests=1,
                 results=None, caught_exceptions=None):
        self.ie = IECore()
        log.info('Reading network from IR...')
        bin_file_path = osp.splitext(xml_file_path)[0] + '.bin'
        self.net = ie.read_network(model=xml_file_path, weights=bin_file_path)

        log.info('Loading network to plugin...')
        if 'CPU' in device:
            self.check_cpu_support(ie, self.net)
        self.max_num_requests = max_num_requests
        self.exec_net = ie.load_network(network=self.net, device_name=device, config=plugin_config,
                                        num_requests=max_num_requests)

        self.requests = self.exec_net.requests
        self.empty_requests = deque(self.requests)
        self.completed_request_results = results if results is not None else []
        self.callback_exceptions = caught_exceptions if caught_exceptions is not None else {}
        self.event = threading.Event()

    @staticmethod
    def check_cpu_support(ie, net):
        log.info('Check that all layers are supported...')
        supported_layers = ie.query_network(net, 'CPU')
        not_supported_layers = [l for l in net.layers.keys() if l not in supported_layers]
        if len(not_supported_layers) != 0:
            unsupported_info = '\n\t'.join('{} ({} with params {})'.format(layer_id,
                                                                           net.layers[layer_id].type,
                                                                           str(net.layers[layer_id].params))
                                           for layer_id in not_supported_layers)
            log.warning('Following layers are not supported '
                        'by the CPU plugin:\n\t{}'.format(unsupported_info))
            log.warning('Please try to specify cpu extensions library path.')
            raise ValueError('Some of the layers are not supported.')

    def unify_inputs(self, inputs):
        if not isinstance(inputs, dict):
            inputs_dict = {next(iter(self.net.input_info)): inputs}
        else:
            inputs_dict = inputs
        return inputs_dict

    @abstractmethod
    def preprocess(self, inputs):
        pass

    @abstractmethod
    def postprocess(self, outputs, meta):
        pass

    def inference_completion_callback(self, status, callback_args):
        request, frame_id, frame_meta = callback_args
        try:
            if status != 0:
                raise RuntimeError('Infer Request has returned status code {}'.format(status))
            raw_outputs = {key: blob.buffer for key, blob in request.output_blobs.items()}
            self.completed_request_results[frame_id] = (frame_meta, raw_outputs)
            self.empty_requests.append(request)
        except Exception as e:
            self.callback_exceptions.append(e)
        self.event.set()

    def __call__(self, inputs, id, meta):
        request = self.empty_requests.popleft()
        inputs = self.unify_inputs(inputs)
        inputs, preprocessing_meta = self.preprocess(inputs)
        meta.update(preprocessing_meta)
        request.set_completion_callback(py_callback=self.inference_completion_callback,
                                        py_data=(request, id, meta))
        self.event.clear()
        request.async_infer(inputs=inputs)

    def await_all(self):
        for request in self.exec_net.requests:
            request.wait()

    def await_any(self):
        self.event.wait()


class PoseEstimator(Model):
    def __init__(self, *args, **kwargs):
        super(PoseEstimator, self).__init__(*args, **kwargs)
        self.image_blob_name = self._get_inputs()
        self.stride = 8
        self.output_blob_name = list(net.outputs.keys())
        self.n, self.c, self.h, self.w = self.net.input_info[self.image_blob_name].input_data.shape
        assert self.n == 1, 'Only batch size == 1 is supported.'

    def _get_inputs(self):
        assert len(self.net.input_info) == 1, "Expected 1 input blob"
        image_blob_name = None
        for blob_name, blob in self.net.input_info.items():
            if len(blob.input_data.shape) == 4:
                image_blob_name = blob_name
            else:
                raise RuntimeError('Unsupported {}D input layer "{}". Only 2D and 4D input layers are supported'
                                   .format(len(blob.shape), blob_name))
        if image_blob_name is None:
            raise RuntimeError('Failed to identify the input for the image.')
        return image_blob_name

    def preprocess(self, inputs):
        image = inputs[self.image_blob_name]
        height, width, _ = image.shape
        scale = (self.w / width, self.h / height)
        meta = {'scale': scale}
        scaled_image = cv2.resize(image, (0, 0), fx=scale[0], fy=scale[1],
                                  interpolation=cv2.INTER_CUBIC)
        scaled_image = (scaled_image - 128) / 255.0
        scaled_image = np.transpose(scaled_image, (2, 0, 1))
        scaled_image = np.expand_dims(scaled_image, axis=0)
        return scaled_image, meta

    def postprocess(self, outputs, meta):
        heatmaps = np.squeeze(outputs[self.output_blob_name[0]])
        pafs = np.squeeze(outputs[self.output_blob_name[1]])
        heatmaps = heatmaps.transpose((1, 2, 0))
        pafs = pafs.transpose((1, 2, 0))
        heatmaps = cv2.resize(heatmaps, (0, 0),
                              fx=self.stride, fy=self.stride,
                              interpolation=cv2.INTER_CUBIC)
        pafs = cv2.resize(pafs, (0, 0),
                          fx=self.stride, fy=self.stride,
                          interpolation=cv2.INTER_CUBIC)
        return heatmaps, pafs, meta['scale']


def get_plugin_configs(device, num_streams, num_threads):
    config_user_specified = {}
    config_min_latency = {}

    devices_nstreams = {}
    if num_streams:
        devices_nstreams = {device: num_streams for device in ['CPU', 'GPU'] if device in device} \
            if num_streams.isdigit() \
            else dict(device.split(':', 1) for device in num_streams.split(','))

    if 'CPU' in device:
        if num_threads is not None:
            config_user_specified['CPU_THREADS_NUM'] = str(num_threads)
        if 'CPU' in devices_nstreams:
            config_user_specified['CPU_THROUGHPUT_STREAMS'] = devices_nstreams['CPU'] \
                if int(devices_nstreams['CPU']) > 0 \
                else 'CPU_THROUGHPUT_AUTO'

        config_min_latency['CPU_THROUGHPUT_STREAMS'] = '1'

    if 'GPU' in device:
        if 'GPU' in devices_nstreams:
            config_user_specified['GPU_THROUGHPUT_STREAMS'] = devices_nstreams['GPU'] \
                if int(devices_nstreams['GPU']) > 0 \
                else 'GPU_THROUGHPUT_AUTO'

        config_min_latency['GPU_THROUGHPUT_STREAMS'] = '1'

    return config_user_specified, config_min_latency

device = 'CPU'
ie = IECore()
xml_file_path = '/home/hoangbm/model_library/lw-pose/openvino/lw_pose_16.xml'
bin_file_path = osp.splitext(xml_file_path)[0] + '.bin'

config_user_specified, config_min_latency = get_plugin_configs(device, '', 3)

net = ie.read_network(model=xml_file_path, weights=bin_file_path)
exec_net = ie.load_network(net, device, config=config_min_latency, num_requests=3)
requests = exec_net.requests
print(requests)