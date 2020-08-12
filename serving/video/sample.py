import logging
import threading
import os
import sys
from collections import deque
from argparse import ArgumentParser, SUPPRESS
from time import perf_counter
from enum import Enum

import cv2
import numpy as np
from openvino.inference_engine import IECore

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'common'))

import monitors
from src.utils.keypoints_grouping import extract_keypoints, group_keypoints, Pose


logging.basicConfig(format="[ %(levelname)s ] %(message)s", level=logging.INFO, stream=sys.stdout)
log = logging.getLogger()


def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    args.add_argument("-m", "--model", help="Required. Path to an .xml file with a trained model.",
                      required=True, type=str)
    args.add_argument("-i", "--input", help="Required. Path to an image/video file. (Specify 'cam' to work with "
                                            "camera)", required=True, type=str)
    args.add_argument("-l", "--cpu_extension",
                      help="Optional. Required for CPU custom layers. Absolute path to a shared library with "
                           "the kernels implementations.", type=str, default=None)
    args.add_argument("-d", "--device",
                      help="Optional. Specify the target device to infer on; CPU, GPU, FPGA, HDDL or MYRIAD is"
                           " acceptable. The sample will look for a suitable plugin for device specified. "
                           "Default value is CPU", default="CPU", type=str)
    args.add_argument("--labels", help="Optional. Labels mapping file", default=None, type=str)
    args.add_argument("-t", "--prob_threshold", help="Optional. Probability threshold for detections filtering",
                      default=0.5, type=float)
    args.add_argument("-iout", "--iou_threshold", help="Optional. Intersection over union threshold for overlapping "
                                                       "detections filtering", default=0.4, type=float)
    args.add_argument("-r", "--raw_output_message", help="Optional. Output inference results raw values showing",
                      default=False, action="store_true")
    args.add_argument("-nireq", "--num_infer_requests", help="Optional. Number of infer requests",
                      default=1, type=int)
    args.add_argument("-nstreams", "--num_streams",
                      help="Optional. Number of streams to use for inference on the CPU or/and GPU in throughput mode "
                           "(for HETERO and MULTI device cases use format <device1>:<nstreams1>,<device2>:<nstreams2> "
                           "or just <nstreams>)",
                      default="", type=str)
    args.add_argument("-nthreads", "--number_threads",
                      help="Optional. Number of threads to use for inference on CPU (including HETERO cases)",
                      default=None, type=int)
    args.add_argument("-loop_input", "--loop_input", help="Optional. Iterate over input infinitely",
                      action='store_true')
    args.add_argument("-no_show", "--no_show", help="Optional. Don't show output", action='store_true')
    args.add_argument('-u', '--utilization_monitors', default='', type=str,
                      help='Optional. List of monitors to show initially.')

    return parser


class Modes(Enum):
    USER_SPECIFIED = 0
    MIN_LATENCY = 1


class Mode():
    def __init__(self, value):
        self.current = value

    def get_other(self):
        return Modes.MIN_LATENCY if self.current == Modes.USER_SPECIFIED \
            else Modes.USER_SPECIFIED

    def switch(self):
        self.current = self.get_other()


def preprocess_frame(frame, input_height, input_width, nchw_shape):
    height, width, _ = frame.shape
    scale = (input_width / width, input_height / height)
    in_frame = cv2.resize(frame, (0, 0), fx=scale[0], fy=scale[1],
                          interpolation=cv2.INTER_CUBIC)
    in_frame = (in_frame - 128) / 255.0
    if nchw_shape:
        in_frame = in_frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW
    in_frame = np.expand_dims(in_frame, axis=0)
    return in_frame, scale


def postprocess(outputs, scale, output_name, stride, dataset='coco'):
    if dataset == 'coco':
        num_keypoints = 18
    elif dataset == 'kinect':
        num_keypoints = 32
    else:
        raise NotImplementedError
    heatmaps = np.squeeze(outputs[output_name[0]])
    pafs = np.squeeze(outputs[output_name[1]])
    heatmaps = heatmaps.transpose((1, 2, 0))
    pafs = pafs.transpose((1, 2, 0))

    heatmaps = cv2.resize(heatmaps, (0, 0),
                          fx=stride, fy=stride,
                          interpolation=cv2.INTER_CUBIC)
    pafs = cv2.resize(pafs, (0, 0),
                      fx=stride, fy=stride,
                      interpolation=cv2.INTER_CUBIC)

    total_keypoints_num = 0
    all_keypoints_by_type = []

    for kpt_idx in range(num_keypoints):
        total_keypoints_num = extract_keypoints(heatmaps[:, :, kpt_idx], all_keypoints_by_type,
                                                total_keypoints_num)

    pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs, pose_entry_size=num_keypoints + 2,
                                                  demo=True, dataset=dataset)
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

        pose = Pose(pose_keypoints, pose_entries[n][num_keypoints], dataset=dataset)

        current_poses.append(pose)
    return current_poses


def async_callback(status, callback_args):
    request, frame_id, frame_mode, frame, scale, start_time, completed_request_results, empty_requests, \
    mode, event, callback_exceptions = callback_args

    try:
        if status != 0:
            raise RuntimeError('Infer Request has returned status code {}'.format(status))

        completed_request_results[frame_id] = (frame, request.output_blobs, scale, start_time,
                                               frame_mode == mode.current)

        if mode.current == frame_mode:
            empty_requests.append(request)
    except Exception as e:
        callback_exceptions.append(e)

    event.set()


def await_requests_completion(requests):
    for request in requests:
        request.wait()


def main():
    args = build_argparser().parse_args()

    # ------------- 1. Plugin initialization for specified device and load extensions library if specified -------------
    log.info("Creating Inference Engine...")
    ie = IECore()

    config_user_specified = {}
    config_min_latency = {}

    devices_nstreams = {}
    if args.num_streams:
        devices_nstreams = {device: args.num_streams for device in ['CPU', 'GPU'] if device in args.device} \
            if args.num_streams.isdigit() \
            else dict([device.split(':') for device in args.num_streams.split(',')])

    if 'CPU' in args.device:
        if args.cpu_extension:
            ie.add_extension(args.cpu_extension, 'CPU')
        if args.number_threads is not None:
            config_user_specified['CPU_THREADS_NUM'] = str(args.number_threads)
        if 'CPU' in devices_nstreams:
            config_user_specified['CPU_THROUGHPUT_STREAMS'] = devices_nstreams['CPU'] \
                if int(devices_nstreams['CPU']) > 0 \
                else 'CPU_THROUGHPUT_AUTO'

        config_min_latency['CPU_THROUGHPUT_STREAMS'] = '1'

    if 'GPU' in args.device:
        if 'GPU' in devices_nstreams:
            config_user_specified['GPU_THROUGHPUT_STREAMS'] = devices_nstreams['GPU'] \
                if int(devices_nstreams['GPU']) > 0 \
                else 'GPU_THROUGHPUT_AUTO'

        config_min_latency['GPU_THROUGHPUT_STREAMS'] = '1'

    # -------------------- 2. Reading the IR generated by the Model Optimizer (.xml and .bin files) --------------------
    log.info("Loading network")
    net = ie.read_network(args.model, os.path.splitext(args.model)[0] + ".bin")

    # ---------------------------------- 3. Load CPU extension for support specific layer ------------------------------
    if "CPU" in args.device:
        supported_layers = ie.query_network(net, "CPU")
        not_supported_layers = [l for l in net.layers.keys() if l not in supported_layers]
        if len(not_supported_layers) != 0:
            log.error("Following layers are not supported by the plugin for specified device {}:\n {}".
                      format(args.device, ', '.join(not_supported_layers)))
            log.error("Please try to specify cpu extensions library path in sample's command line parameters using -l "
                      "or --cpu_extension command line argument")
            sys.exit(1)

    assert len(net.input_info) == 1, "Sample supports only YOLO V3 based single input topologies"

    # ---------------------------------------------- 4. Preparing inputs -----------------------------------------------
    log.info("Preparing inputs")
    input_blob = next(iter(net.input_info))
    output_blob = list(net.outputs.key())

    # Read and pre-process input images
    if net.input_info[input_blob].input_data.shape[1] == 3:
        input_height, input_width = net.input_info[input_blob].input_data.shape[2:]
        nchw_shape = True
    else:
        input_height, input_width = net.input_info[input_blob].input_data.shape[1:3]
        nchw_shape = False

    input_stream = 0 if args.input == "cam" else args.input

    mode = Mode(Modes.USER_SPECIFIED)
    cap = cv2.VideoCapture(input_stream)
    wait_key_time = 1

    # ----------------------------------------- 5. Loading model to the plugin -----------------------------------------
    log.info("Loading model to the plugin")
    exec_nets = {}

    exec_nets[Modes.USER_SPECIFIED] = ie.load_network(network=net, device_name=args.device,
                                                      config=config_user_specified,
                                                      num_requests=args.num_infer_requests)
    exec_nets[Modes.MIN_LATENCY] = ie.load_network(network=net, device_name=args.device.split(":")[-1].split(",")[0],
                                                   config=config_min_latency,
                                                   num_requests=1)

    empty_requests = deque(exec_nets[mode.current].requests)
    completed_request_results = {}
    next_frame_id = 0
    next_frame_id_to_show = 0
    prev_mode_active_request_count = 0
    event = threading.Event()
    callback_exceptions = []

    # ----------------------------------------------- 6. Doing inference -----------------------------------------------
    log.info("Starting inference...")
    print("To close the application, press 'CTRL+C' here or switch to the output window and press ESC key")
    print("To switch between min_latency/user_specified modes, press TAB key in the output window")

    presenter = monitors.Presenter(args.utilization_monitors, 55,
                                   (round(cap.get(cv2.CAP_PROP_FRAME_WIDTH) / 4),
                                    round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) / 8)))

    while (cap.isOpened()
           or completed_request_results
           or len(empty_requests) < len(exec_nets[mode.current].requests)) \
            and not callback_exceptions:
        if next_frame_id_to_show in completed_request_results:
            frame, outputs, scale, start_time, is_same_mode = completed_request_results.pop(next_frame_id_to_show)

            next_frame_id_to_show += 1

            poses = postprocess(outputs, scale, output_name=output_blob, dataset='coco', stride=8)

            presenter.drawGraphs(frame)
            for pose in poses:
                pose.draw(frame)

            if is_same_mode and prev_mode_active_request_count == 0:
                pass
            else:
                prev_mode_active_request_count -= 1

            if not args.no_show:
                cv2.imshow("Detection Results", frame)
                key = cv2.waitKey(wait_key_time)

                if key in {ord("q"), ord("Q"), 27}:  # ESC key
                    break
                if key == 9:  # Tab key
                    if prev_mode_active_request_count == 0:
                        prev_mode = mode.current
                        mode.switch()

                        prev_mode_active_request_count = len(exec_nets[prev_mode].requests) - len(empty_requests)
                        empty_requests.clear()
                        empty_requests.extend(exec_nets[mode.current].requests)

                else:
                    presenter.handleKey(key)

        elif empty_requests and prev_mode_active_request_count == 0 and cap.isOpened():
            start_time = perf_counter()
            ret, frame = cap.read()
            if not ret:
                if args.loop_input:
                    cap.open(input_stream)
                else:
                    cap.release()
                continue

            request = empty_requests.popleft()

            # resize input_frame to network size
            in_frame, scale = preprocess_frame(frame, input_height, input_width, nchw_shape)

            # Start inference
            request.set_completion_callback(py_callback=async_callback,
                                            py_data=(request,
                                                     next_frame_id,
                                                     mode.current,
                                                     frame,
                                                     scale,
                                                     start_time,
                                                     completed_request_results,
                                                     empty_requests,
                                                     mode,
                                                     event,
                                                     callback_exceptions))
            request.async_infer(inputs={input_blob: in_frame})
            next_frame_id += 1

        else:
            event.wait()
            event.clear()

    if callback_exceptions:
        raise callback_exceptions[0]

    print(presenter.reportMeans())

    for exec_net in exec_nets.values():
        await_requests_completion(exec_net.requests)


if __name__ == '__main__':
    sys.exit(main() or 0)