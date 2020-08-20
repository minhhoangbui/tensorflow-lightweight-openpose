from collections import deque
from itertools import cycle

from openvino.inference_engine import IECore


class IEModel:
    def __init__(self, model_xml, model_bin, target_device, num_requests, batch_size=1):
        # Read IR
        print("Reading IR...")
        ie_core = IECore()
        self.net = ie_core.read_network(model_xml, model_bin)
        self.net.batch_size = batch_size

        print("Loading IR to the plugin...")

        self.exec_net = ie_core.load_network(network=self.net, device_name=target_device, num_requests=num_requests)
        self.input_name = next(iter(self.net.input_info))
        self.output_name = list(self.net.outputs.keys())
        self.num_requests = num_requests

    def infer(self, frame):
        input_data = {self.input_name: frame}
        infer_result = self.exec_net.infer(input_data)
        return infer_result[self.output_name]

    def async_infer(self, frame, req_id):
        input_data = {self.input_name: frame}
        self.exec_net.start_async(request_id=req_id, inputs=input_data)

    def wait_request(self, req_id):
        self.exec_net.requests[req_id].wait()
        return self.exec_net.requests[req_id].output_blobs


class AsyncWrapper:
    def __init__(self, ie_model, num_requests):
        self.net = ie_model
        self.num_requests = num_requests
        self._result_ready = False
        self._req_ids = cycle(range(num_requests))
        self._result_ids = cycle(range(num_requests))
        self._frames = deque(maxlen=num_requests)

    def infer(self, model_input, frame=None):
        """Schedule current model input to infer, return last result"""
        next_req_id = next(self._req_ids)
        self.net.async_infer(model_input, next_req_id)

        last_frame = self._frames[0] if self._frames else frame

        self._frames.append(frame)
        if next_req_id == self.num_requests - 1:
            self._result_ready = True

        if self._result_ready:
            result_req_id = next(self._result_ids)
            result = self.net.wait_request(result_req_id)
            return result, last_frame
        else:
            return None, None
