import time
from itertools import cycle

import cv2
import numpy as np

from .meters import MovingAverageMeter
from .pipeline import PipelineStep, AsyncPipeline
from .queue import Signal
from .models import AsyncWrapper


def preprocess_frame(frame, input_height, input_width):
    height, width, _ = frame.shape
    scale = (input_width / width, input_height / height)
    in_frame = cv2.resize(frame, (0, 0), fx=scale[0], fy=scale[1],
                          interpolation=cv2.INTER_CUBIC)
    in_frame = (in_frame - 128) / 255.0
    in_frame = in_frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW
    in_frame = np.expand_dims(in_frame, axis=0)
    return in_frame, scale


def run_pipeline(video, estimator, render_obj):
    pipeline = AsyncPipeline()
    pipeline.add_step('Data', DataStep(video), parallel=False)
    pipeline.add_step('Estimator', EstimatorStep(estimator), parallel=False)
    pipeline.add_step('Render', RenderStep(render_obj, fps=30), parallel=False)
    pipeline.run()
    pipeline.close()
    pipeline.print_statistics()


class DataStep(PipelineStep):

    def __init__(self, video_list, loop=True):
        super().__init__()
        self.video_list = video_list
        self.cap = None

        if loop:
            self._video_cycle = cycle(self.video_list)
        else:
            self._video_cycle = iter(self.video_list)

    def setup(self):
        self._open_video()

    def process(self, item):
        if not self.cap.isOpened() and not self._open_video():
            return Signal.STOP
        status, frame = self.cap.read()
        if not status:
            return Signal.STOP
        return frame

    def end(self):
        self.cap.release()

    def _open_video(self):
        next_video = next(self._video_cycle)
        try:
            next_video = int(next_video)
        except ValueError:
            pass
        self.cap = cv2.VideoCapture(next_video)
        if not self.cap.isOpened():
            return False
        return True


class EstimatorStep(PipelineStep):
    def __init__(self, estimator):
        super(EstimatorStep, self).__init__()
        self.estimator = estimator
        self.async_model = AsyncWrapper(self.estimator, self.estimator.num_requests)

    def process(self, frame):
        # TODO: Extract shape from self.estimator
        preprocessed, scale = preprocess_frame(frame, 368, 368)
        outputs, frame = self.async_model.infer(preprocessed, frame)
        if outputs is None:
            return None
        return frame, outputs, scale, {'estimation': self.own_time.last}


class RenderStep(PipelineStep):
    """Passes inference result to render function"""

    def __init__(self, renderer, fps):
        super().__init__()
        self.renderer = renderer
        self.render = renderer.render_frame
        self.meta = renderer.meta
        self.fps = fps
        self._frames_processed = 0
        self._t0 = None
        self._render_time = MovingAverageMeter(0.9)

    def process(self, item):
        if item is None:
            return
        self._sync_time()
        render_start = time.time()
        status = self.render(*item, self._frames_processed)
        self._render_time.update(time.time() - render_start)

        self._frames_processed += 1
        if status is not None and status < 0:
            return Signal.STOP_IMMEDIATELY
        return status

    def end(self):
        cv2.destroyAllWindows()
        if hasattr(self.renderer, 'writer'):
            self.renderer.writer.release()

    def _sync_time(self):
        now = time.time()
        if self._t0 is None:
            self._t0 = now
        expected_time = self._t0 + (self._frames_processed + 1) / self.fps
        if self._render_time.avg:
            expected_time -= self._render_time.avg
        if expected_time > now:
            time.sleep(expected_time - now)




