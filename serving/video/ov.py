import sys
from argparse import ArgumentParser
from os import path

from serving.video.common.models import IEModel
from serving.video.common.steps import run_pipeline
from serving.video.common.renderer import ResultRenderer


def demo(estimator, videos, output, meta):
    result_presenter = ResultRenderer(output_dir=output, meta=meta, num_requests=8)
    run_pipeline(videos, estimator, result_presenter)


def build_argparser():
    parser = ArgumentParser()
    args = parser.add_argument_group('Options')
    args.add_argument('-p', '--m_pose_estimator',
                      help="Optional. Path to estimation model", type=str, default=None)
    args.add_argument("-i", "--input",
                      help="Required. Id of the video capturing device to open (to open default camera just pass 0), "
                           "path to a video or a .txt file with a list of ids or video files (one object per line)",
                      required=True, type=str)
    args.add_argument("-ni", '--num_requests', default=1, type=int)
    args.add_argument("-d", "--device",
                      help="Optional. Specify a target device to infer on. CPU, GPU, FPGA, HDDL or MYRIAD is "
                           "acceptable. The demo will look for a suitable plugin for the device specified. "
                           "Default value is CPU",
                      default="CPU", type=str)
    args.add_argument("--fps", help="Optional. FPS for renderer", default=30, type=int)
    args.add_argument("--output", type=str, help="Optional. Export output or show directly")
    return parser


def main():
    args = build_argparser().parse_args()

    full_name = path.basename(args.input)
    extension = path.splitext(full_name)[1]
    if '.txt' in extension:
        with open(args.input) as f:
            videos = [line.strip() for line in f.read().split('\n')]
    else:
        videos = [args.input]

    pose_xml = args.m_pose_estimator
    pose_bin = args.m_pose_estimator.replace(".xml", ".bin")

    estimator = IEModel(pose_xml, pose_bin, args.device, num_requests=args.num_requests)

    meta = {'dataset': 'coco',
            'output_names': estimator.output_name,
            'stride': 8}
    demo(estimator, videos, args.output, meta)


if __name__ == '__main__':
    sys.exit(main() or 0)
