from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import argparse
from core.collections import AttrDict
from pathlib import Path


def add_global_arguments(parser):

    parser.add_argument('--batch-size', default=128, type=int, help='Per GPU batch size.')
    parser.add_argument('--epochs', default=20, type=int, help='Number of total epochs to run.')
    parser.add_argument('--start-epoch', type=int, default=1, help='used for resume')
    
    # Model parameters
    parser.add_argument('--arch', type=str, choices=['resnet18', 'resnet50'], default='resnet18',
                        help='Name of architecture to train.')
    parser.add_argument('--projection-hidden-dim', type=int, default=256, help='Projector hidden dimension.')
    parser.add_argument('--projection-dim', type=int, default=256, help='Projector output dimension.')
    parser.add_argument('--prediction-hidden-dim', type=int, default=256, help='Predictor hidden dimension.')
    parser.add_argument('--momentum-target', type=float, default=0.99,
                        help='''Base EMA parameter for target network update.
                        The value is increased to 1 during training with cosine schedule.''')
    parser.add_argument('--pos-radius', type=float, default=0.1, help='Positive radius.')
    parser.add_argument('--remove-stride-layers', type=str, nargs='+', default=('layer3', 'layer4'),
                        help='Reduce the stride of some layers in order to obtain a higher resolution feature map.')

    # Optimization parameters
    parser.add_argument('--lr', default=0.001, type=float, help='Learning rate.')
    parser.add_argument('--weight-decay', type=float, default=0, help='Weight decay')

    # Augmentation parameters
    parser.add_argument('--img-size', default=256, type=int, help='Image input size.')
    parser.add_argument('--min-crop', type=float, default=0.0, help='Minimum scale for random cropping.')
    parser.add_argument('--max-crop', type=float, default=1.0, help='Maximum scale for random cropping.')

    # Dataset parameters
    parser.add_argument('--data-path', default='', help='Dataset path.')
    parser.add_argument('--frame-rate', default=30, type=int, help='Frame rate (fps) of dataset.')
    parser.add_argument('--clip-len', default=1, type=int, help='Number of frames per clip.')
    parser.add_argument('--clips-per-video', default=80, type=int, help='Maximum number of clips per video to consider.')
    parser.add_argument('--workers', default=16, type=int)

    parser.add_argument('--output-dir', default='./checkpoints', help='Path where to save.')
    parser.add_argument('--auto-resume', action='store_true', help='auto resume from current.pth')
    parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint')
    parser.add_argument('--save-ckpt-freq', default=1, type=int)
    parser.add_argument('--device', default='cuda', help='Device to use for training.')
    parser.add_argument('--seed', type=int, default=42, help='Manual seed.')
    parser.add_argument('--enable-amp', default=True, help='use mixed precision')
    # Weights and Biases arguments
    parser.add_argument('--enable-wandb', default=False,
                        help="Enable logging to Weights and Biases.")
    parser.add_argument('--project', default='SViP', type=str,
                        help="The name of the W&B project where you're sending the new run.")
    # Testing arguments
    parser.add_argument('--test-model', default='hvc', type=str)
    parser.add_argument('--merge-model', default='mocov1', type=str)
    parser.add_argument('--pool-workers', default=32, type=int)
    parser.add_argument('--model-path', default='./checkpoints/hvc_coco.pth', type=str)
    parser.add_argument("--infer-list", default="val.txt", type=str)
    parser.add_argument('--mask-output-dir', type=str, default="./results", help='path where to save masks')
    
    args = parser.parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

def check_global_arguments(args):

    args.cuda = torch.cuda.is_available()
    print("Available threads: ", torch.get_num_threads())

def get_arguments(args_in):
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Hybrid Visual Correspondence for Self-Supervised Video Object Segmentation")

    add_global_arguments(parser)
    args = parser.parse_args(args_in)
    check_global_arguments(args)

    return args


__C = AttrDict()
cfg = __C
# ---------------------------------------------------------------------------- #
# Dataset options (+ augmentation options)
# ---------------------------------------------------------------------------- #
__C.DATASET = AttrDict()
__C.DATASET.ROOT = "data"
__C.DATASET.VIDEO_LEN = 5
__C.DATASET.NUM_CLASSES = 20
__C.VERBOSE = False

# inference-time parameters double-model
__C.TEST = AttrDict()
__C.TEST.RADIUS = 12
__C.TEST.TEMP = 0.1
__C.TEST.KNN = 15
__C.TEST.CXT_SIZE = 50
__C.TEST.INPUT_SIZE = -1
__C.TEST.LAMBDA = 1.05
__C.TEST.MERGE = True
__C.TEST.MODEL = 'hvc'
__C.TEST.EXP = "hvc"
