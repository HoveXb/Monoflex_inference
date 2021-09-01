import argparse
import os

import torch

from utils import comm
from utils.miscellaneous import mkdir
from utils.logger import setup_logger
from utils.collect_env import collect_env_info
from utils.envs import seed_all_rng

__all__ = ["default_argument_parser", "default_setup"]


def default_argument_parser():
    parser = argparse.ArgumentParser(description="Detectron2 Training")
    parser.add_argument("--config", dest='config_file', default="/media/hovexb/HSJ2/Monoflex_inference/runs/monoflex.yaml",
                        metavar="FILE", help="path to config file")

    parser.add_argument("--eval", dest='eval_only', action="store_false", help="perform evaluation only")
    parser.add_argument("--eval_iou", action="store_true", help="evaluate disentangling IoU")
    parser.add_argument("--eval_depth", action="store_true", help="evaluate depth errors")
    parser.add_argument("--eval_all_depths", action="store_true")

    parser.add_argument("--eval_score_iou", action="store_true", 
                            help="evaluate the relationship between scores and IoU")

    parser.add_argument("--test", action="store_false", help="test mode")
    parser.add_argument("--vis", action="store_false", help="visualize when evaluating")
    parser.add_argument(
        "--ckpt",
        help="The path to the checkpoint for test, default is the latest checkpoint.",
        default='/media/hovexb/HSJ2/Monoflex_inference/model_moderate_best_soft.pth',
    )
    
    parser.add_argument("--num_gpus", type=int, default=1, help="number of gpu")
    parser.add_argument("--batch_size", type=int, default=2, help="number of batch_size")
    parser.add_argument("--num_work", type=int, default=8, help="number of workers for dataloader")
    parser.add_argument("--output", type=str, default='/home/hovexb/CODE/MonoFlex/output/exp')

    parser.add_argument("--vis_thre", type=float, default=0.25, help="threshold for visualize results of detection")
    parser.add_argument("--num-machines", type=int, default=1)
    parser.add_argument(
        "--machine-rank", type=int, default=0, help="the rank of this machine (unique per machine)"
    )

    # PyTorch still may leave orphan processes in multi-gpu training.
    # Therefore we use a deterministic way to obtain port,
    # so that users are aware of orphan processes by seeing the port occupied.
    port = 2 ** 15 + 2 ** 14 + hash(os.getuid()) % 2 ** 13
    # parser.add_argument("--dist-url", default="tcp://127.0.0.1:{}".format(port))
    parser.add_argument("--dist-url", default="auto")

    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    return parser

def default_setup(cfg, args):

    rank = comm.get_rank()

    # make sure each worker has a different, yet deterministic seed if specified
    seed_all_rng(None if cfg.SEED < 0 else cfg.SEED + rank)

    # cudnn benchmark has large overhead. It shouldn't be used considering the small size of
    # typical validation set.
    if not (hasattr(args, "eval_only") and args.eval_only):
        torch.backends.cudnn.benchmark = cfg.CUDNN_BENCHMARK
