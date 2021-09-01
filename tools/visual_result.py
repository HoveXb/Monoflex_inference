import torch
import datetime
import warnings

warnings.filterwarnings("ignore")

from config import cfg
from utils.check_point import DetectronCheckpointer
from my_engine import (
    default_argument_parser,
    default_setup,
)
from utils import comm
from my_engine.test_net import run_test
from model.detector import KeypointDetector
import resource

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

torch.backends.cudnn.enabled = True  # enable cudnn and uncertainty imported
# torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True  # enable cudnn to search the best algorithm



def setup(args):

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    cfg.SOLVER.IMS_PER_BATCH = args.batch_size
    cfg.DATALOADER.NUM_WORKERS = args.num_work
    cfg.TEST.EVAL_DIS_IOUS = args.eval_iou
    cfg.TEST.EVAL_DEPTH = args.eval_depth

    if args.vis_thre > 0:
        cfg.TEST.VISUALIZE_THRESHOLD = args.vis_thre

    if args.output is not None:
        cfg.OUTPUT_DIR = args.output

    if args.test:
        cfg.DATASETS.TEST_SPLIT = 'test'
        cfg.DATASETS.TEST = ("kitti_test",)

    cfg.START_TIME = datetime.datetime.strftime(datetime.datetime.now(), '%m-%d %H:%M:%S')
    default_setup(cfg, args)

    return cfg


if __name__ == '__main__':

    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)

    cfg = setup(args)

    distributed = comm.get_world_size() > 1
    if not distributed: cfg.MODEL.USE_SYNC_BN = False

    model = KeypointDetector(cfg)
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)

    checkpointer = DetectronCheckpointer(
        cfg, model, save_dir=cfg.OUTPUT_DIR
    )
    ckpt = cfg.MODEL.WEIGHT if args.ckpt is None else args.ckpt
    _ = checkpointer.load(ckpt, use_latest=args.ckpt is None)

    run_test(cfg, checkpointer.model, vis=args.vis, eval_score_iou=args.eval_score_iou,
                    eval_all_depths=args.eval_all_depths)