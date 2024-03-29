import os
with_torch_launch = "WORLD_SIZE" in os.environ

from datasets import CommonMetric, build_dataset, get_collate_wrapper
from modules.build import build_model
from configs import Config

from ignite.contrib.handlers import ProgressBar
from ignite.engine import create_supervised_evaluator
import ignite.distributed as idist

import argparse
import torch


def get_dataflow(cfg):
    dataset = build_dataset(cfg.DATASET, "VAL")
    idist.barrier()

    ccfg = cfg.DATALOADER.VAL
    dataloader = idist.auto_dataloader(
        dataset, batch_size=ccfg.BATCH_SIZE,
        num_workers=ccfg.NUM_WORKERS,
        collate_fn=get_collate_wrapper(ccfg.COLLATE_FN),
        shuffle=False, drop_last=False
    )

    return dataloader


def create(cfg, ckpt):
    model = build_model(cfg.MODEL)

    ccfg = cfg.SAVE
    ckpt = os.path.join(ccfg.OUTPUT_PATH, ckpt)
    checkpoint = torch.load(ckpt, map_location='cpu')
    model.load_state_dict(checkpoint['model'])

    model.eval()
    model = idist.auto_model(model)

    ccfg = cfg.MISC
    with_amp = ccfg.get("WITH_AMP", False)

    ccfg = cfg.EVALUATOR
    evaluator = create_supervised_evaluator(
        model, metrics = {ccfg.METRIC_NAME: CommonMetric(ccfg)},
        device=idist.device(),
        prepare_batch=model.prepare_eval_batch,
        output_transform=lambda x, y, y_pred: (model.predict(y_pred), y),
        amp_mode='amp' if with_amp else None
    )

    if idist.get_rank() == 0:
        ProgressBar().attach(evaluator)

    return evaluator


def eval(local_rank, cfg, ckpt):
    dataloader = get_dataflow(cfg)
    evaluator = create(cfg, ckpt)
    evaluator.run(dataloader)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True,
                        help="config file")
    parser.add_argument("--ckpt", type=str, required=True,
                        help="checkpoint filename")
    args = parser.parse_args()
    return args


if __name__ == "__main__" and (not with_torch_launch):
    """
    Single node with 1 GPU: python eval.py --config xxx
    """
    assert torch.cuda.is_available(), "cuda invalid!"
    assert torch.backends.cudnn.is_available(), "cudnn invalid!"
    torch.backends.cudnn.benchmark = True

    args = parse_args()
    cfg = Config(Config.load_yaml_with_base(args.config))
    eval(0, cfg, args.ckpt)


if __name__ == "__main__" and with_torch_launch:
    """
    Single node with 4 GPUS
    torchrun --nproc_per_node=4 eval.py -- --config xxx

    Multi nodes with multi GPUS
    node 0:
        torchrun --nnodes=2 --node_rank=0 --master_addr=master_ip --master_port=59344 --nproc_per_node=4 train.py -- --config xxx
    node 1:
        torchrun --nnodes=2 --node_rank=1 --master_addr=master_ip --master_port=59344 --nproc_per_node=4 train.py -- --config xxx
    """
    assert torch.cuda.is_available(), "cuda invalid!"
    assert torch.backends.cudnn.is_available(), "cudnn invalid!"
    torch.backends.cudnn.benchmark = True

    args = parse_args()
    cfg = Config(Config.load_yaml_with_base(args.config))
    with idist.Parallel(backend="nccl") as parallel:
        parallel.run(eval, cfg)
