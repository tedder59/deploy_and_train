import os
with_torch_launch = "WORLD_SIZE" in os.environ


from pytorch_quantization import nn as quant_nn
from pytorch_quantization.tensor_quant import QuantDescriptor
from pytorch_quantization import calib


quant_desc_input = QuantDescriptor(calib_method='histogram')
quant_nn.QuantConv2d.set_default_quant_desc_input(quant_desc_input)
quant_nn.QuantLinear.set_default_quant_desc_input(quant_desc_input)


from pytorch_quantization import quant_modules
quant_nn.TensorQuantizer.use_fb_fake_quant = True
quant_modules.initialize()


from datasets import CommonMetric, build_dataset, get_collate_wrapper
from modules import build_model, build_criterion, create_visualizer
from configs import Config

from ignite.handlers import Checkpoint, ModelCheckpoint, TerminateOnNan
from ignite.engine import create_supervised_evaluator
from ignite.contrib.handlers import ProgressBar
from ignite.contrib.engines import common
from ignite.engine import Engine, Events
import ignite.distributed as idist

from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.optim as optim
from tqdm import tqdm
import argparse
import torch


def enable_calib(model):
    for module in model.modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.disable_quant()
                module.enable_calib()
            else:
                module.disable()


def enable_quant(model):
    for module in model.modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.enable_quant()
                module.disable_calib()
            else:
                module.enable()


def collect_statics(model, dataloader):
    enable_calib(model)

    for data in tqdm(dataloader):
        inputs = data
        model(inputs.cuda())
        
    enable_quant(model)


def compute_amax(model, **kwargs):
    for module in model.modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                if isinstance(module._calibrator, calib.MaxCalibrator):
                    module.load_calib_amax(strict=False)
                else:
                    module.load_calib_amax(strict=False, **kwargs)
    model.cuda()


def get_training_dataflow(cfg):
    train_dataset = build_dataset(cfg.DATASET, "TRAIN")
    if idist.get_rank() == 0:
        idist.barrier()

    ccfg = cfg.DATALOADER.TRAIN
    train_dataloader = idist.auto_dataloader(
        train_dataset, batch_size=ccfg.BATCH_SIZE,
        num_workers=ccfg.NUM_WORKERS,
        collate_fn=get_collate_wrapper(ccfg.COLLATE_FN),
        shuffle=True, drop_last=True
    )

    val_dataset = build_dataset(cfg.DATASET, "VAL")
    if idist.get_rank() == 0:
        idist.barrier()

    ccfg = cfg.DATALOADER.VAL
    val_dataloader = idist.auto_dataloader(
        val_dataset, batch_size=ccfg.BATCH_SIZE,
        num_workers=ccfg.NUM_WORKERS,
        collate_fn=get_collate_wrapper(ccfg.COLLATE_FN),
        shuffle=False, drop_last=False
    )

    return train_dataloader, val_dataloader


def get_dataflow(cfg, split="TEST"):
    split = split.upper()
    dataset = build_dataset(cfg.DATASET, split)

    ccfg = cfg.DATALOADER.get(split, None)
    assert ccfg, "dataloader config no {}".format(split)
    dataloader = idist.auto_dataloader(
        dataset, batch_size=ccfg.BATCH_SIZE,
        num_workers=ccfg.NUM_WORKERS,
        collate_fn=get_collate_wrapper(ccfg.COLLATE_FN),
        shuffle=False, drop_last=False
    )

    return dataloader


@torch.no_grad()
def ptq(cfg):
    dataloader = get_dataflow(cfg)
    model = build_model(cfg.MODEL)
    model.cuda()

    to_load = {
        "model": model
    }

    ccfg = cfg.SAVE
    assert ccfg.RESUME, "resume can't be none"
    print(f'resume form: {ccfg.RESUME}')
    checkpoint = torch.load(ccfg.RESUME, map_location='cpu')
    Checkpoint.load_objects(to_load=to_load, checkpoint=checkpoint)

    model.eval()
    collect_statics(model, dataloader)
    compute_amax(model, method='percentile', percentile=99.99)
    
    ckpt = cfg.SAVE.PTQ
    torch.save({'model': model.state_dict()}, ckpt)

    dataloader = get_dataflow(cfg, "VAL")
    ccfg = cfg.EVALUATOR
    evaluator = create_supervised_evaluator(
        model, metrics = {ccfg.METRIC_NAME: CommonMetric(ccfg)},
        device=idist.device(),
        prepare_batch=model.prepare_eval_batch,
        output_transform=lambda x, y, y_pred: (model.predict(y_pred), y)
    )
    evaluator.run(dataloader)


def qat(local_rank, cfg):
    train_loader, val_loader = get_training_dataflow(cfg)

    model = build_model(cfg.MODEL)
    model = idist.auto_model(model)
    criterion = build_criterion(cfg.CRITERION)

    ccfg = cfg.SOLVER
    optimizer = optim.SGD(model.parameters(),
                          lr=ccfg.BASE_LR,
                          momentum=ccfg.get("MOMENTUM", 0.9),
                          weight_decay=ccfg.get("WEIGHT_DECAY", 1e-5),
                          nesterov=ccfg.get("NESTEROV", False))
    optimizer = idist.auto_optim(optimizer)
    lr_scheduler = CosineAnnealingLR(optimizer, ccfg.MAX_EPOCHS)

    def train_step(engine, batch):
        x, y = model.prepare_train_batch(batch)
        outs = model(x)
        losses, log_dict = criterion(outs, y)
        
        losses.backward()
        optimizer.step()
        optimizer.zero_grad()
        return outs, log_dict

    def empty_cuda_cache():
        torch.cuda.empty_cache()
        import gc
        gc.collect()

    trainer = Engine(train_step)
    trainer.add_event_handler(Events.ITERATION_COMPLETED, TerminateOnNan())
    trainer.add_event_handler(Events.EPOCH_STARTED, lambda: model.train())
    trainer.add_event_handler(Events.EPOCH_COMPLETED, lambda: lr_scheduler.step())
    trainer.add_event_handler(Events.EPOCH_COMPLETED, empty_cuda_cache)

    to_save = {
        "model": model,
        "optimizer": optimizer,
        "lr_scheduler": lr_scheduler,
        "trainer": trainer,
    }

    ccfg = cfg.EVALUATOR
    evaluator = create_supervised_evaluator(
        model, metrics = {ccfg.METRIC_NAME: CommonMetric(ccfg)},
        device=idist.device(),
        prepare_batch=model.prepare_eval_batch,
        output_transform=lambda x, y, y_pred: (model.predict(y_pred), y)
    )
    events = Events.EPOCH_COMPLETED(every=ccfg.INTERVAL)
    events |= Events.COMPLETED

    @trainer.on(events)
    def eval():
        model.eval()
        evaluator.run(val_loader)
        empty_cuda_cache()

    ccfg = cfg.SAVE
    if idist.get_rank() == 0:
        save_handler = ModelCheckpoint(
            dirname=ccfg.get("OUTPUT_PATH", "data/runs"),
            filename_prefix=cfg.MODEL.META_ARCHITECTURE,
            require_empty=False,
            score_name=ccfg.VAL_SCORE
        )

        evaluator.add_event_handler(
            Events.EPOCH_COMPLETED,
            save_handler, to_save
        )

        save_handler = ModelCheckpoint(
            dirname=ccfg.get("OUTPUT_PATH", "data/runs"),
            filename_prefix=cfg.MODEL.META_ARCHITECTURE,
            require_empty=False,
            n_saved=ccfg.NUM_CHECKPOINTS
        )

        trainer.add_event_handler(
            Events.EPOCH_COMPLETED(lambda _, x: x % ccfg.INTERVAL == 0),
            save_handler, to_save
        )

        ProgressBar().attach(trainer)
        visualize_handler = create_visualizer(train_loader, model, cfg.VISUALIZE)

    assert ccfg.RESUME or ccfg.PTQ, "resume and ptq can't be both none"
    if ccfg.RESUME:
        print(f'resume form: {ccfg.RESUME}')
        checkpoint = torch.load(ccfg.RESUME, map_location='cpu')
        Checkpoint.load_objects(to_load=to_save, checkpoint=checkpoint)
    elif ccfg.PTQ:
        print(f'load from ptq: {ccfg.PTQ}')
        checkpoint = torch.load(ccfg.PTQ, map_location="cpu")
        model.load_state_dict(checkpoint['model'])

    if idist.get_rank() == 0:
        tb_logger = common.setup_tb_logging(
            cfg.SAVE.get("OUTPUT_PATH", "data/runs"),
            trainer, optimizer, evaluator
        )

        tb_logger.attach_output_handler(
            trainer,
            event_name=Events.ITERATION_COMPLETED,
            tag='qat',
            output_transform=lambda output: output[1]
        )

        tb_logger.attach(
            trainer,
            log_handler=visualize_handler,
            event_name=Events.ITERATION_COMPLETED
        )

    trainer.run(train_loader,
                max_epochs=cfg.SOLVER.MAX_EPOCHS)
    
    if idist.get_rank() == 0:
        tb_logger.close()


if __name__ == "__main__" and (not with_torch_launch):
    """
    Single node with 1 GPU: python quantize.py ptq --config xxx
    """
    assert torch.cuda.is_available(), "cuda invalid!"
    assert torch.backends.cudnn.is_available(), "cudnn invalid!"
    torch.backends.cudnn.benchmark = True

    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=['QAT', 'PTQ'],
                        help="quantize option:[QAT or PTQ]")
    parser.add_argument("--config", type=str, required=True,
                        help="config file")
    args = parser.parse_args()
    
    cfg = Config(Config.load_yaml_with_base(args.config))
    if args.mode == 'PTQ':
        ptq(cfg)
    elif args.mode == "QAT":
        qat(0, cfg)


if __name__ == "__main__" and with_torch_launch:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["QAT"], required=False,
                        default="QAT", help="multi gpus qat mode")
    parser.add_argument("--config", type=str, required=True,
                        help="config file")
    args = parser.parse_args()

    cfg = Config.load_yaml_with_base(args.config)
    with idist.Parallel(backend="nccl") as parallel:
        parallel.run(qat, cfg)
