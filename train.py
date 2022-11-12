import os
with_torch_launch = "WORLD_SIZE" in os.environ

from datasets import CommonMetric, build_dataset, get_collate_wrapper
from modules import build_model, build_criterion, create_visualizer
from configs import Config

from ignite.handlers import Checkpoint, ModelCheckpoint, TerminateOnNan
from ignite.contrib.handlers import ProgressBar
from ignite.contrib.engines import common
from ignite.engine import Engine, Events
from ignite.engine import create_supervised_evaluator
import ignite.distributed as idist

from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler
import torch.optim as optim
import argparse
import torch


def get_dataflow(cfg):
    train_dataset = build_dataset(cfg.DATASET, "TRAIN")
    val_dataset = build_dataset(cfg.DATASET, "VAL")
    idist.barrier()

    num_world = idist.get_world_size()

    ccfg = cfg.DATALOADER.TRAIN
    train_dataloader = idist.auto_dataloader(
        train_dataset, batch_size=ccfg.BATCH_SIZE * num_world,
        num_workers=ccfg.NUM_WORKERS,
        collate_fn=get_collate_wrapper(ccfg.COLLATE_FN),
        shuffle=True, drop_last=True
    )

    ccfg = cfg.DATALOADER.VAL
    val_dataloader = idist.auto_dataloader(
        val_dataset, batch_size=ccfg.BATCH_SIZE * num_world,
        num_workers=ccfg.NUM_WORKERS,
        collate_fn=get_collate_wrapper(ccfg.COLLATE_FN),
        shuffle=False, drop_last=False
    )

    return train_dataloader, val_dataloader


def create_trainer(cfg, train_loader, val_loader):
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

    ccfg = cfg.MISC
    with_amp = ccfg.get("WITH_AMP", False)
    scaler = GradScaler(enabled=with_amp)

    if ccfg.ASP:
        from apex.contrib.sparsity import ASP
        ASP.prune_trained_model(model, optimizer)

    def train_step(engine, batch):
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            x, y = model.module.prepare_train_batch(batch)
        else:
            x, y = model.prepare_train_batch(batch)
        with autocast(enabled=with_amp):
            outs = model(x)
            losses, log_dict = criterion(outs, y)
        
        scaler.scale(losses).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        return outs, log_dict

    def empty_cuda_cache():
        torch.cuda.empty_cache()
        import gc
        gc.collect()

    def distributed_sampler_shuffle(engine):
        train_loader.sampler.set_epoch(engine.state.epoch)

    trainer = Engine(train_step)
    trainer.add_event_handler(Events.ITERATION_COMPLETED, TerminateOnNan())
    trainer.add_event_handler(Events.EPOCH_STARTED, lambda: model.train())
    trainer.add_event_handler(Events.EPOCH_COMPLETED, lambda: lr_scheduler.step())
    trainer.add_event_handler(Events.EPOCH_COMPLETED, empty_cuda_cache)

    if idist.get_world_size() > 1:
        trainer.add_event_handler(Events.EPOCH_STARTED,
                                  distributed_sampler_shuffle)

    to_save = {
        "model": model,
        "optimizer": optimizer,
        "lr_scheduler": lr_scheduler,
        "trainer": trainer,
        "amp": scaler
    }

    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        eval_prepare_batch = model.module.prepare_eval_batch
        predict = model.module.predict
    else:
        eval_prepare_batch = model.prepare_eval_batch
        predict = model.predict

    ccfg = cfg.EVALUATOR
    evaluator = create_supervised_evaluator(
        model, metrics = {ccfg.METRIC_NAME: CommonMetric(ccfg)},
        device=idist.device(),
        prepare_batch=eval_prepare_batch,
        output_transform=lambda x, y, y_pred: (predict(y_pred), y),
        amp_mode='amp' if with_amp else None
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

        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            visualize_handler = create_visualizer(train_loader, model.module, cfg.VISUALIZE)
        else:
            visualize_handler = create_visualizer(train_loader, model, cfg.VISUALIZE)
    else:
        visualize_handler = None

    if ccfg.get('RESUME', None) is not None:
        print(f'resume form: {ccfg.RESUME}')
        checkpoint = torch.load(ccfg.RESUME, map_location='cpu')
        Checkpoint.load_objects(to_load=to_save, checkpoint=checkpoint)

    evaluators = {
        'eval': evaluator
    }
    
    optimizers = {
        'optim': optimizer
    }

    return trainer, evaluators, optimizers, visualize_handler

def train(local_rank, cfg):
    train_loader, val_loader = get_dataflow(cfg)
    trainer, evaluators, optimizers, visualize_handler =\
        create_trainer(cfg, train_loader, val_loader)

    if local_rank == 0:
        tb_logger = common.setup_tb_logging(
            cfg.SAVE.get("OUTPUT_PATH", "data/runs"),
            trainer, optimizers, evaluators
        )

        tb_logger.attach_output_handler(
            trainer,
            event_name=Events.ITERATION_COMPLETED,
            tag='training',
            output_transform=lambda output: output[1]
        )

        if visualize_handler is not None:
            tb_logger.attach(
                trainer,
                log_handler=visualize_handler,
                event_name=Events.ITERATION_COMPLETED
            )

    trainer.run(train_loader,
                max_epochs=cfg.SOLVER.MAX_EPOCHS)
    if local_rank == 0:
        tb_logger.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True,
                        help="config file")
    args = parser.parse_args()
    return args


if __name__ == "__main__" and (not with_torch_launch):
    """
    Single node with 1 GPU: python train.py --config xxx
    """
    assert torch.cuda.is_available(), "cuda invalid!"
    assert torch.backends.cudnn.is_available(), "cudnn invalid!"
    torch.backends.cudnn.benchmark = True

    args = parse_args()
    cfg = Config(Config.load_yaml_with_base(args.config))
    train(0, cfg)


if __name__ == "__main__" and with_torch_launch:
    """
    Single node with 4 GPUS
    torchrun --nproc_per_node=4 train.py -- --config xxx

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
        parallel.run(train, cfg)
