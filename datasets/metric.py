from .build import build_eval
from ignite.metrics.metric import reinit__is_reduced
from ignite.metrics import Metric
import ignite.distributed as idist
import torch


class CommonMetric(Metric):
    def __init__(self, cfg):
        self._eval = build_eval(cfg)
        super().__init__()

    @reinit__is_reduced
    def reset(self):
        self._eval.reset()
        super().reset()

    @torch.no_grad()
    def update(self, output):
        y_pred, y = output
        y_pred = [idist.all_gather(x.cpu()) for x in y_pred]
        y = idist.all_gather(y)
        self._eval.process(y, y_pred)

    @idist.one_rank_only(rank=0)
    def compute(self):
        return self._eval.evaluate()
