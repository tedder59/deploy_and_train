# Copyright (c) 2022 by zhuxiaolong, All rights Reserved.
from fvcore.common.registry import Registry


DATASET_REGISTRY = Registry("DATASET")
DATASET_REGISTRY.__doc__ = """
Registry for datasets.
"""

COLLATE_REGIESTRY = Registry("COLLATE")
COLLATE_REGIESTRY.__doc__ = """
Registry for batch collate wrapper functions, called by DataLoader.
"""

TRANSFORM_REGISTRY = Registry("TRANSFORM")
TRANSFORM_REGISTRY.__doc__ = """
Registry for DATASET transform
"""

EVALUATOR_REGISTRY = Registry("EVALUATOR")
EVALUATOR_REGISTRY.__doc__ = """
Registry dataset evaluator.
"""


def build_dataset(cfg, split):
    name = cfg.NAME
    split = split.upper()
    return DATASET_REGISTRY.get(name)(cfg[split])


def get_collate_wrapper(cfg):
    name = cfg.NAME
    if name is not None:
        return COLLATE_REGIESTRY.get(name)(cfg)
    else:
        return None


def build_transform(cfg):
    class Pipeline:
        def __init__(self, cfg):
            self.trans = [
                TRANSFORM_REGISTRY.get(k)(**x) \
                if isinstance(x, dict) \
                else TRANSFORM_REGISTRY.get(k)()
                for k, x in cfg.items()
            ]

        def __call__(self, data):
            for t in self.trans:
                data = t(data) 
            return data

    return Pipeline(cfg)


def build_eval(cfg):
    name = cfg.NAME
    return EVALUATOR_REGISTRY.get(name)(cfg)
