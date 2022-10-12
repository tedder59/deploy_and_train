# Copyright (c) 2022 by zhuxiaolong, All rights Reserved.
from fvcore.common.registry import Registry


MODULE_REGISTRY = Registry("MODULE")
MODULE_REGISTRY.__doc__ = """
Registry for modules.
"""

def build_module(**kwargs):
    name = kwargs['NAME']
    del kwargs['NAME']
    return MODULE_REGISTRY.get(name)(**kwargs)


META_ARCH_REGISTRY = Registry("META_ARCH")
META_ARCH_REGISTRY.__doc__ = """
Registry for meta-architectures, i.e. the whole model.
"""

def build_model(cfg):
    name = cfg.META_ARCHITECTURE
    return META_ARCH_REGISTRY.get(name)(cfg)


CRITERION_REGISTRY = Registry("CRITERION")
CRITERION_REGISTRY.__doc__ = """
Registry for criterion, i.e. the loss calculate of whole model.
"""

def build_criterion(cfg):
    name = cfg.NAME
    return CRITERION_REGISTRY.get(name)(cfg)

    