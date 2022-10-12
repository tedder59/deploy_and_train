# Copyright (c) 2022 by zhuxiaolong, All rights Reserved.
from ..backbone import *
from ..neck import *
from ..head import *
from ..build import META_ARCH_REGISTRY, build_module
import torch.nn as nn
import torch


@META_ARCH_REGISTRY.register()
class RetinaNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        sub_nets = []
        for module_cfg in cfg.MODULES:
            sub_nets.append(build_module(**module_cfg))
        self.sub_nets = nn.ModuleList(sub_nets)

    def forward(self, x):
        out = self.sub_nets[0](x)
        for i in range(1, len(self.sub_nets)):
            out = self.sub_nets[i](out)
        
        return [x for x in out.values()]
