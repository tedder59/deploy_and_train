# Copyright (c) 2022 by zhuxiaolong, All rights Reserved.
from ..build import MODULE_REGISTRY
import torch.nn as nn


class HeadBlock(nn.Module):
    def __init__(self, in_channels, head_channels, out_channels,
                 prior_bias=0):
        super().__init__()

        if head_channels > 0:
            self.head_conv = nn.Sequential(
                nn.Conv2d(in_channels, head_channels, 3, padding=1),
                nn.ReLU(inplace=True)
            )
            in_channels = head_channels
        else:
            self.head_conv = nn.Identity()
            
        self.fc = nn.Conv2d(in_channels, out_channels, 1)
        nn.init.constant_(self.fc.bias, prior_bias)

    def forward(self, x):
        x = self.head_conv(x)
        x = self.fc(x)
        return x


@MODULE_REGISTRY.register()
class CenterNetHead(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        in_channels = kwargs['in_channels']
        
        heads = {}
        for head_cfg in kwargs['branches']:
            name = head_cfg['name']
            del head_cfg['name']
            heads[name] = HeadBlock(in_channels=in_channels, **head_cfg)
        self.heads = nn.ModuleDict(heads)
        
    def forward(self, x):
        outs = {}
        for k, layer in self.heads.items():
            outs[k] = layer(x)
            
        return outs
