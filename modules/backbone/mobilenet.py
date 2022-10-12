# Copyright (c) 2022 by zhuxiaolong, All rights Reserved.
from ..build import MODULE_REGISTRY
from ..layer import FrozenBatchNorm2d
from torchvision.ops.misc import ConvNormActivation, SqueezeExcitation
from functools import partial
import torch.nn as nn
import torch
import collections


def _make_divisible(v, divisor, min_value= None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor

    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def _freeze(m):
    for p in m.parameters():
        p.requires_grad = False
    FrozenBatchNorm2d.convert_frozen_batchnorm(m)


InvertResidualConfig = collections.namedtuple(
    "InvertResidualConfig",
    ["input_channels", "kernel", "expanded_channels",
    "out_channels", "use_se", "use_hs", "stride", "dilation"]
)

_default_se_layer = partial(SqueezeExcitation,
                            scale_activation=nn.Sigmoid)


class InvertedResidual(nn.Module):
    def __init__(self, cfg, norm_layer,
                 se_layer=_default_se_layer):
        super().__init__()
        self.use_residual = cfg.stride == 1\
                and cfg.input_channels == cfg.out_channels
        activation_layer = nn.Hardswish if cfg.use_hs else nn.ReLU6

        layers = []
        if cfg.expanded_channels != cfg.input_channels:
            layers.append(
                ConvNormActivation(
                    cfg.input_channels,
                    cfg.expanded_channels,
                    kernel_size=1,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer
                )
            )

        stride = 1 if cfg.dilation > 1 else cfg.stride
        layers.append(
            ConvNormActivation(
                cfg.expanded_channels,
                cfg.expanded_channels,
                kernel_size=cfg.kernel,
                stride=stride,
                dilation=cfg.dilation,
                groups=cfg.expanded_channels,
                norm_layer=norm_layer,
                activation_layer=activation_layer
            )
        )

        if cfg.use_se:
            squeeze_channels = _make_divisible(cfg.expanded_channels // 4, 8)
            layers.append(se_layer(cfg.expanded_channels, squeeze_channels))

        layers.append(
            ConvNormActivation(
                cfg.expanded_channels,
                cfg.out_channels,
                kernel_size=1,
                norm_layer=norm_layer,
                activation_layer=None
            )
        )

        self.block = nn.Sequential(*layers)
        self.out_channels = cfg.out_channels
        self._is_cn = cfg.stride > 1

    def forward(self, x):
        out = self.block(x)
        if self.use_residual:
            out += x
        
        return out


@MODULE_REGISTRY.register()
class MobileNetV3(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.out_features = kwargs['out_features']
        self.out_names = kwargs['out_names']
        norm_layer = partial(nn.BatchNorm2d, eps=1e-5, momentum=1e-3)
        layers = []

        layers.append(
            ConvNormActivation(
                3, kwargs['base_channels'],
                kernel_size=3,
                stride=2,
                padding=1,
                norm_layer=norm_layer,
                activation_layer=nn.ReLU6
            )
        )

        for cfg in kwargs['inverted_residuals']:
            cfg = InvertResidualConfig(*cfg)
            layers.append(InvertedResidual(cfg, norm_layer))

        self.features = nn.Sequential(*layers)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d,)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

        if 'pretrained' in kwargs:
            checkpoint = torch.load(kwargs['pretrained'])
            self.load_state_dict(checkpoint, strict=False)

        if 'freeze_at' in kwargs and kwargs['freeze_at'] >= 0:
            for i, layer in enumerate(self.features):
                if i <= kwargs['freeze_at']:
                    _freeze(layer)

    def forward(self, x):
        outs = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in self.out_features:
                outs.append(x)
        return {
            k: v for k, v in zip(self.out_names, outs)
        }
