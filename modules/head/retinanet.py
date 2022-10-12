# Copyright (c) 2022 by zhuxiaolong, All rights Reserved.
from ..build import MODULE_REGISTRY
import torch.nn as nn
import math


class HeadBlock(nn.Module):
    box_dim = 4
    prior_prob = 0.01

    def __init__(self, channels, num_classes, num_anchors):
        super().__init__()

        cls_subnet = []
        box_subnet = []
        for in_channels, out_channels in zip(channels[:-1], channels[1:]):
            cls_subnet.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
                nn.BatchNorm2d(out_channels, momentum=1e-3),
                nn.ReLU6(True)
            ))
            box_subnet.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
                nn.BatchNorm2d(out_channels, momentum=1e-3),
                nn.ReLU6(True)
            ))

        self.cls_subnet = nn.Sequential(*cls_subnet)
        self.box_subnet = nn.Sequential(*box_subnet)

        num_features = channels[-1]
        self.cls_pred = nn.Conv2d(num_features, num_anchors * num_classes, 1)
        self.box_pred = nn.Conv2d(num_features, num_anchors * HeadBlock.box_dim, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        bias_val = -(math.log((1 - HeadBlock.prior_prob) / HeadBlock.prior_prob))
        nn.init.constant_(self.cls_pred.bias, bias_val)

    def forward(self, x):
        logits = self.cls_pred(self.cls_subnet(x))
        regress = self.box_pred(self.box_subnet(x))
        return logits, regress


@MODULE_REGISTRY.register()
class RetinaNetHead(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        num_classes = kwargs['num_classes']
        num_anchors = kwargs['num_anchors']

        self.shared = kwargs['shared']
        self.in_names = kwargs['in_names']
        self.out_names = kwargs['out_names']

        conv_channels = kwargs['conv_channels']
        if self.shared:
            self.head = HeadBlock(conv_channels, num_classes, num_anchors)
        else:
            heads = [
                HeadBlock(channels, num_classes, num_anchors)
                for channels in conv_channels
            ]
            self.heads = nn.ModuleList(heads)
        
    def forward(self, xs):
        logits = []
        regress = []

        for i, x in enumerate([xs[k] for k in self.in_names]):
            if self.shared:
                cls_pred, box_pred = self.head(x)
            else:
                cls_pred, box_pred = self.heads[i](x)

            logits.append(cls_pred)
            regress.append(box_pred)

        return {
            k: v for k, v in zip(self.out_names, [logits, regress])
        }
