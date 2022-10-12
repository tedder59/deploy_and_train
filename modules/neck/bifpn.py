# Copyright (c) 2022 by zhuxiaolong, All rights Reserved.
from ..build import MODULE_REGISTRY
import torch.nn as nn
import torch


class SeparableConvBlock(nn.Module):
    def __init__(self, channels, norm=True, activation=False):
        super().__init__()

        self.depthwise_conv = nn.Conv2d(channels, channels, 3,
                                        padding=1,
                                        groups=channels,
                                        bias=False)
        self.pointwise_conv = nn.Conv2d(channels, channels, 1,
                                        bias=False if norm else True)

        self.norm = norm
        if self.norm:
            self.bn = nn.BatchNorm2d(channels, 1e-3, 1e-2)
        
        self.activation = activation
        if self.activation:
            self.silu = nn.SiLU(True)

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)

        if self.norm:
            x = self.bn(x)

        if self.activation:
            x = self.silu(x)

        return x


@MODULE_REGISTRY.register()
class BiFPN(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.eps = 1e-5
        self.in_names = kwargs['in_names']
        self.out_names = kwargs['out_names']

        num_channels = kwargs['num_channels']
        self.conv6_up = SeparableConvBlock(num_channels)
        self.conv5_up = SeparableConvBlock(num_channels)
        self.conv4_up = SeparableConvBlock(num_channels)
        self.conv3_up = SeparableConvBlock(num_channels)

        self.conv4_down = SeparableConvBlock(num_channels)
        self.conv5_down = SeparableConvBlock(num_channels)
        self.conv6_down = SeparableConvBlock(num_channels)
        self.conv7_down = SeparableConvBlock(num_channels)

        self.upsample = nn.Upsample(scale_factor=2)
        self.downsample = nn.MaxPool2d(3, 2, 1)
        self.silu = nn.SiLU(True)

        conv_channels = kwargs['conv_channels']
        self.p5_proj_1 = nn.Sequential(
            nn.Conv2d(conv_channels[2], num_channels, 1, bias=False),
            nn.BatchNorm2d(num_channels, 1e-3, 1e-2)
        )
        self.p4_proj_1 = nn.Sequential(
            nn.Conv2d(conv_channels[1], num_channels, 1, bias=False),
            nn.BatchNorm2d(num_channels, 1e-3, 1e-2)
        )
        self.p5_proj_2 = nn.Sequential(
            nn.Conv2d(conv_channels[2], num_channels, 1, bias=False),
            nn.BatchNorm2d(num_channels, 1e-3, 1e-2)
        )
        self.p4_proj_2 = nn.Sequential(
            nn.Conv2d(conv_channels[1], num_channels, 1, bias=False),
            nn.BatchNorm2d(num_channels, 1e-3, 1e-2)
        )
        self.p3_proj = nn.Sequential(
            nn.Conv2d(conv_channels[0], num_channels, 1, bias=False),
            nn.BatchNorm2d(num_channels, 1e-3, 1e-2)
        )

        self.p5_to_p6 = nn.Sequential(
            nn.Conv2d(conv_channels[2], num_channels, 1, bias=False),
            nn.BatchNorm2d(num_channels, 1e-3, 1e-2),
            nn.MaxPool2d(3, 2, 1)
        )

        self.p6_to_p7 = nn.MaxPool2d(3, 2, 1)

        self.p6_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32))
        self.p5_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32))
        self.p4_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32))
        self.p3_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32))
        
        self.p7_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32))
        self.p6_w2 = nn.Parameter(torch.ones(2, dtype=torch.float32))
        self.p5_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32))
        self.p4_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32))

        self.relu = nn.ReLU()

    def forward(self, xs):
        p3, p4, p5 = [xs[k] for k in self.in_names]

        p6_in = self.p5_to_p6(p5)
        p7_in = self.p6_to_p7(p6_in)

        p3_in = self.p3_proj(p3)
        p4_in = self.p4_proj_1(p4)
        p5_in = self.p5_proj_1(p5)

        p6_w = self.relu(self.p6_w1)
        w = p6_w / (torch.sum(p6_w) + self.eps)
        p6_up = self.silu(w[0] * p6_in + w[1] * self.upsample(p7_in))
        p6_up = self.conv6_up(p6_up)

        p5_w = self.relu(self.p5_w1)
        w = p5_w / (torch.sum(p5_w) + self.eps)
        p5_up = self.silu(w[0] * p5_in + w[1] * self.upsample(p6_up))
        p5_up = self.conv5_up(p5_up)

        p4_w = self.relu(self.p4_w1)
        w = p4_w / (torch.sum(p4_w) + self.eps)
        p4_up = self.silu(w[0] * p4_in + w[1] * self.upsample(p5_up))
        p4_up = self.conv4_up(p4_up)

        p3_w = self.relu(self.p3_w1)
        w = p3_w / (torch.sum(p3_w) + self.eps)
        p3_up = self.silu(w[0] * p3_in + w[1] * self.upsample(p4_up))
        p3_out = self.conv3_up(p3_up)

        p4_in = self.p4_proj_2(p4)
        p5_in = self.p5_proj_2(p5)

        p4_w = self.relu(self.p4_w2)
        w = p4_w / (torch.sum(p4_w) + self.eps)
        p4_out = self.silu(w[0] * p4_in + w[1] * p4_up + w[2] * self.downsample(p3_out))
        p4_out = self.conv4_down(p4_out)

        p5_w = self.relu(self.p5_w2)
        w = p5_w / (torch.sum(p5_w) + self.eps)
        p5_out = self.silu(w[0] * p5_in + w[1] * p5_up + w[2] * self.downsample(p4_out))
        p5_out = self.conv5_down(p5_out)

        p6_w = self.relu(self.p6_w2)
        w = p6_w / (torch.sum(p6_w) + self.eps)
        p6_out = self.silu(w[0] * p6_up + w[1] * self.downsample(p5_out))
        p6_out = self.conv6_down(p6_out)

        p7_w = self.relu(self.p7_w1)
        w = p7_w / (torch.sum(p7_w) + self.eps)
        p7_out = self.silu(w[0] * p7_in + w[1] * self.downsample(p6_out))
        p7_out = self.conv7_down(p7_out)

        return {
            k: v
            for k, v in zip(self.out_names,
                            [p3_out, p4_out, p5_out, p6_out, p7_out])
        }
