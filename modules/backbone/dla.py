# Copyright (c) 2022 by zhuxiaolong, All rights Reserved.
from ..build import MODULE_REGISTRY
from ..layer.deform_conv import DCN, DCNV2
from torch.nn import Conv2d
import torch.nn as nn
import numpy as np
import torch


class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=dilation,
                               dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=dilation,
                               dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out += residual

        out = self.relu(out)
        return out

    
class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, in_planes, planes, stride=1, dilation=1):
        super().__init__()
        expansion = Bottleneck.expansion
        bottle_planes = planes // expansion

        self.conv1 = nn.Conv2d(in_planes, bottle_planes,
                               kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(bottle_planes)
        self.conv2 = nn.Conv2d(bottle_planes, bottle_planes,
                               kernel_size=3, stride=stride,
                               padding=dilation, dilation=dilation,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(bottle_planes)
        self.conv3 = nn.Conv2d(bottle_planes, planes,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out += residual

        out = self.relu(out)
        return out


class BottleneckX(nn.Module):
    expansion = 2
    cardinality = 32

    def __init__(self, in_planes, planes, stride=1,
                 dilation=1):
        super().__init__()
        cardinality = BottleneckX.cardinality
        bottle_planes = planes * cardinality // 32
        
        self.conv1 = nn.Conv2d(in_planes, bottle_planes,
                               kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(bottle_planes)
        self.conv2 = nn.Conv2d(bottle_planes, bottle_planes,
                               kernel_size=3, stride=stride,
                               padding=dilation, dilation=dilation,
                               groups=cardinality, bias=False)
        self.bn2 = nn.BatchNorm2d(bottle_planes)
        self.conv3 = nn.Conv2d(bottle_planes, planes,
                               kernel_size=1, bias=False)
        
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride
    
    def forward(self, x, residual=None):
        if residual is None:
            residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out += residual

        out = self.relu(out)
        return out


class Root(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 residual):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, 1,
                              padding=(kernel_size - 1) // 2,
                              bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.residual = residual

    def forward(self, *xs):
        out = torch.cat(xs, dim=1)
        out = self.conv(out)
        out = self.bn(out)
        
        if self.residual:
            out += xs[0]
        out = self.relu(out)

        return out


class Tree(nn.Module):
    def __init__(self, levels, block, in_channels, out_channels,
                 stride, dilation=1, level_root=False,
                 root_dim=0, root_kernel_size=1,
                 root_residual=False):
        super().__init__()
        if root_dim == 0:
            root_dim = 2 * out_channels
        
        if level_root:
            root_dim += in_channels

        if levels == 1:
            self.tree1 = block(in_channels, out_channels, stride,
                               dilation=dilation)
            self.tree2 = block(out_channels, out_channels, 1,
                               dilation=dilation)
        else:
            self.tree1 = Tree(levels - 1, block, in_channels,
                              out_channels, stride,
                              dilation=dilation, root_dim=0,
                              root_kernel_size=root_kernel_size,
                              root_residual=root_residual)
            self.tree2 = Tree(levels - 1, block, out_channels,
                              out_channels, 1, dilation=dilation,
                              root_dim=root_dim + out_channels,
                              root_kernel_size=root_kernel_size,
                              root_residual=root_residual)

        if levels == 1:
            self.root = Root(root_dim, out_channels,
                             root_kernel_size, root_residual)

        self.level_root = level_root
        self.root_dim = root_dim
        self.levels = levels

        self.downsample = None
        self.project = None
        if stride > 1:
            self.downsample = nn.MaxPool2d(stride, stride=stride)
        if in_channels != out_channels:
            self.project = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x, residual=None, children=None):
        children = [] if children is None else children
        bottom = self.downsample(x) if self.downsample else x
        residual = self.project(bottom) if self.project else bottom

        if self.level_root:
            children.append(bottom)

        x1 = self.tree1(x, residual=residual)
        if self.levels == 1:
            x2 = self.tree2(x1)
            x = self.root(x2, x1, *children)
        else:
            children.append(x1)
            x = self.tree2(x1, children=children)

        return x


class DLA(nn.Module):
    def __init__(self, levels, channels, block=BasicBlock,
                 residual_root=False):
        super().__init__()
        self.channels = channels
        self.base_layer = nn.Sequential(
            nn.Conv2d(3, channels[0], kernel_size=7, padding=3,
                      bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True)
        )
        self.level0 = self._make_conv_level(channels[0], channels[0],
                                            levels[0])
        self.level1 = self._make_conv_level(channels[0], channels[1],
                                            levels[1], stride=2)
        self.level2 = Tree(levels[2], block, channels[1], channels[2], 2,
                           level_root=False, root_residual=residual_root)
        self.level3 = Tree(levels[3], block, channels[2], channels[3], 2,
                           level_root=True, root_residual=residual_root)
        self.level4 = Tree(levels[4], block, channels[3], channels[4], 2,
                           level_root=True, root_residual=residual_root)
        self.level5 = Tree(levels[5], block, channels[4], channels[5], 2,
                           level_root=True, root_residual=residual_root)

    def _make_conv_level(self, in_planes, planes, convs, stride=1,
                         dilation=1):
        modules = []
        for i in range(convs):
            modules.extend([
                nn.Conv2d(in_planes, planes, kernel_size=3,
                          stride=stride if i == 0 else 1,
                          padding=dilation, dilation=dilation,
                          bias=False),
                nn.BatchNorm2d(planes),
                nn.ReLU(inplace=True)
            ])
            in_planes = planes
        return nn.Sequential(*modules)

    def forward(self, x):
        x = self.base_layer(x)
        outs = []
        for i in range(6):
            x = getattr(self, f'level{i}')(x)
            outs.append(x)
        return outs


@MODULE_REGISTRY.register()
def dla34(**kwargs):
    model = DLA([1, 1, 1, 2, 2, 1],
                [16, 32, 64, 128, 256, 512],
                block=BasicBlock)

    if 'pretrained' in kwargs:
        checkpoint = torch.load(kwargs['pretrained'])
        model.load_state_dict(checkpoint, strict=False)

    return model


class IDAUp(nn.Module):
    def __init__(self, conv, out_channels, channels, ups):
        super().__init__()
        for i, c, f in zip(range(1, len(channels)), channels[1:], ups[1:]):
            proj = nn.Sequential(
                conv(c, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            node = nn.Sequential(
                conv(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            up = nn.Upsample(scale_factor=f)

            setattr(self, f'proj_{i}', proj)
            setattr(self, f'up_{i}', up)
            setattr(self, f'node_{i}', node)

    def forward(self, layers, start, end):
        for i in range(start + 1, end):
            proj = getattr(self, f'proj_{i - start}')(layers[i])
            up = getattr(self, f'up_{i - start}')(proj)
            layers[i] = getattr(self, f'node_{i - start}')(up + layers[i - 1])
    

class DLAUp(nn.Module):
    def __init__(self, conv, start, channels, scales):
        super().__init__()
        self.start = start
        scales = np.array(scales)
        for i in range(len(channels) - 1):
            j = -i - 2
            setattr(self, f'ida_{i}',
                    IDAUp(conv, channels[j], channels[j:],
                          scales[j:] // scales[j]))
            scales[j + 1:] = scales[j]
            channels[j + 1:] = [channels[j] for _ in channels[j + 1:]]

    def forward(self, layers):
        outs = [layers[-1]]
        for i in range(len(layers) - self.start - 1):
            ida = getattr(self, f'ida_{i}')
            ida(layers, len(layers) - i - 2, len(layers))
            outs.insert(0, layers[-1])
        return outs


@MODULE_REGISTRY.register()
class DLASeg(nn.Module):
    def __init__(self, conv_name, base_name, pretrained=None,
                 first_level=2, last_level=5):
        super().__init__()
        self.first_level = first_level
        self.last_level = last_level

        conv = globals()[conv_name]
        self.base = globals()[base_name](pretrained=pretrained)
        channels = self.base.channels
        scales = [
            2 ** i for i in range(len(channels) - self.first_level)
        ]
        self.dla_up = DLAUp(conv, first_level,
                            channels[self.first_level:], scales)
        
        scales = [
            2 ** i for i in range(self.last_level - self.first_level)
        ]
        self.ida_up = IDAUp(conv, channels[self.first_level],
                            channels[self.first_level:self.last_level],
                            scales)

    def forward(self, x):
        x = self.base(x)
        x = self.dla_up(x)

        outs = []
        for i in range(self.last_level - self.first_level):
            outs.append(x[i])
        
        self.ida_up(outs, 0, len(outs))
        return outs[-1]
