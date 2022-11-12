# Copyright (c) 2022 by zhuxiaolong, All rights Reserved.
from torch.utils.cpp_extension import load
from torch.nn.modules.utils import _pair
from torch.nn.parameter import Parameter
from torch.autograd import Function
from torchvision.ops.deform_conv import DeformConv2d
import torch.nn.init as init
import torch.nn as nn
import torch
import math


# deformable_conv2d = load(name="deformable_conv2d",
#                          sources=['csrc/deformable_conv2d.cpp',
#                                   'csrc/deformable_conv2d.cu'])

# class DeformableConv2dFunction(Function):
#     @staticmethod
#     def forward(ctx, input, offset, mask, weight, bias):
#         output = deform_conv2d.forward(input, offset, mask, weight, bias)
#         ctx.save_for_backward()

#     @staticmethod
#     def backward(ctx, grad_output):
#         pass


# deform_conv2d = DeformableConv2dFunction.apply


# class DeformConv2d(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size,
#                  stride=1, padding=0, dilation=1, groups=1,
#                  bias=True) -> None:
#         super().__init__()

#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.kernel_size = _pair(kernel_size)
#         self.stride = _pair(stride)
#         self.padding = _pair(padding)
#         self.dilation = _pair(dilation)
#         self.groups = groups

#         self.weight = Parameter(
#             torch.empty(self.out_channels, self.in_channels // self.groups,
#                         self.kernel_size[0], self.kernel_size[1])
#         )

#         if bias:
#             self.bias = Parameter(torch.empty(self.out_channels))
#         else:
#             self.register_parameter("bias", None)
        
#         self.reset_parameters()

#     def reset_parameters(self):
#         init.kaiming_normal_(self.weight, a=math.sqrt(5))
        
#         if self.bias is not None:
#             fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
#             bound = 1 / math.sqrt(fan_in)
#             init.uniform_(self.bias, -bound, bound)

#     def forward(self, input, offset, mask):
#         return deform_conv2d(input, offset, mask, self.weight, self.bias)


class DCN(DeformConv2d):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1,
                 bias=True):
        super().__init__(in_channels, out_channels, kernel_size,
                         stride, padding, dilation, groups, bias)
        
        self.conv_offset = nn.Conv2d(
            self.in_channels,
            self.groups * 2 * self.kernel_size[0] * self.kernel_size[1],
            kernel_size=self.kernel_size,
            stride=self.stride, padding=self.padding,
            bias=True
        )

        self.reset_offset()

    def reset_offset(self):
        self.conv_offset.weight.zero_()
        self.conv_offset.bias.zero_()

    def forward(self, input):
        offset = self.conv_offset(input)
        return DeformConv2d.forward(self, input, offset, None)


class DCNV2(DCN):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1,
                 bias=True):
        super().__init__(in_channels, out_channels, kernel_size,
                         stride, padding, dilation, groups, bias)

        self.conv_mask = nn.Conv2d(
            self.in_channels,
            self.groups * self.kernel_size[0] * self.kernel_size[1],
            kernel_size=self.kernel_size,
            stride=self.stride, padding=self.padding,
            bias=True
        )

        self.reset_mask()

    def reset_mask(self):
        self.conv_mask.weight.zero_()
        self.conv_mask.bias.zero_()

    def forward(self, input):
        offset = self.conv_offset(input)
        mask = self.conv_mask(input)
        mask = torch.sigmoid(mask)
        return DeformConv2d.forward(self, input, offset, mask)
