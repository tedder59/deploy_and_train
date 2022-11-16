from pytorch_quantization import nn as quant_nn
from pytorch_quantization.tensor_quant import QuantDescriptor


quant_desc_input = QuantDescriptor(calib_method='histogram')
quant_nn.QuantConv2d.set_default_quant_desc_input(quant_desc_input)
quant_nn.QuantLinear.set_default_quant_desc_input(quant_desc_input)


from pytorch_quantization import quant_modules
quant_nn.TensorQuantizer.use_fb_fake_quant = True
# quant_modules.initialize()


from torch.onnx import register_custom_op_symbolic
from torch.onnx.symbolic_helper import parse_args
from torch.utils import cpp_extension
import torch.nn.functional as F
import torch.nn as nn
import argparse
import torch
import os


from importmagician import import_from
with import_from(f'{os.path.dirname(os.path.abspath(__file__))}/../..'):
    from modules.meta_arch.centernet import CenterNet
    from configs import Config


inline_sources = """
#include <torch/script.h>

torch::Tensor centernet_decode(const torch::Tensor& topk_inds,
                               const torch::Tensor& topk_scores,
                               const torch::Tensor& reg,
                               const torch::Tensor& wh)
{
    int batch = topk_inds.size(0);
    int num_output = topk_inds.size(1);
    auto options = torch::TensorOptions().dtype(torch::kFloat32)
                                         .device(torch::kCPU);
    auto out = torch::empty({batch, num_output, 6});
    return out;
}

TORCH_LIBRARY(custom_ops, m) {
    m.def("centernet_decode", &centernet_decode);
}
"""
cpp_extension.load_inline(
    name="inline_extensions",
    cpp_sources=inline_sources,
    is_python_module=False,
    verbose=True
)


@parse_args('v', 'v', 'v', 'v')
def symbolic_centernet_decode(g, topk_inds, topk_scores, reg, wh):
    return g.op('custom_domain::CenterNetDecode',
                topk_inds, topk_scores, reg, wh);


register_custom_op_symbolic("custom_ops::centernet_decode",
                            symbolic_centernet_decode, 13)


@parse_args('v', 'v', 'v', 'v', 'v', 'i', 'i', 'i', 'i', 'i', 'i', 'i', 'i', 'b')
def symbolic_deformable_conv2d(g, input, weight, offset, mask, bias,
                               stride_h, stride_w, pad_h, pad_w,
                               dil_h, dil_w, n_weight_grps,
                               n_offset_grps, use_mask):
    return g.op('torchvision::DeformConv2d',
                input, weight, offset, mask, bias,
                stride_h_i=stride_h,
                stride_w_i=stride_w,
                pad_h_i=pad_h,
                pad_w_i=pad_w,
                dil_h_i=dil_h,
                dil_w_i=dil_w,
                n_weight_grps_i=n_weight_grps,
                n_offset_grps_i=n_offset_grps,
                use_mask_i=use_mask)

    
register_custom_op_symbolic("torchvision::deform_conv2d",
                            symbolic_deformable_conv2d, 13)


class Wrapper(nn.Module):
    def __init__(self, model, max_output_num):
        super().__init__()
        self.model = model
        self.max_output_num = max_output_num

    def forward(self, x):
        outs = self.model(x)
        hm, wh, reg = outs
        hm = hm.sigmoid()

        hmax = F.max_pool2d(hm, (3, 3), stride=1, padding=1)
        keep = (hmax == hm).float()
        hm = keep * hm

        b = hm.shape[0]
        hm = hm.reshape(b, -1)
        topk_scores, topk_inds = torch.topk(hm, self.max_output_num)
        
        dets = torch.ops.custom_ops.centernet_decode(
            topk_inds, topk_scores, reg, wh
        )
        return dets


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
                        help='path to config file')
    parser.add_argument('-o', '--output', type=str,
                        default='out.onnx',
                        help='output onnx file path')
    args = parser.parse_args()

    cfg = Config(Config.load_yaml_with_base(args.config))
    ccfg = cfg.MODEL
    input_size = ccfg.PREDICT.INPUT_SIZE
    model = CenterNet(ccfg)

    ccfg = cfg.SAVE
    ckpt = torch.load(ccfg.RESUME, map_location='cpu')
    model.load_state_dict(ckpt['model'])
    model.eval()
    # model.cpu()

    ccfg = cfg.MODEL.PREDICT
    m = Wrapper(model, ccfg.NUM_OUTPUTS)
    h, w = input_size
    dummy = torch.empty((1, 3, h, w), dtype=torch.float32)
    torch.onnx.export(m, dummy, args.output,
                      input_names=['input'],
                      output_names=['dets'],
                      opset_version=13)
