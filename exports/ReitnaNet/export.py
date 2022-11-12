from pytorch_quantization import nn as quant_nn
from pytorch_quantization.tensor_quant import QuantDescriptor


quant_desc_input = QuantDescriptor(calib_method='histogram')
quant_nn.QuantConv2d.set_default_quant_desc_input(quant_desc_input)
quant_nn.QuantLinear.set_default_quant_desc_input(quant_desc_input)


from pytorch_quantization import quant_modules
quant_nn.TensorQuantizer.use_fb_fake_quant = True
quant_modules.initialize()


from torch.onnx import register_custom_op_symbolic
from torch.onnx.symbolic_helper import parse_args
from torch.utils import cpp_extension
import torch.nn as nn
import argparse
import torch
import os


from importmagician import import_from
with import_from(f'{os.path.dirname(__file__)}/../..'):
    from modules.meta_arch.retinanet import RetinaNet
    from configs import Config


inline_sources = """
#include <torch/script.h>
#include <tuple>

torch::Tensor box2d_decode(const torch::Tensor& anchors,
                           const torch::Tensor& regress)
{
    const int box_dim = 4;
    int num_anchors = anchors.size(0);
    auto options = torch::TensorOptions().dtype(torch::kFloat32)
                                         .device(torch::kCUDA);
    
    auto bboxes = torch::empty({num_anchors, box_dim}, options);
    return bboxes;
}

torch::Tensor batch_cls_nms_2d(const torch::Tensor& batch_idxes,
                               const torch::Tensor& cats,
                               const torch::Tensor& scores,
                               const torch::Tensor& bboxes,
                               int64_t batch_num,
                               double iou_thr,
                               double max_edge,
                               int64_t max_output_num)
{
    const int box_dim = 6;
    auto options = torch::TensorOptions().dtype(torch::kFloat32)
                                         .device(torch::kCUDA);
    return torch::empty({batch_num, max_output_num, box_dim});
}

TORCH_LIBRARY(custom_ops, m) {
    m.def("box2d_decode", &box2d_decode);
    m.def("batch_cls_nms_2d", &batch_cls_nms_2d);
}
"""
cpp_extension.load_inline(
    name="inline_extensions",
    cpp_sources=inline_sources,
    is_python_module=False,
    verbose=True
)


@parse_args('v', 'v')
def symbolic_box2d_decode(g, anchors, regress):
    return g.op('custom_domain::Box2dDecode',
                anchors, regress);


register_custom_op_symbolic("custom_ops::box2d_decode",
                            symbolic_box2d_decode, 13)


@parse_args('v', 'v', 'v', 'v', 'i', 'f', 'f', 'i')
def symbolic_efficient_nms(g, batch_idxes, cats, scores, bboxes,
                           batch_size, iou_thr, max_edge,
                           max_output_num):
    return g.op('custom_domain::BatchClsNMS2d',
                batch_idxes, cats, scores, bboxes, batch_size,
                iou_thr_f=iou_thr,
                max_edge_f=max_edge,
                max_output_num_i=max_output_num)


register_custom_op_symbolic("custom_ops::batch_cls_nms_2d",
                            symbolic_efficient_nms, 13)


class Wrapper(nn.Module):
    def __init__(self, model, conf_thr, iou_thr,
                 max_output_num, num_classes,
                 batch_size=1):
        super().__init__()
        self.model = model
        self.conf_thr = conf_thr
        self.iou_thr = iou_thr
        self.max_output_num = max_output_num
        self.num_classes = num_classes

        self.anchors = self.model.anchors.tile(batch_size, 1, 1)
        self.max_edge = float(max(self.model.input_size))

    def forward(self, x):
        outs = self.model(x)
        logits, regress = outs

        fn = lambda x, s: \
            x.reshape(-1, self.model.num_anchors * self.model.num_classes, s)\
             .permute(0, 2, 1)\
             .reshape(-1, s * self.model.num_anchors, self.model.num_classes)

        logits = [fn(x, s) for x, s in zip(logits, self.model.spatials)]
        logits = torch.cat(logits, dim=1)

        fn = lambda x, s: \
            x.reshape(-1, self.model.num_anchors * 4, s)\
             .permute(0, 2, 1)\
             .reshape(-1, s * self.model.num_anchors, 4)

        regress = [fn(x, s) for x, s in zip(regress, self.model.spatials)]
        regress = torch.cat(regress, dim=1)
        scores = logits.sigmoid_()

        mask = torch.greater_equal(scores, self.conf_thrs)
        mask = torch.any(mask, dim=2, keepdim=True)

        batch_num = logits.shape[0]
        batch_anchors = self.anchors.tile(batch_num, 1, 1)
        scores = scores[mask]
        regress = regress[mask]
        anchors = batch_anchors[mask]

        anchor_num = logits.shape[1]
        batch_idxes = torch.arange(batch_num)[..., None]
        batch_idxes = batch_idxes.tile(1, anchor_num)[mask]

        scores, cats = scores.max(dim=1)
        bboxes = torch.ops.custom_ops.box2d_decode(anchors, regress)
        dets = torch.ops.custom_ops.batch_cls_nms_2d(
            batch_idxes, cats, scores, bboxes, batch_num,
            self.iou_thr, self.max_edge, self.max_output_num
        )

        return dets, outs[0], outs[1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
                        help='path to config file')
    parser.add_argument('-o', '--output', type=str,
                        default='out.onnx',
                        help='output onnx file path')
    parser.add_argument("--int8", action="store_true",
                        default=False)
    args = parser.parse_args()

    if not args.int8:
        quant_modules.deactivate()

    cfg = Config(Config.load_yaml_with_base(args.config))
    ccfg = cfg.MODEL
    input_size = ccfg.PREDICT.INPUT_SIZE
    model = RetinaNet(ccfg)

    ccfg = cfg.SAVE
    ckpt = torch.load(ccfg.RESUME, map_location='cpu')
    model.load_state_dict(ckpt['model'])
    model.eval()
    model.cuda()

    ccfg = cfg.MODEL.PREDICT

    m = Wrapper(model, ccfg.SCORE_THRESHOLDS[0],
                ccfg.IOU_THRESHOLD,
                ccfg.NUM_OUTPUTS,
                ccfg.NUM_CLASSES)
    h, w = input_size
    dummy = torch.empty((1, 3, h, w), dtype=torch.float32, device="cuda")
    torch.onnx.export(m, dummy, args.output,
                      input_names=['input'],
                      output_names=['num_dets', 'dets'],
                      opset_version=13)
