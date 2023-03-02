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


inline_sources = """
#include <torch/script.h>
#include <tuple>

torch::Tensor box2d_decode(const torch::Tensor& anchors,
                           const torch::Tensor& regress)
{
    const int box_dim = 4;
    int batch = anchors.size(0);
    int num_anchors = anchors.size(1);
    auto options = torch::TensorOptions().dtype(torch::kFloat32)
                                         .device(torch::kCUDA);
    
    auto bboxes = torch::empty({batch, num_anchors, box_dim}, options);
    return bboxes;
}

torch::Tensor batch_cls_nms(const torch::Tensor& cats,
                            const torch::Tensor& scores,
                            const torch::Tensor& boxes,
                            const torch::Tensor& score_thrs,
                            int64_t max_output_num,
                            double iou_thr)
{
    const int box_dim = 6;
    int batch = cats.size(0);
    auto options = torch::TensorOptions().dtype(torch::kFloat32)
                                         .device(torch::kCUDA);
    return torch::empty({batch, max_output_num, box_dim});
}

TORCH_LIBRARY(custom_ops, m) {
    m.def("box2d_decode", &box2d_decode);
    m.def("batch_cls_nms", &batch_cls_nms);
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
    return g.op('custom_domain::RetinaNetDecode',
                anchors, regress);


register_custom_op_symbolic("custom_ops::box2d_decode",
                            symbolic_box2d_decode, 13)


@parse_args('v', 'v', 'v', 'v', 'i', 'f')
def symbolic_batch_cls_nms(g, cats, scores, boxes,
                           score_thr,
                           max_output_num, iou_thr):
    return g.op('custom_domain::BatchClsNms2D',
                cats, scores, boxes, score_thr,
                max_output_num_i=max_output_num,
                iou_thr_f=iou_thr)


register_custom_op_symbolic("custom_ops::batch_cls_nms",
                            symbolic_batch_cls_nms, 13)


class Wrapper(nn.Module):
    def __init__(self, model, conf_thr, max_output_num):
        super().__init__()
        self.model = model
        self.conf_thr = torch.tensor(conf_thr)
        self.max_output_num = max_output_num

        self.iou_thr = model.iou_thr
        self.num_classes = model.num_classes
        self.num_anchors = model.num_anchors
        self.topk_num = model.num_preds
        self.lvl_spatials = model.spatials

        self.anchors = model.anchors

    def decode(self, logits, regress):
        f = lambda x, s: \
            x.reshape(-1, self.num_anchors * self.num_classes, s)\
             .permute(0, 2, 1)\
             .reshape(-1, s * self.num_anchors, self.num_classes)

        logits = [f(x, s) for x, s in zip(logits, self.lvl_spatials)]
        logits = torch.cat(logits, dim=1)
        logits = logits.sigmoid()

        f = lambda x, s: \
            x.reshape(-1, self.num_anchors * 4, s)\
             .permute(0, 2, 1)\
             .reshape(-1, s * self.num_anchors, 4)
        
        regress = [f(x, s) for x, s in zip(regress, self.lvl_spatials)]
        regress = torch.cat(regress, dim=1)

        batch_cls = []
        batch_scores = []
        batch_anchors = []
        batch_regress = []
        for cls, reg in zip(logits, regress):
            sorted_scores, sorted_idx = torch.topk(cls.flatten(), self.topk_num)
            sorted_anchor_idx = torch.div(sorted_idx, self.num_classes, rounding_mode='trunc')
            sorted_cls = torch.remainder(sorted_idx, self.num_classes)

            batch_cls.append(sorted_cls)
            batch_scores.append(sorted_scores)

            batch_anchors.append(self.anchors[sorted_anchor_idx])
            batch_regress.append(reg[sorted_anchor_idx])

        labels = torch.stack(batch_cls)
        scores = torch.stack(batch_scores)
        anchors = torch.stack(batch_anchors)
        regress = torch.stack(batch_regress)

        boxes = torch.ops.custom_ops.box2d_decode(anchors, regress)
        return labels, scores, boxes

    def forward(self, x):
        outs = self.model(x)
        logits, regress = outs
        labels, scores, boxes = self.decode(logits, regress)
        objs = torch.ops.custom_ops.batch_cls_nms(
            labels, scores, boxes, self.conf_thr,
            self.max_output_num, self.iou_thr
        )
        return objs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
                        help='path to config file')
    parser.add_argument('--ckpt', type=str, required=True,
                        help='ckpt name')
    parser.add_argument('-o', '--output', type=str,
                        default='out.onnx',
                        help='output onnx file path')
    parser.add_argument("--int8", action="store_true",
                        default=False)
    args = parser.parse_args()

    if not args.int8:
        quant_modules.deactivate()

    from modules.meta_arch.retinanet import RetinaNet
    from configs import Config
    
    cfg = Config(Config.load_yaml_with_base(args.config))
    ccfg = cfg.MODEL
    input_size = ccfg.PREDICT.INPUT_SIZE
    model = RetinaNet(ccfg)

    ccfg = cfg.SAVE
    ckpt = os.path.join(ccfg.OUTPUT_PATH, args.ckpt)
    ckpt = torch.load(ckpt, map_location='cpu')
    model.load_state_dict(ckpt['model'])
    model.eval()
    model.cuda()

    ccfg = cfg.MODEL.PREDICT

    m = Wrapper(model, [0.2, 0.2, 0.2], 50)
    h, w = input_size
    dummy = torch.empty((1, 3, h, w), dtype=torch.float32, device="cuda")
    torch.onnx.export(m, dummy, args.output,
                      input_names=['input'],
                      output_names=['dets'],
                      opset_version=13)
