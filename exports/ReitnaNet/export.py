from curses import wrapper
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

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> efficient_nms(
    const torch::Tensor& regress, const torch::Tensor& logits,
    const torch::Tensor& anchors, double conf_thr,
    double iou_thr, int64_t max_output_num,
    int64_t num_classes
) {
    int batch_size = regress.size(0);

    auto options = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    auto num_detections = torch::empty({batch_size, 1}, options);
    auto detection_classes = torch::empty({batch_size, max_output_num}, options);

    options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto detection_boxes = torch::empty({batch_size, max_output_num, 4}, options);
    auto detection_scores = torch::empty({batch_size, max_output_num}, options);
    
    return std::make_tuple(num_detections, detection_boxes,
                           detection_scores, detection_classes);
}

TORCH_LIBRARY(custom_ops, m) {
    m.def("efficient_nms", &efficient_nms);
}
"""

cpp_extension.load_inline(
    name="inline_extensions",
    cpp_sources=inline_sources,
    is_python_module=False,
    verbose=True
)

@parse_args('v', 'v', 'v', 'f', 'f', 'i', 'i')
def symbolic_efficient_nms(g, regress, scores, anchors, conf_thr, iou_thr, max_output_num, num_classes):
    return g.op('custom_domain::EfficientNMS_TRT',
                regress, scores, anchors,
                score_threshold_f=conf_thr,
                iou_thresholod_f=iou_thr,
                max_output_boxes_i=max_output_num,
                background_class_i=num_classes,
                score_activation_i=0,
                box_coding_i=0,
                outputs=4)

register_custom_op_symbolic("custom_ops::efficient_nms", symbolic_efficient_nms, 13)

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

        logits = logits.sigmoid_()
        # mask = torch.greater_equal(logits, self.conf_thr)
        # mask = torch.any(mask, dim=2)
        # mask = mask.tile(1, 1, self.model.num_classes)

        # logits = logits[mask]
        # regress = regress[mask]

        num_dets, det_boxes, det_scores, det_classes = \
            torch.ops.custom_ops.efficient_nms(
                regress, logits, self.anchors, self.conf_thr, self.iou_thr,
                self.max_output_num, self.num_classes
            )
        
        det_classes = det_classes.to(det_boxes.dtype)
        dets = torch.cat(
            [det_classes[..., None], det_scores[..., None], det_boxes],
            dim=2
        )
        return num_dets, dets


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
