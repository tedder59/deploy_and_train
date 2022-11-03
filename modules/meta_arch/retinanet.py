# Copyright (c) 2022 by zhuxiaolong, All rights Reserved.
from ..backbone import *
from ..neck import *
from ..head import *
from ..build import build_module
from ..build import META_ARCH_REGISTRY, CRITERION_REGISTRY, VISUALIZER_REGISTRY
from fvcore.nn import sigmoid_focal_loss_jit, smooth_l1_loss
from torchvision.ops import nms
import ignite.distributed as idist
import torch.nn.functional as F
import torch.nn as nn
import torch
import math
import cv2


def _box_decode(anchors, deltas, weights=[1.0, 1.0, 1.0, 1.0], 
                scale_clamp=math.log(1000.0 / 16)):
    widths = anchors[:, 2] - anchors[:, 0]
    heights = anchors[:, 3] - anchors[:, 1]

    ctr_x = (anchors[:, 0] + anchors[:, 2]) * 0.5
    ctr_y = (anchors[:, 1] + anchors[:, 3]) * 0.5

    wx, wy, ww, wh = [1.0 / x for x in weights]
    dx = deltas[:, 0] * wx
    dy = deltas[:, 1] * wy
    dw = deltas[:, 2] * ww
    dh = deltas[:, 3] * wh

    # Prevent sending too large values into torch.exp()
    dw = torch.clamp(dw, max=scale_clamp)
    dh = torch.clamp(dh, max=scale_clamp)

    pred_ctr_x = dx * widths + ctr_x
    pred_ctr_y = dy * heights + ctr_y
    pred_w = torch.exp(dw) * widths
    pred_h = torch.exp(dh) * heights

    x1 = pred_ctr_x - 0.5 * pred_w
    y1 = pred_ctr_y - 0.5 * pred_h
    x2 = pred_ctr_x + 0.5 * pred_w
    y2 = pred_ctr_y + 0.5 * pred_h

    pred_boxes = torch.stack((x1, y1, x2, y2), dim=-1)
    return pred_boxes


@META_ARCH_REGISTRY.register()
class RetinaNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.device = idist.device()

        sub_nets = []
        for module_cfg in cfg.MODULES:
            sub_nets.append(build_module(**module_cfg))
        self.sub_nets = nn.ModuleList(sub_nets)

        ccfg = cfg.PREDICT
        self.conf_thrs = torch.tensor(ccfg.SCORE_THRESHOLDS,
                                      dtype=torch.float32,
                                      device=self.device)
        self.num_preds = ccfg.NUM_OUTPUTS
        self.iou_thr = ccfg.IOU_THRESHOLD
        self.num_anchors = ccfg.NUM_ANCHORS
        self.num_classes = ccfg.NUM_CLASSES
        self.input_size = ccfg.INPUT_SIZE

        ccfg = ccfg.ANCHORS
        self.anchors = RetinaNet.generate_anchors(self.device, **ccfg)
        self.spatials = [
            x[0] * x[1] for x in ccfg.features_spatial
        ]

    def forward(self, x):
        out = self.sub_nets[0](x)
        for i in range(1, len(self.sub_nets)):
            out = self.sub_nets[i](out)
        
        return [x for x in out.values()]
    
    def predict(self, xs):
        logits, regress = xs
        fn = lambda x, s: \
            x.reshape(-1, self.num_anchors * self.num_classes, s)\
             .permute(0, 2, 1)\
             .reshape(-1, s * self.num_anchors, self.num_classes)
        
        logits = [fn(x, s) for x, s in zip(logits, self.spatials)]
        logits = torch.cat(logits, dim=1)

        fn = lambda x, s: \
            x.reshape(-1, self.num_anchors * 4, s)\
             .permute(0, 2, 1)\
             .reshape(-1, s * self.num_anchors, 4)
        
        regress = [fn(x, s) for x, s in zip(regress, self.spatials)]
        regress = torch.cat(regress, dim=1)

        logits = logits.sigmoid_()
        mask = torch.greater_equal(logits, self.conf_thrs)
        mask = torch.any(mask, dim=2)

        max_edge = max(self.input_size)
        preds = []
        for cls, bbs, m in zip(logits, regress, mask):
            cls = cls[m]
            bbs = bbs[m]
            anchors = self.anchors[m]

            scores, cats = cls.max(dim=1)
            bboxes = _box_decode(anchors, bbs)
            _min = bboxes.new([0, 0, 0, 0])
            _h = self.input_size[0] - 1
            _w = self.input_size[1] - 1
            _max = bboxes.new([_w, _h, _w, _h])
            bboxes.clamp_(_min, _max)

            offsets = cats * max_edge
            cls_bbs = bboxes + offsets[..., None]
            keep = nms(cls_bbs, scores, self.iou_thr)
            keep = keep[:self.num_preds]

            pred = bboxes.new_full([self.num_preds, 6], -1)
            num_preds = keep.numel()
            
            if num_preds > 0:
                cats = cats.to(scores.dtype)
                pred[:num_preds, 0] = cats[keep]
                pred[:num_preds, 1] = scores[keep]
                pred[:num_preds, 2:] = bboxes[keep]
            preds.append(pred)

        return [torch.stack(preds)]

    def prepare_train_batch(self, batch):
        images, cats, bboxes, _ = batch
        cvt = lambda x: x.to(self.device, non_blocking=True)
        images = cvt(images)
        cats = [cvt(x) for x in cats]
        bboxes = [cvt(x) for x in bboxes]
        return images, (cats, bboxes)

    @staticmethod
    def prepare_eval_batch(batch, device, non_blocking):
        images, images_id = batch
        cvt = lambda x: x.to(device, non_blocking=non_blocking)
        images = cvt(images)
        return images, images_id

    @staticmethod
    def generate_anchors(device, offset, sizes,
                         aspect_ratios, strides,
                         features_spatial):
        def base(sizes, aspects, device):
            anchors = []
            for edge in sizes:
                area = edge * edge
                for r in aspects:
                    w = math.sqrt(area / r) * 0.5
                    h = w * r
                    anchors.append([-w, -h, w, h])
            return torch.tensor(anchors, dtype=torch.float32,
                                device=device)
        
        outs = []
        for grid, s, edge_sizes, ratios in zip(
            features_spatial, strides, sizes, aspect_ratios
        ):
            base_anchors = base(edge_sizes, ratios, device)
            gh, gw = grid
            shift_x = torch.arange(offset * s, gw * s, s,
                                   dtype=torch.float32,
                                   device=device)
            shift_y = torch.arange(offset * s, gh * s, s,
                                   dtype=torch.float32,
                                   device=device)
            
            shift_ys, shift_xs = torch.meshgrid(shift_y, shift_x,
                                                indexing="ij")
            shift_xs = shift_xs.reshape(-1)
            shift_ys = shift_ys.reshape(-1)

            shift = torch.stack((shift_xs, shift_ys, shift_xs, shift_ys),
                                dim=1)
            anchors = shift.view(-1, 1, 4) + base_anchors.view(1, -1, 4)
            anchors = anchors.view(-1, 4)
            outs.append(anchors)

        return torch.cat(outs)


def _pairwise_intersection(bbs1, bbs2):
    inter_width_height = torch.min(bbs1[:, None, 2:], bbs2[:, 2:])\
        - torch.max(bbs1[:, None, :2], bbs2[:, :2])
    inter_width_height.clamp_(min=0)
    intersections = inter_width_height.prod(dim=2)
    return intersections


def _pairwise_iou(bbs1, bbs2):
    fn = lambda x: (x[:, 2:] - x[:, :2]).prod(dim=1)
    A1, A2 = [fn(x) for x in (bbs1, bbs2)]
    inter = _pairwise_intersection(bbs1, bbs2)
    return inter / (A1[:, None] + A2 - inter + 1e-5)


def _box_deltas(anchors, gt_boxes, weights=[1.0, 1.0, 1.0, 1.0]):
    widths = anchors[:, 2] - anchors[:, 0]
    heights = anchors[:, 3] - anchors[:, 1]

    ctr_x = (anchors[:, 0] + anchors[:, 2]) * 0.5
    ctr_y = (anchors[:, 1] + anchors[:, 3]) * 0.5

    target_widths = gt_boxes[:, 2] - gt_boxes[:, 0]
    target_heights = gt_boxes[:, 3] - gt_boxes[:, 1]

    target_ctr_x = gt_boxes[:, 0] + 0.5 * target_widths
    target_ctr_y = gt_boxes[:, 1] + 0.5 * target_heights

    wx, wy, ww, wh = weights
    dx = wx * (target_ctr_x - ctr_x) / widths
    dy = wy * (target_ctr_y - ctr_y) / heights
    dw = ww * torch.log(target_widths / widths)
    dh = wh * torch.log(target_heights / heights)

    deltas = torch.stack((dx, dy, dw, dh), dim=1)
    return deltas


@CRITERION_REGISTRY.register()
class RetinaNetCriterion:
    def __init__(self, cfg):
        self.device = idist.device()
        neg_iou_thrs = [x[0] for x in cfg.IOU_THRESHOLDS]
        pos_iou_thrs = [x[1] for x in cfg.IOU_THRESHOLDS]

        self.neg_iou_thrs = torch.tensor(neg_iou_thrs,
                                         dtype=torch.float32,
                                         device=self.device)
        self.pos_iou_thrs = torch.tensor(pos_iou_thrs,
                                         dtype=torch.float32,
                                         device=self.device)

        self.focal_loss_gamma = cfg.FOCAL_LOSS_GAMMA
        self.focal_loss_alpha = cfg.FOCAL_LOSS_ALPHA
        self.num_classes = cfg.NUM_CLASSES
        self.num_anchors = cfg.NUM_ANCHORS

        ccfg = cfg.ANCHORS
        self.anchors = RetinaNet.generate_anchors(
            self.device, **ccfg
        )
        self.spatials = [
            x[0] * x[1] for x in ccfg.features_spatial
        ]
        
    def __call__(self, out, y):
        logits, regress = out
        fn = lambda x, s: \
            x.reshape(-1, self.num_anchors * self.num_classes, s)\
             .permute(0, 2, 1)\
             .reshape(-1, s * self.num_anchors, self.num_classes)
        
        logits = [fn(x, s) for x, s in zip(logits, self.spatials)]
        logits = torch.cat(logits, dim=1)

        fn = lambda x, s: \
            x.reshape(-1, self.num_anchors * 4, s)\
             .permute(0, 2, 1)\
             .reshape(-1, s * self.num_anchors, 4)
        
        regress = [fn(x, s) for x, s in zip(regress, self.spatials)]
        regress = torch.cat(regress, dim=1)

        gt_labels, gt_bboxes = self.label_anchors(y)
        losses, log_dict = self.losses(logits, regress, gt_labels, gt_bboxes)
        return losses, log_dict

    @torch.no_grad()
    def label_anchors(self, tgts):
        cats, bboxes = tgts
        gt_labels = []
        gt_bboxes = []

        for cats_per_image, bbs_per_image in zip(cats, bboxes):
            match_matrix = _pairwise_iou(self.anchors, bbs_per_image)
            match_vals, matches = match_matrix.max(dim=1)

            cats_per_image = cats_per_image.to(dtype=torch.long)
            gt_labels_per_image = matches.new_full(
                matches.size(), -1, dtype=cats_per_image.dtype
            )

            neg_iou_thrs = self.neg_iou_thrs[cats_per_image]
            neg_iou_thrs = neg_iou_thrs[matches]
            gt_labels_per_image[match_vals < neg_iou_thrs] = self.num_classes
            
            pos_iou_thrs = self.pos_iou_thrs[cats_per_image]
            pos_iou_thrs = pos_iou_thrs[matches]
            pos_mask = torch.greater(match_vals, pos_iou_thrs)
            gt_labels_per_image[pos_mask] = cats_per_image[matches[pos_mask]]
            gt_labels.append(gt_labels_per_image)

            gt_bboxes_per_image = bbs_per_image[matches]
            gt_bboxes.append(gt_bboxes_per_image)

        gt_labels = torch.stack(gt_labels)
        gt_bboxes = torch.stack(gt_bboxes)
        return gt_labels, gt_bboxes

    def losses(self, logits, regress, gt_labels, gt_bboxes):
        num_classes = self.num_classes
        valid_mask = gt_labels >= 0
        pos_mask = torch.logical_and(valid_mask, (gt_labels != num_classes))
        num_pos = pos_mask.sum().detach().item()

        gt_labels = gt_labels[valid_mask]
        gt_labels = F.one_hot(gt_labels, num_classes + 1)[:, :-1]
        gt_labels = gt_labels.to(dtype=logits.dtype)
        
        loss_cls = sigmoid_focal_loss_jit(
            logits[valid_mask], gt_labels,
            alpha=self.focal_loss_alpha,
            gamma=self.focal_loss_gamma,
            reduction="sum"
        )

        if num_pos <= 0:
            loss_reg = torch.tensor([0], dtype=torch.float32,
                                    device=gt_bboxes.device)
        else:
            batch = pos_mask.shape[0]
            anchors = self.anchors.tile(batch, 1, 1)[pos_mask]
            gt_regress = gt_bboxes[pos_mask]
            regress = regress[pos_mask]
            deltas = _box_deltas(anchors, gt_regress)
            loss_reg = smooth_l1_loss(regress, deltas, 0.1, "sum")
        
        total_loss = (loss_cls + loss_reg) / num_pos
        return total_loss, {
            "total_loss": total_loss.detach().item(),
            "cls_loss": loss_cls.detach().item(),
            "reg_loss": loss_reg.detach().item(),
            "num_positive": num_pos
        }


@VISUALIZER_REGISTRY.register()
class RetinaNetVisualizer:
    def __init__(self, cfg):
        self.classes_name = cfg.CLASSES_NAME
        self.input_size = cfg.INPUT_SIZE
        self.classes_color = cfg.CLASSES_COLOR
        self.num_images = cfg.NUM_IMAGES

    def __call__(self, batch, outs):
        images_file = batch[3]
        outs = outs[0].detach().cpu()

        out_dict = {}
        for i, image_file, pred in zip(range(self.num_images), images_file, outs):
            im = cv2.imread(image_file, cv2.IMREAD_COLOR)
            h, w = im.shape[:2]
            sx = float(w) / self.input_size[1]
            sy = float(h) / self.input_size[0]

            for obj in pred:
                c, s, x1, y1, x2, y2 = obj.tolist()
                x1, x2 = [x * sx for x in (x1, x2)]
                y1, y2 = [x * sy for x in (y1, y2)]
                c, x1, y1, x2, y2 = [
                    round(x) for x in (c, x1, y1, x2, y2)
                ]

                if c < 0:
                    break

                cv2.rectangle(im, (x1, y1), (x2, y2),
                              self.classes_color[c])
                text = '{}: {:.2f}'.format(self.classes_name[c], s)
                cv2.putText(im, text, (x1, y1),
                            cv2.FONT_HERSHEY_COMPLEX,
                            1, self.classes_color[c])
            out_dict[image_file] = im

        return out_dict
