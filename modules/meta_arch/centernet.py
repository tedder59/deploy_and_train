# Copyright (c) 2022 by zhuxiaolong, All rights Reserved.
from ..backbone import *
from ..head import *
from ..build import build_module
from ..build import META_ARCH_REGISTRY, CRITERION_REGISTRY, VISUALIZER_REGISTRY
from scipy.stats import multivariate_normal
import ignite.distributed as idist
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch
import math
import cv2


def _nms(heat, kernel=3):
    pad = (kernel - 1) // 2
    hmax = F.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return keep * heat


def _topk(scores, K=40):
    b, c, h, w = scores.shape
    topk_scores, topk_inds = torch.topk(scores.reshape(b, c, -1), K)
    topk_inds = topk_inds % (h * w)
  
    topk_score, topk_ind = torch.topk(topk_scores.reshape(b, -1), K)
    topk_cls = (topk_ind / K).long()

    topk_inds = topk_inds.reshape(b, -1).gather(1, topk_ind)
    topk_ys = (topk_inds / w).int().float()
    topk_xs = (topk_inds % w).int().float()

    return topk_score, topk_inds, topk_cls, topk_ys, topk_xs


@META_ARCH_REGISTRY.register()
class CenterNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.device = idist.device()
        self.output_names = cfg.OUTPUT_NAMES

        sub_nets = []
        for module_cfg in cfg.MODULES:
            sub_nets.append(build_module(**module_cfg))
        self.sub_nets = nn.ModuleList(sub_nets)

        ccfg = cfg.PREDICT
        self.predict = getattr(self, ccfg.NAME)
        self.conf_thrs = torch.tensor(ccfg.SCORE_THRESHOLDS,
                                      dtype=torch.float32,
                                      device=self.device)
        self.num_preds = ccfg.NUM_OUTPUTS

    def forward(self, x):
        for sub_net in self.sub_nets:
            x = sub_net(x)
        return [x[n] for n in self.output_names]
    
    @torch.no_grad()
    def ctdet_predict(self, x):
        heat, wh, reg = x
        heat = heat.sigmoid()
        heat = _nms(heat)
        scores, inds, cls, ys, xs = _topk(heat, self.num_preds)

        b, c = wh.shape[:2]
        _inds = inds.unsqueeze(2).expand(-1, -1, c)
        wh = wh.permute(0, 2, 3, 1)
        wh = wh.reshape(b, -1, c).gather(1, _inds)

        b, c = reg.shape[:2]
        _inds = inds.unsqueeze(2).expand(-1, -1, c)
        reg = reg.permute(0, 2, 3, 1)
        reg = reg.reshape(b, -1, c).gather(1, _inds)

        xs = xs[..., None] + reg[:, :, 0:1]
        ys = ys[..., None] + reg[:, :, 1:2]
        wh = wh * 0.5

        x1 = (xs - wh[:, :, 0:1])
        y1 = (ys - wh[:, :, 1:2])
        x2 = (xs + wh[:, :, 0:1])
        y2 = (ys + wh[:, :, 1:2])

        thr = self.conf_thrs[cls]
        mask = torch.less(scores, thr)
        cls[mask] = -1
        
        dets = torch.cat([
            cls.float()[..., None],
             scores[..., None],
            x1, y1, x2, y2
        ], dim=2)

        return [dets]

    def prepare_train_batch(self, batch):
        images, cats, bboxes, _ = batch
        cvt = lambda x: x.to(self.device, non_blocking=True)
        images = cvt(images)
        return images, (cats, bboxes)

    @staticmethod
    def prepare_eval_batch(batch, device, non_blocking):
        images, images_id = batch
        cvt = lambda x: x.to(device, non_blocking=non_blocking)
        images = cvt(images)
        return images, images_id


def _neg_loss(pred, gt, alpha=0.25, gamma=2):
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()
    neg_weights = torch.pow(1 - gt, 1.0 / alpha)

    pos_loss = torch.log(pred) * torch.pow(1 - pred, gamma)
    pos_loss *= pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2)
    neg_loss *= neg_weights * neg_inds

    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()
    loss = - (pos_loss + neg_loss)
    return loss


def _reg_loss(regr, gt_regr, mask):
    regr_loss = F.smooth_l1_loss(regr * mask, gt_regr * mask,
                                 reduction="sum")
    return regr_loss


def gaussian_radius(box, min_overlap=0.7):
    w, h = box[..., 0], box[..., 1]
    
    a1 = 1
    b1 = w + h
    c1 = w * h * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 - sq1) / (2 * a1)

    a2 = 4
    b2 = 2 * (h + w)
    c2 = (1 - min_overlap) * w * h
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 - sq2) / (2 * a2)

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (h + w)
    c3 = (min_overlap - 1) * w * h
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / (2 * a3)

    return np.minimum(r1, np.minimum(r2, r3))


def gaussian2D(shape, sigma):
    m, n = shape
    y, x = np.ogrid[-m: m + 1, -n: n + 1]

    gauss = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    gauss[gauss < np.finfo(gauss.dtype).eps * gauss.max()] = 0
    return gauss


def draw_gaussian(hm, ctr, radius):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((radius, radius), diameter / 6)
    x, y = ctr
    h, w = hm.shape

    l, r = min(x, radius), min(w - x, radius + 1)
    t, b = min(y, radius), min(h - y, radius + 1)
    masked = hm[y - t: y + b, x - l: x + r]
    masked_gaussian = gaussian[radius - t: radius + b, radius - l: radius + r]

    if min(masked.shape) > 0 and min(masked_gaussian.shape) > 0:
        m = np.maximum(masked, masked_gaussian)
        hm[y - t: y + b, x - l: x + r] = m


def gather_features(output, index, mask):
    b, c = output.shape[:2]
    output = output.view(b, c, -1).permute(0, 2, 1).contiguous()
    index = index.unsqueeze(2).expand(*index.shape, c)
    pred = output.gather(dim=1, index=index)
    return pred


def reg_l1_loss(output, mask, index, target):
    pred = gather_features(output, index, mask)
    # norm = torch.sum(target, dim=1, keepdim=True) * 0.5 + 1e-4
    # norm = torch.reciprocal(norm)
    mask = mask.unsqueeze(2).expand_as(pred).float()
    loss = F.l1_loss(pred * mask, target * mask,
                     reduction="sum")
    loss = loss / (mask.sum() + 1e-4)
    return loss


def modified_focal_loss(pred, gt):
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    neg_weight = torch.pow(1 - gt, 4)
    pred = torch.sigmoid(pred)
    pred = torch.clamp(pred, 1e-12)

    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weight * neg_inds

    num_pos = pos_inds.sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos < 1:
        loss = -neg_loss
    else:
        loss = -(pos_loss + neg_loss) / num_pos
    return loss, num_pos


class FocalLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.neg_loss = _neg_loss

    def forward(self, out, target):
        return self.neg_loss(out, target)


class RegLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, pred, target, mask):
        return _reg_loss(pred, target, mask)


class RegL1Loss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, pred, target, mask):
        loss = F.l1_loss(pred * mask, target * mask,
                         reduction="sum")
        return loss


class NormRegL1Loss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, pred, target, mask):
        pred = pred / (target + 1e-4)
        target = target * 0 + 1
        loss = F.l1_loss(pred * mask, target * mask,
                         reduction="sum")
        return loss


@CRITERION_REGISTRY.register()
class CenterNetDetCriterion:
    _criterion_dict = {
        "focal_loss": modified_focal_loss,
        "mse": nn.MSELoss,
        "smooth_l1": RegLoss,
        "l1": reg_l1_loss,
        "norm_l1": NormRegL1Loss
    }

    def __init__(self, cfg):
        self.device = idist.device()

        self.heatmap_loss_type = cfg.HEATMAP_LOSS_TYPE
        self.heatmap_loss = CenterNetDetCriterion._criterion_dict[self.heatmap_loss_type]
        self.heatmap_loss_weight = cfg.HEATMAP_LOSS_WEIGHT

        self.size_loss_type = cfg.SIZE_LOSS_TYPE
        self.size_loss = CenterNetDetCriterion._criterion_dict[self.size_loss_type]
        self.size_loss_weight = cfg.SIZE_LOSS_WEIGHT

        self.reg_loss_type = cfg.REG_LOSS_TYPE
        self.reg_loss = CenterNetDetCriterion._criterion_dict[self.reg_loss_type]
        self.reg_loss_weight = cfg.REG_LOSS_WEIGHT

        self.num_classes = cfg.NUM_CLASSES
        self.output_size = cfg.OUTPUT_SIZE
        self.down_ratio = cfg.DOWN_RATIO
        self.min_overlap = cfg.MIN_OVERLAP
        self.num_outputs = cfg.NUM_OUTPUTS

    def __call__(self, out, y):
        hm, wh, reg = out
        gt_hm, gt_wh, gt_reg, mask, index = self.label_objs(y)

        loss_hm, num = self.heatmap_loss(hm, gt_hm)
        loss_wh = self.size_loss(wh, mask, index, gt_wh)
        loss_reg = self.reg_loss(reg, mask, index, gt_reg)
        
        losses = loss_hm * self.heatmap_loss_weight \
               + loss_wh * self.size_loss_weight \
               + loss_reg * self.reg_loss_weight

        return losses, {
            "pos_num": num.detach().item(),
            "hm": loss_hm.detach().item(),
            "wh": loss_wh.detach().item(),
            "reg": loss_reg.detach().item(),
            "total": losses.detach().item()
        }
    
    @torch.no_grad()
    def label_objs(self, tgts):
        cats, bboxes = tgts
        box_scale = 1.0 / self.down_ratio
        bboxes = [boxes * box_scale for boxes in bboxes]

        h, w = self.output_size
        b = len(cats)

        gt_hm = np.zeros((b, self.num_classes, h, w), dtype=np.float32)
        gt_wh = np.zeros((b, self.num_outputs, 2), dtype=np.float32)
        gt_reg = np.zeros((b, self.num_outputs, 2), dtype=np.float32)
        mask = np.zeros((b, self.num_outputs), dtype=np.float32)
        index = np.zeros((b, self.num_outputs), dtype=np.int64)

        for i, cls, bbox in zip(range(b), cats, bboxes):
            num_boxes = min(bbox.shape[0], self.num_outputs)
            ctrs = torch.stack([bbox[:, 0] + bbox[:, 2],
                                bbox[:, 1] + bbox[:, 3]], dim=1)
            ctrs *= 0.5
            ctrs = ctrs.numpy()
            ctrs = ctrs[:num_boxes]
            ctrs_int = ctrs.astype(np.int32)
            gt_reg[i, :num_boxes] = ctrs - ctrs_int

            index[i, :num_boxes] = ctrs_int[:, 1] * w + ctrs_int[:, 0]
            mask[i, :num_boxes] = 1

            wh = torch.stack([bbox[:, 2] - bbox[:, 0],
                              bbox[:, 3] - bbox[:, 1]], dim=1)
            wh = wh[:num_boxes]
            wh = wh.numpy()
            gt_wh[i, :num_boxes] = wh

            radius = gaussian_radius(wh, self.min_overlap)
            radius = np.clip(radius, 0, None)
            radius = radius.astype(np.int32)
            
            for j, c in enumerate(cls.tolist()):
                draw_gaussian(gt_hm[i, c], ctrs_int[j], radius[j])
    
        gt_hm = torch.tensor(gt_hm, dtype=torch.float32, device=self.device)
        gt_wh = torch.tensor(gt_wh, dtype=torch.float32, device=self.device)
        gt_reg = torch.tensor(gt_reg, dtype=torch.float32, device=self.device)
        gt_mask = torch.tensor(mask, dtype=torch.float32, device=self.device)
        index = torch.tensor(index, dtype=torch.long, device=self.device)
        return gt_hm, gt_wh, gt_reg, gt_mask, index


@VISUALIZER_REGISTRY.register()
class CenterNetDetVisualizer:
    def __init__(self, cfg):
        self.classes_name = cfg.CLASSES_NAME
        self.input_size = cfg.INPUT_SIZE
        self.classes_color = cfg.CLASSES_COLOR
        self.num_images = cfg.NUM_IMAGES

    def __call__(self, batch, outs):
        images_file = batch[3]
        outs = outs[0].detach().cpu()

        out_dict = {}
        for _, image_file, pred in zip(range(self.num_images), images_file, outs):
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
            out_dict[image_file] = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        return out_dict