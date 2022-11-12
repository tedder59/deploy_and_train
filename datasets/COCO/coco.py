# Copyright (c) 2022 by zhuxiaolong, All rights Reserved.
from .transform import *
from ..build import build_transform
from ..build import DATASET_REGISTRY, COLLATE_REGIESTRY, EVALUATOR_REGISTRY
from torch.utils.data import Dataset
from torchvision import transforms
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
import ujson as json
import numpy as np
import itertools
import torch
import copy
import os


class DatasetFromList(Dataset):
    def __init__(self, lst, **kwargs):
        self._lst = lst
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __len__(self):
        return len(self._lst)

    def __getitem__(self, index):
        datas = copy.deepcopy(self._lst[index])
        datas = self.trans(datas)
        return datas


@DATASET_REGISTRY.register()
def build_coco_dataset(cfg):
    coco_api = COCO(cfg.ANNOTATION)

    cats_id = coco_api.getCatIds()
    cats = coco_api.loadCats(cats_id)
    cats_name = [x['name'] for x in sorted(cats, key=lambda x: x['id'])]

    images_id = sorted(coco_api.imgs.keys())
    images = coco_api.loadImgs(images_id)
    annotations = [
        coco_api.imgToAnns[image_id] for image_id in images_id
    ]

    datas = []
    for image_info, ann in zip(images, annotations):
        r = image_info
        r['annotations'] = [
            {name: x[name] for name in ['category_id', 'iscrowd', 'bbox']}
            for x in ann
        ]
        datas.append(r)
    
    return DatasetFromList(
        datas, trans=build_transform(cfg.TRANSFORMS),
        category_names=cats_name,
    )


@COLLATE_REGIESTRY.register()
class COCOTrainCollator:
    def __init__(self, cfg) -> None:
        self.trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(cfg.PIXEL_MEAN, cfg.PIXEL_STD)
        ])

    def __call__(self, batch):
        images = torch.stack([self.trans(x['image']) for x in batch])

        cats = [
            torch.tensor([
                ann['category_id'] for ann in data['annotations']
            ], dtype=torch.int32)
            for data in batch
        ]

        bbs = [
            torch.tensor([
                ann['bbox'] for ann in data['annotations']
            ], dtype=torch.float32)
            for data in batch
        ]

        images_file = [data['file_name'] for data in batch]

        return images, cats, bbs, images_file


@COLLATE_REGIESTRY.register()
class COCOValCollator:
    def __init__(self, cfg):
        self.trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(cfg.PIXEL_MEAN, cfg.PIXEL_STD)
        ])

    def __call__(self, batch):
        images = torch.stack([self.trans(x['image']) for x in batch])
        ids = torch.tensor([data['id'] for data in batch],
                           dtype=torch.int32)
        return images, ids


@COLLATE_REGIESTRY.register()
class COCOTestCollator:
    def __init__(self, cfg):
        self.trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(cfg.PIXEL_MEAN, cfg.PIXEL_STD)
        ])

    def __call__(self, batch):
        images = torch.stack([self.trans(x['image']) for x in batch])
        return images


@EVALUATOR_REGISTRY.register()
class COCOEvaluator:
    def __init__(self, cfg):
        self.output_dir = cfg.VAL_OUTPUT
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        max_dets_per_image = cfg.get("MAX_DETS", None)
        if max_dets_per_image is None:
            self.max_dets_per_image = [1, 10, 100]
        else:
            self.max_dets_per_image = [1, 10, max_dets_per_image]
        
        anno = cfg.ANNOTATION
        self.coco_api = COCO(anno)
        cats_id = sorted(self.coco_api.getCatIds())
        self.classes_name = [cat['name'] for cat in self.coco_api.loadCats(cats_id)]
        self.input_size = cfg.INPUT_SIZE

        self.predictions = []
        
    def reset(self):
        self.predictions = []

    def process(self, y, y_pred):
        y_pred = y_pred[0]
        for image_id, preds in zip(y, y_pred):
            image_id = image_id.item()
            preds[:, 4] -= preds[:, 2]
            preds[:, 5] -= preds[:, 3]

            sy = float(self.coco_api.imgs[image_id]["height"]) / self.input_size[0]
            sx = float(self.coco_api.imgs[image_id]["width"]) / self.input_size[1]
            scale = torch.tensor([sx, sy, sx, sy], dtype=torch.float32)
            preds[:, 2:] *= scale

            objs = [
                {
                    "image_id": image_id,
                    "category_id": int(p[0].item()),
                    "bbox": p[2:].tolist(),
                    "score": p[1].item()
                }
                for p in preds if not torch.less(p[0], 0)
            ]

            if len(objs) > 1:
                self.predictions.append({
                    "image_id": image_id,
                    "instances": objs
                })

    def evaluate(self):
        coco_results = list(itertools.chain(*[x['instances'] for x in self.predictions]))
        output_file = os.path.join(self.output_dir, 'coco_instances_results.json')
        with open(output_file, 'w') as f:
            json.dump(coco_results, f)
            f.flush()
        
        if len(coco_results) <= 0:
            print('eval no result!!')
            return 0.0

        coco_dt = self.coco_api.loadRes(coco_results)
        coco_eval = COCOeval(self.coco_api, coco_dt, 'bbox')
        coco_eval.params.maxDets = self.max_dets_per_image

        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        metrics = ["AP", "AP50", "AP75", "APs", "APm", "APl"]
        results = {
            metric: float(coco_eval.stats[i] if coco_eval.stats[i] >= 0 else 'nan')
            for i, metric in enumerate(metrics)
        }

        precisions = coco_eval.eval['precision']
        results_per_category = []
        for idx, name in enumerate(self.classes_name):
            precision = precisions[:, :, idx, 0, -1]
            precision = precision[precision > -1]
            ap = np.mean(precision) if precision.size else float("nan")
            results_per_category.append(("{}".format(name), float(ap)))

        results.update({'AP-' + name: float(ap) for name, ap in results_per_category})
        for k, v in results.items():
            print('{:<40}: {:.2%}'.format(k, v))
        
        return results['AP']
