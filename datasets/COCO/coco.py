# Copyright (c) 2022 by zhuxiaolong, All rights Reserved.
from ..build import DATASET_REGISTRY, COLLATE_REGIESTRY, build_transform
from torch.utils.data import Dataset
from torchvision import transforms
from pycocotools.coco import COCO
import torch
import copy


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

        ids = torch.tensor([data['id'] for data in batch],
                           dtype=torch.int32)

        return images, cats, bbs


@COLLATE_REGIESTRY.register()
class COCOTestCollator:
    def __init__(self, cfg) -> None:
        self.trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(cfg.PIXEL_MEAN, cfg.PIXEL_STD)
        ])

    def __call__(self, batch):
        images = torch.stack([self.trans(x['image']) for x in batch])

        ids = torch.tensor([data['id'] for data in batch],
                           dtype=torch.int32)

        return images, ids
