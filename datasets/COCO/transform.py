# Copyright (c) 2022 by zhuxiaolong, All rights Reserved.
from ..build import TRANSFORM_REGISTRY
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import imgaug.augmenters as iaa
import imgaug as ia
import imageio.v3 as iio


@TRANSFORM_REGISTRY.register()
class COCOLoadImage:
    def __init__(self, **kwargs):
        pass

    def __call__(self, record):
        im = iio.imread(record['file_name'])
        record['image'] = im
        return record


@TRANSFORM_REGISTRY.register()
class COCOResize:
    def __init__(self, **kwargs):
        self.image_size = kwargs['INPUT_SIZE']

    def __call__(self, record):
        image = record['image']
        scales = [
            float(self.image_size[0]) / record['height'],
            float(self.image_size[1]) / record['width']
        ]

        resized = ia.imresize_single_image(image, self.image_size)
        
        record['image'] = resized
        record['scales'] = scales
        return record


@TRANSFORM_REGISTRY.register()
class COCOBBoxResize:
    def __init__(self, **kwargs):
        self.image_size = kwargs['INPUT_SIZE']

    def __call__(self, record):
        image = record['image']

        bbs = BoundingBoxesOnImage([
            BoundingBox(x1=x['bbox'][0], y1=x['bbox'][1], 
                        x2=x['bbox'][0] + x['bbox'][2],
                        y2=x['bbox'][1] + x['bbox'][3])
            for x in record['annotations']
        ], shape=image.shape)
        
        resized = ia.imresize_single_image(image, self.image_size)
        bbs_rescaled = bbs.on(resized)

        record['image'] = resized
        for i in range(len(bbs)):
            bb = bbs_rescaled[i]
            record['annotations'][i]['bbox'] = [
                bb.x1, bb.y1, bb.x2, bb.y2
            ]

        return record

