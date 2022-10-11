# Copyright (c) 2022 by zhuxiaolong, All rights Reserved.
import sklearn.model_selection as model_selection
import PIL.Image as Image
import collections
import argparse
import json
import os


EXCLUDE_TYPES = ['Tram', 'Misc', 'DontCare']
MIN_OBJ_AREA = 64
MIN_OBJ_EDGE = 8

CATS = [
    {"name": "Car", "id": 0},
    {"name": "Cyclist", "id": 1},
    {"name": "Pedestrian", "id": 2}
]
CATS_ID = {
    'Car': 0,
    'Van': 0,
    'Truck': 0,
    'Cyclist': 1,
    'Pedestrian': 2,
    'Person_sitting': 2
}


"""
KITTI label text format:
    type truncated occluded alpha x1 y1 x2 y2 height width length center_x center_y center_z rotation_y

truncated:
    0(non-truncated), 1(truncated)

occluded:
    0(fully visible), 1(partly occluded), 2(largely occluded), 3(unknown)
"""
class KittiObject(collections.namedtuple(
    "KittiObject", ["type", "truncated", "occluded", "alpha", "x1", "y1", "x2", "y2",
    "height", "width", "length", "x", "y", "z", "rotation_y"]
)):
    def __new__(cls, type, truncated, occluded, alpha, x1, y1, x2, y2, h, w, l, x, y, z,
                rotation):
        truncated, alpha, x1, y1, x2, y2, h, w, l, x, y, z, rotation = [
            float(x) 
            for x in [truncated, alpha, x1, y1, x2, y2, h, w, l, x, y, z, rotation]
        ]
        occluded = int(occluded)

        return super().__new__(cls, type, truncated, occluded, alpha, x1, y1, x2, y2,
                               h, w, l, x, y, z, rotation)


def save_json(images_map, image_label_list, out_file, start_obj_id=0):
    images, annotations = [], []
    obj_id = start_obj_id

    for image_file, label_file in image_label_list:
        image_file = os.path.basename(image_file)
        image_id = int(os.path.splitext(image_file)[0])
        objs = []

        with open(label_file) as f:
            for row in f.readlines():
                row = row.strip()
                obj = KittiObject(*row.split())
                if obj.type in EXCLUDE_TYPES:
                    continue

                w = obj.x2 - obj.x1
                h = obj.y2 - obj.y1
                if (w * h) < MIN_OBJ_AREA or min(w, h) < MIN_OBJ_EDGE:
                    continue

                objs.append({
                    "id": obj_id,
                    "image_id": image_id,
                    "category_id": CATS_ID[obj.type],
                    "bbox": [obj.x1, obj.y1, w, h],
                    "area": w * h,
                    "iscrowd": 1 if obj.occluded >= 2 else 0
                })
                obj_id += 1

        if len(objs) > 0:
            images.append(images_map[image_id])
            annotations.extend(objs)

    with open(out_file, 'wt') as f:
        json.dump({
            "images": images,
            "annotations": annotations,
            "categories": CATS
        }, f)
    return obj_id


def gather_files(image_root, label_root):
    image_label_list = []
    for image_file in os.listdir(image_root):
        label_file = f'{os.path.splitext(image_file)[0]}.txt'
        label_file = os.path.join(label_root, label_file)
        if not os.path.exists(label_file):
            continue

        image_file = os.path.join(image_root, image_file)
        image_label_list.append((image_file, label_file))

    return image_label_list


def gather_categories_infos(image_label_list):
    cats_stat = {}
    for _, label_file in image_label_list:
        with open(label_file) as f:
            for row in f.readlines():
                row = row.strip()
                cat = row[:row.find(' ')]
                if cat in EXCLUDE_TYPES:
                    continue

                if cat not in cats_stat:
                    cats_stat[cat] = 1
                else:
                    cats_stat[cat] += 1

    print('categories statistic:')
    for k, v in cats_stat.items():
        print('\t{:20}:{:>10}'.format(k, v))
    
    return cats_stat


def gather_images_infos(image_label_list):
    images_map = {}
    for image_file, _ in image_label_list:
        im = Image.open(image_file)
        w, h = im.size

        image_name = os.path.basename(image_file)
        image_id = int(os.path.splitext(image_name)[0])

        images_map[image_id] = {
            "id": image_id,
            "file_name": image_file,
            "height": h,
            "width": w
        }
    
    return images_map


if __name__ == "__main__":
    parser = argparse.ArgumentParser("kitti_to_coco")
    parser.add_argument("--kitti", type=str, required=True, 
                        help="path to kitt dataset")
    args = parser.parse_args()

    image_root = os.path.join(args.kitti, 'image_2')
    label_root = os.path.join(args.kitti, 'label_2')

    image_label_list = gather_files(image_root, label_root)
    images_map = gather_images_infos(image_label_list)

    train_split, test_split = model_selection.train_test_split(
        image_label_list, test_size=0.1, train_size=0.9
    )

    start_id = 0
    start_id = save_json(images_map, train_split, 'kitti_train.json', start_id)
    save_json(images_map, test_split, 'kitti_val.json', start_id)
