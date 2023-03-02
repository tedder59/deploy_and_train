# 模型训练和部署

## 介绍
“工欲善其事，必先利其器”，我总结了一些模型训练和部署相关的代码。一方面加深自己对于模型的理解，另一方面也为日常工作提供一些便利。

## 运行环境
推荐使用docker镜像[tedder/deploy:23.03](https://hub.docker.com/repository/docker/tedder/deploy/general)。
需要准备数据集(Kitti, COCO, CULane, nuscenes)，存储结构类似下图：

![](https://github.com/tedder59/deploy_and_train/blob/main/docs/images/20230302-184413.png)

容器启动命令:
- dt    (容器名称 $1)
- tedder/deploy:23.03   (镜像名称 $2)
- path_to_datasets  (datasets目录   $3)

```shell
docker/launch.sh dt tedder/deploy:23.03 path_to_datasets
```

部署代码编译：
```shell
cd trt/src
cmake -B build
cmake --build build
```

## Models
### RetinaNet
个人最爱的单阶段检测模型，使用kitti数据集，检测汽车，行人，自行车三个类别。
- kitti数据集转coco格式
    ```shell
    python datasets/COCO/kitti_to_coco.py --kitti /workspace/datasets/kitti --output data/RetinaNet
    ```
- 训练
    - 单机单卡
        ```shell
        python3 train.py --config configs/RetinaNet/kitti_mobilenetv3_bifpn.yaml
        ```
- 部署（相关c++代码位于trt目录下，使用cmake进行编译）
    - 导出onnx文件
        ```shell
        python exports/ReitnaNet/export.py --config configs/RetinaNet/kitti_mobilenetv3_bifpn.yaml\
         --ckpt 'RetinaNet_checkpoint_AP=0.4705.pt' -o retinanet.onnx
        ```
    - build trt engine
    - c++ demo

### CenterNet
当下最流行检测模型，使用kitti数据集，检测汽车，行人，自行车三个类别。
- kitti数据集转coco格式

### CLRNet
基于检测的车道线模型，使用CULane数据集。

### PointPillar
激光雷达点云检测入门模型，使用kitti数据集。

### Deformable-DETR
transformer架构检测模型，使用coco数据集。

### Panoptic SegFormer
优雅的全景分割模型，使用coco数据集。



## 量化（TensorRT）

## 稀疏化（Sparsity）

## Benchmark测试模型推理时延