#!/bin/bash

docker run -it --rm --gpus=all --ipc=host \
--ulimit memlock=-1 --ulimit stack=67108864 \
--net=host \
-v $PWD/..:/workspace/projects \
-v $3:/workspace/datasets \
--name $1 $2
