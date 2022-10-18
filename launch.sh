#!/bin/bash 
docker run -it --rm --gpus=all --network=host --ipc=host -v $PWD:/workspace --name dt $1 /bin/bash