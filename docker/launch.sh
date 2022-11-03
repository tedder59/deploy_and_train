#!/bin/bash 
docker run -it --rm --gpus=all --network=host --ipc=host -v $PWD:/workspace --name dt tedder/deploy:22-11-03
