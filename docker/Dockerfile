FROM nvcr.io/nvidia/pytorch:22.08-py3
MAINTAINER Xiaolong Zhu <zhuxiaolong_auto@163.com>
ADD requirements.txt ./
RUN apt-get update && apt-get install -y libgl1 libbenchmark-dev &&\
    pip3 install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple &&\
    rm requirements.txt
ENV PYTHONPATH=.
