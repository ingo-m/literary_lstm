#!/bin/bash

data_path="/home/john/Dropbox/Ice_and_Fire/"
analy_path="/home/john/PhD/GitLab/literary_lstm/"

docker run --gpus all -it --rm \
    -v ${analy_path}:${analy_path} \
    -v ${data_path}:${data_path} \
    -u $(id -u):$(id -g) \
    tensorflow/tensorflow:1.14.0-gpu-py3 python ${analy_path}main_w2v_ligemo.py
