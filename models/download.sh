#!/usr/bin/env bash

cd $(dirname $0)
curl -O http://cnnlocalization.csail.mit.edu/demoCAM/models/imagenet_googlenetCAM_train_iter_120000.caffemodel
