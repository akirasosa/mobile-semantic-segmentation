#!/usr/bin/env bash

set -eu

# Download data
aws s3 cp ${S3_BUCKET}/data/faces/images-128.npy /tmp/images.npy
aws s3 cp ${S3_BUCKET}/data/faces/masks-128.npy /tmp/masks.npy

# Download pre-trained models
mkdir -p ${HOME}/.keras/models/
aws s3 sync ${S3_BUCKET}/models/ ${HOME}/.keras/models/

rm -rf logs && mkdir logs
rm -rf artifacts && mkdir artifacts

#python train_top_model.py \
#  --img_file=/tmp/images.npy \
#  --mask_file=/tmp/masks.npy

#python train_fine_tune.py \
#  --img_file=/tmp/images.npy \
#  --mask_file=/tmp/masks.npy \

python train_full.py \
  --img_file=/tmp/images.npy \
  --mask_file=/tmp/masks.npy

# Save results
ts=$(date +%s)
aws s3 sync . ${S3_BUCKET}/experiments/${ts}/
