#!/bin/bash
export PYTHONPATH=/home/seunghyo/LIBERO:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=1

HYDRA_FULL_ERROR=1 python train.py --config-name=train_autoencoder.yaml \
    task=libero_90 \
    algo=quest \
    exp_name=quest \
    variant_name=block_32_ds_4 \
    algo.skill_block_size=32 \
    algo.downsample_factor=4 \
    seed=0