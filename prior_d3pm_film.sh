#!/bin/bash
export PYTHONPATH=/home/seunghyo/LIBERO:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=1

HYDRA_FULL_ERROR=1 python train.py --config-name=train_prior_d3pm_film.yaml \
    task=libero_90 \
    algo=quest_d3pm_film \
    exp_name=test \
    variant_name=block_32_ds_4 \
    algo.skill_block_size=32 \
    algo.downsample_factor=4 \
    checkpoint_path=/home/seunghyo/QueST_diffusion/experiments/libero/LIBERO_90/quest/quest/block_32_ds_4/0/run_000 \
    seed=0
