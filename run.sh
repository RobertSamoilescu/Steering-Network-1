#!/bin/bash

BATCH_SIZE=32
VIS_INT=500
LOG_INT=50
DATASET_DIR=./dataset
MODEL=resnet

export CUDA_VISIBLE_DEVICES=0

echo $MODEL" + SPEED"
python3 train.py \
	--model $MODEL\
	--batch_size $BATCH_SIZE \
	--vis_int $VIS_INT \
	--log_int $LOG_INT \
	--use_speed \
	--dataset_dir $DATASET_DIR \
	> "print_logs/"$MODEL"_SPEED.out" 2>&1

echo $MODEL" + SPEED + AUG"
python3 train.py \
	--model $MODEL\
	--batch_size $BATCH_SIZE \
	--vis_int $VIS_INT \
	--log_int $LOG_INT \
	--use_speed \
	--use_aug \
	--dataset_dir $DATASET_DIR \
	> "print_logs/"$MODEL"_SPEED_AUG.out" 2>&1
