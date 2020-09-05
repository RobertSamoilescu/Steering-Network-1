#!/bin/bash

BATCH_SIZE=32
VIS_INT=500
LOG_INT=50
DATASET_DIR=./dataset
MODEL=resnet

export CUDA_VISIBLE_DEVICES=0

# train model on the raw dataset and append speed
echo $MODEL" + BALANCE + SPEED"
python train.py \
	--model $MODEL\
	--batch_size $BATCH_SIZE \
	--vis_int $VIS_INT \
	--log_int $LOG_INT \
	--use_speed \
	--use_balance \
 	--dataset_dir $DATASET_DIR \

# train model using 2D persepctive augmentation and append speed
echo $MODEL" + SPEED + BALANCE + AUG"
python3 train.py \
	--model $MODEL\
	--batch_size $BATCH_SIZE \
	--vis_int $VIS_INT \
	--log_int $LOG_INT \
	--use_speed \
	--use_aug \
	--use_balance \
	--dataset_dir $DATASET_DIR \