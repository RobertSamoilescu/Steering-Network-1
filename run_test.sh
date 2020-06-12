#!/bin/bash

BATCH_SIZE=32
VIS_INT=500
LOG_INT=50
DATASET_DIR=./dataset
MODEL=resnet


# RGB + SPEED
echo $MODEL" + SPEED"
python3 test.py \
	--model $MODEL\
	--batch_size $BATCH_SIZE \
	--use_speed \
	--dataset_dir $DATASET_DIR \

# RGB + SPEED + AUGM
echo $MODEL" + SPEED + AUGM"
python3 test.py \
 	--model $MODEL\
 	--batch_size $BATCH_SIZE \
 	--use_speed \
 	--use_augm \
  	--dataset_dir $DATASET_DIR \


