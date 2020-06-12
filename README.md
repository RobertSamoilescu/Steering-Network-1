# Steering-Network-1


## Create dataset

```shell
mkdir raw_dataset
```

* Download the UBP dataset into the "raw_dataset" directory. A sample of the UPB dataset is available <a href="https://drive.google.com/drive/folders/1p_2-_Xo-Wd9MCnkYqPfGyKs2BnbeApqn?usp=sharing">here</a>.

```shell
mkdir scene_splits
```

* Download the scene splits into the "scene_splits" directory. The train-validation split is available <a href="https://github.com/RobertSamoilescu/UPB-Dataset-Split">here</a>.
In the "scene_splits" directory you should have: "train_scenes.txt" and "test_scenes.txt".

```shell
cd scripts

# create the dataset
python3 create_dataset.py --root_dir ../raw_dataset

# split the dataset into train-test
python3 split_dataset.py --train ../scene_splits/train_scenes.txt --test ../scene_splits/test_scenes.txt

# create synthetic dataset by performin 2D perspective augmentations only for the training dataset
python3 create_aug_dataset.py --root_dir ../raw_dataset --train ../scene_splits/train_scenes.txt
```

## Train models

``` shell
BATCH_SIZE=32
VIS_INT=500
LOG_INT=50
DATASET_DIR=./dataset
MODEL=resnet

export CUDA_VISIBLE_DEVICES=0

# train model on the raw dataset and append speed
echo $MODEL" + SPEED"
python3 train.py \
	--model $MODEL\
	--batch_size $BATCH_SIZE \
	--vis_int $VIS_INT \
	--log_int $LOG_INT \
	--use_speed \
	--dataset_dir $DATASET_DIR \

# train model using 2D persepctive augmentation and append speed
echo $MODEL" + SPEED + AUG"
python3 train.py \
	--model $MODEL\
	--batch_size $BATCH_SIZE \
	--vis_int $VIS_INT \
	--log_int $LOG_INT \
	--use_speed \
	--use_aug \
	--dataset_dir $DATASET_DIR \
  ```
