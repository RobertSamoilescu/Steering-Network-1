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
```shell
./run_train.sh
```
  
## Test models - Open loop evaluation
```shell
./run_test.sh
```
