#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import pandas as pd


# In[ ]:


with open("/mnt/storage/workspace/roberts/disertatie/scenes_split/train_scenes.txt", "rt") as fin:
    train_scenes = fin.read()

with open("/mnt/storage/workspace/roberts/disertatie/scenes_split/test_scenes.txt", "rt") as fin:
    test_scenes = fin.read()
    
train_scenes = set(train_scenes.split("\n"))
test_scenes = set(test_scenes.split("\n"))

train_files = []
test_files = []

files = os.listdir("../dataset/img_real")

for file in files:
    scene, _, _ = file.split(".")
    
    if scene in train_scenes:
        train_files.append(file[:-4])
    else:
        test_files.append(file[:-4])

# save as csv
train_csv = pd.DataFrame(train_files, columns=["name"])
test_csv = pd.DataFrame(test_files, columns=["name"])

train_csv.to_csv("../dataset/train_real.csv", index=False)
test_csv.to_csv("../dataset/test_real.csv", index=False)

train_csv = pd.read_csv("../dataset/train_real.csv")
test_csv = pd.read_csv("../dataset/test_real.csv")
