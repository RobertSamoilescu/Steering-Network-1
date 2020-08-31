#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 


import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import torch
import numpy as np
from tqdm import tqdm
import pickle as pkl
import pandas as pd
import argparse
from util.dataset import *

# parse argument
parser = argparse.ArgumentParser()
parser.add_argument("--augm", action='store_true', help="include perspective augmentation into the dataset")
args = parser.parse_args()


# define dataset
train_dataset = UPBDataset("../dataset", train=True, augm=args.augm)

# function to compute weights
def compute_weights(nclasses=6):
	global train_dataset
	
	counts = [0] * nclasses
	rel_course = [0] * len(train_dataset)
	
	for i in tqdm(range(len(train_dataset))):
		rel_course[i] = int(np.clip(np.abs(train_dataset[i]['rel_course_val'].item()), 0, nclasses - 1))
		counts[rel_course[i]] += 1
		
	weights_per_class = [0.] * nclasses
	N = np.sum(counts)
	for i in range(nclasses):
		weights_per_class[i] = N / float(nclasses * counts[i])
	
	weights = [0] * len(train_dataset)
	for i in tqdm(range(len(rel_course))):
		weights[i] = weights_per_class[rel_course[i]]
	return weights


# compute weights
nclasses = 6
weights = compute_weights(nclasses=nclasses)

# data sampler
tf_weights = torch.DoubleTensor(weights)                                       
sampler = torch.utils.data.sampler.WeightedRandomSampler(tf_weights, len(tf_weights))
train_loader = DataLoader(
	train_dataset, 
	batch_size=50, 
	sampler = sampler, 
	num_workers=4, 
	pin_memory=True
)

# display a batch
for i, data in enumerate(train_loader):
	rel_course = data['rel_course_val'].numpy().reshape(-1)
	rel_course = np.clip(np.abs(rel_course), 0, nclasses-1)
	plt.figure()
	sns.distplot(rel_course, bins=6)
	plt.show()
	break

# save to csv
dest_file = "../dataset/weights"
if args.augm:
	dest_file += "_augm"
dest_file += ".csv"

df = pd.DataFrame(data=weights, columns=["name"])
df.to_csv(dest_file, index=False)

# double check by reading the file
df_weights = pd.read_csv(dest_file)
df_weights.head()







