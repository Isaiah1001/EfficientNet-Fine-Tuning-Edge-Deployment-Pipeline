# libraries
from preprocess import data_access

import os
from scipy.io import loadmat
import torch

# =========================
# 0) Data Loading
# =========================
# data location (for this project, the data is retrieved from 99_flower_data)
data_folder = '99_flower_data'
data_path = os.path.join(r'./', data_folder)

# check what's inside the data path
print(os.listdir(data_path))

# read in the text file containing the labels description
with open(os.path.join(data_path, "labels_description.txt"), "r", encoding="utf-8") as f:
    text_description = f.readlines()
print(text_description)

# read .mat file
data_labels_check = loadmat(os.path.join(data_path, 'imagelabels.mat'))
print('Keys in data dictionary:', data_labels_check.keys())
print('Data shape:', data_labels_check['labels'].shape)

# check availability of CPU or GPU
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"Using device: {device}")

# load data
dataset = data_access(data_folder)

# =========================
# 3) Data Checking
# =========================
# create dataset instance  
len(dataset)