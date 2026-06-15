# libraries
from preprocess import data_access

import os
from scipy.io import loadmat
import torch
import matplotlib.pyplot as plt
from collections import Counter
import random

# plot setting
plt.rcParams['font.family'] = 'Times New Roman'

# =========================
# 0) Data Loading
# =========================
# data location (for this project, the data is retrieved from 99_flower_data)
data_folder = '99_flower_data'
data_path = os.path.join(r'./', data_folder)

# check what's inside the data path
print("The data folder consists of: ")
for file in os.listdir(data_path):
    print(f"  - {file}")
    
# read in the text file containing the labels description
with open(os.path.join(data_path, "labels_description.txt"), "r", encoding="utf-8") as f:
    text_description = f.readlines()

# read .mat file
data_labels_check = loadmat(os.path.join(data_path, 'imagelabels.mat'))
# print('Keys in data dictionary:', data_labels_check.keys())
# print('Data shape:', data_labels_check['labels'].shape)

# check availability of CPU or GPU
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"Using device: {device}")

# load data
dataset = data_access(data_folder)

# =========================
# 1) Dataset Information
# =========================
# check number of image
print("\n")
total_images = len(dataset)
print("=" * 50)
print("Statistics of Dataset")
print("=" * 50)
print(f"Total Number of Images: {total_images}")
len(text_description)

# Distribution of each category
labels = data_labels_check['labels'][0]
label_counts = Counter(labels)
print(f"Number of Classes: {len(label_counts)}")
for cls_id, count in sorted(label_counts.items()):
    flower_name = text_description[cls_id-1].strip().strip("'").strip('"')
    print(f"Class {cls_id:3d}: {count:4d} images - {flower_name}")
    
# plot distribution
x = sorted(label_counts.keys())
y = [label_counts[i] for i in x]
num_classes = len(label_counts)
def plot_distri():
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x, y)
    ax.set_xlabel("Class Index", fontsize=14)
    ax.set_ylabel("Number of Images", fontsize=14)
    ax.set_title("Distribution of Images Across Classes", fontsize=16)
    ax.text(
        0.98, 0.95,
        f'Total Classes: {num_classes}\nTotal Images: {total_images}',
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(facecolor='white', edgecolor='black', alpha=0.8)
    )
    plt.tight_layout()
    plt.grid(linestyle='--', alpha=0.5)
    plt.show()
    
plot_distri()

# =========================
# 2) Image Information
# =========================
# plot figures
def plot_random_samples(dataset, n_rows=3, n_cols=4):
    """ Randomly generate flower figs and show information

    Args:
        dataset (tuple): loaded dataset
        n_rows (int, optional): number of rows to show. Defaults to 3.
        n_cols (int, optional): number of columns to show. Defaults to 4.
    """
    n_samples = n_rows * n_cols
    total_images = len(dataset)

    # random index
    random_indices = random.sample(range(total_images), n_samples)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4*n_cols, 4*n_rows)
    )

    axes = axes.flatten()
    for ax, idx in zip(axes, random_indices):
        # extract image, label, description and size
        img = dataset.retrieve_images(idx)
        label = dataset.retrieve_labels(idx)
        name = dataset.retrieve_description(label)
        size = img.size

        ax.imshow(img)

        ax.set_title(
            f"{name}\n",
            fontsize=12,
        )
        ax.text(0.98, 0.95,
        f"Index: {idx}\n"
        f"Label: {label}\n"
        f"Size: ({size[0]}, {size[1]})",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        horizontalalignment='right',
        multialignment='left',
        bbox=dict(facecolor='white', edgecolor='black', alpha=0.8))

        ax.axis("off")

    plt.tight_layout()
    plt.show()
    
plot_random_samples(dataset, n_rows=2, n_cols=3)