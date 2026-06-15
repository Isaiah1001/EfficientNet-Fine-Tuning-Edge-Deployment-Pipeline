# ./prepare_data.py
from preprocess import data_access, get_dataset, data_manipulate, subsetTrans
from scipy.io import loadmat
import random
import matplotlib.pyplot as plt
from torch.utils.data import  DataLoader
import torch
# plot setting
plt.rcParams['font.family'] = 'Times New Roman'

# =========================
# 0) Data and Split ID loading
# =========================
# load dataset
dataset = data_access("99_flower_data")
# load split ids
# the split method is used by paper below
# Nilsback, M-E. and Zisserman, A. Automated flower classification over a large number of classes.
# Proceedings of the Indian Conference on Computer Vision, Graphics and Image Processing (2008) 
# http://www.robots.ox.ac.uk/~vgg/publications/papers/nilsback08.{pdf,ps.gz}.
split_data = loadmat(f"./setid.mat") 
train_dataset, val_dataset, test_dataset = get_dataset(dataset,split_data)

# =========================
# 1) Apply transform
# =========================
# normalize dataset
mean=[0.485, 0.456, 0.406] 
std=[0.229, 0.224, 0.225] # using the mean and std from ImageNet dataset

# get transforms
basic, aug = data_manipulate(mean, std)
train_dataset_trans = subsetTrans(train_dataset, aug)
val_dataset_trans = subsetTrans(val_dataset, basic)
test_dataset_trans = subsetTrans(test_dataset, basic)
dataset = train_dataset_trans
# plot 
def plot_random_samples(dataset, n_rows=3, n_cols=4):
    """
    Randomly show samples from dataset.
    """
    n_samples = n_rows * n_cols
    total_images = len(dataset)
    random_indices = random.sample(range(total_images), n_samples)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
    axes = axes.flatten()
    for ax, idx in zip(axes, random_indices):
        img, label, name = dataset[idx]
        _, h, w = img.shape
        ax.imshow(img.permute(1, 2, 0).cpu())
        ax.set_title(name,fontsize=12,)
        info = (f"Index : {idx}\n"
                f"Label : {label}\n"
                f"Size  : {w}x{h}")
        ax.text(
            0.98,
            0.95,
            info,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            horizontalalignment='right',
            multialignment='left',
            bbox=dict(facecolor='white', edgecolor='black', alpha=0.8))
        ax.axis("off")
    
    plt.tight_layout()
    plt.show()
    
plot_random_samples(train_dataset_trans, n_rows=2, n_cols=3)

# get dataloader
bs = 16
nw = 8
pf = 2
g = torch.Generator()
g.manual_seed(42)

train_loader = DataLoader(train_dataset_trans, batch_size=bs, shuffle=True, num_workers=nw, pin_memory=True, persistent_workers=True, prefetch_factor=pf, generator=g)
val_loader = DataLoader(val_dataset_trans, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True, persistent_workers=True)
test_loader = DataLoader(test_dataset_trans, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True, persistent_workers=True)
    