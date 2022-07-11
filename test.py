# License: BSD
# Author: Sasank Chilamkurthy

from __future__ import print_function, division


import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"

cudnn.benchmark = True
plt.ion()   # interactive mode


#Load Data  資料轉換函數
#----------------------------------------------------------------#
# Data augmentation and normalization for training
# Just normalization for validation
# 訓練資料集採用資料增強與標準化轉換
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224), # 隨機剪裁並縮放
        transforms.RandomHorizontalFlip(), # 隨機水平翻轉
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # 標準化
    ]),
    # 驗證資料集僅採用資料標準化轉換
    'val': transforms.Compose([
        transforms.Resize(256), # 縮放
        transforms.CenterCrop(224), # 中央剪裁
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # 標準化
    ]),
}

# 資料路徑
data_dir = 'data/hymenoptera_data'
# 建立 Dataset
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
# 建立 DataLoader
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'val']}
# 取得訓練資料集與驗證資料集的資料量
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
# 取得各類別的名稱
class_names = image_datasets['train'].classes
print(class_names)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

#Visualize a few images
#----------------------------------------------------------------#
# 將 Tensor 資料轉為原來的圖片
def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    # plt.pause(0.001)  # pause a bit so that plots are updated
    plt.pause(30)  # pause a bit so that plots are updated 停留時間(Seconds)

if __name__ == '__main__':
# Get a batch of training data
# 取得一個 batch 的訓練資料
    inputs, classes = next(iter(dataloaders['train']))

# Make a grid from batch
# 將多張圖片拼成一張 grid 圖
    out = torchvision.utils.make_grid(inputs)

# 顯示圖片
    imshow(out, title=[class_names[x] for x in classes])
    # img = imshow(out)
    # plt.imshow(img)
    # plt.title([class_names[x] for x in classes])