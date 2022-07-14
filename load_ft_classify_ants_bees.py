#!/usr/bin/env torch
# -*- coding:utf-8 -*-
# @Time  : 2022/7/14
# @Author: Lee
# @File  : classify_ants_bees.py
 
# 導入相關的包
from __future__ import print_function, division
 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"  # 關閉重複庫的警告

plt.ion()  #plt繪圖使用交互模式，而不是默認的阻塞模式，繪製第一個figure后繼續程序繪圖，可繪製多張figure
 
# 加載數據集，設置數據處理器，以及數據加載器
# 訓練集和驗證集分為两個數據加載器，分別用於訓練和驗證模型
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),  # 随机剪裁一个area然后再resize
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.2225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.2225])
    ]),
}
 
data_dir = 'data/hymenoptera_data'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                           data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                               shuffle=True, num_workers=4)
               for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_name = image_datasets['train'].classes
 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
 
 
# 可视化部分图像数据，以便了解数据扩充
def imshow(inp, title=None):
    """
    Imshow for tensor
    """
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.2225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated
 
 
# 训练模型，编写一个通用函数来训练模型
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
 
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
 
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
 
        # 每个epoch都有一个训练和验证阶段
        for phase in ['train', 'val']:
            if phase == 'train':
                # scheduler.step()
                model.train()  # Set model to train mode
            else:
                model.eval()  # Set model to evaluate mode
 
            running_loss = 0.0
            running_corrects = 0
 
            # 迭代数据
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
 
                # 零参数梯度
                optimizer.zero_grad()
 
                # 前向传播 track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
 
                    # 后向传播，尽在训练集阶段优化
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
 
                # 统计
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
 
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
 
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
 
            # 深度拷贝
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
 
        if phase == 'train':
            scheduler.step()
 
        print()
 
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:.4f}'.format(best_acc))
 
    # 加载最佳模型权重
    model.load_state_dict(best_model_wts)
    return model
 
 
# 可视化模型的预测结果
def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    image_so_far = 0
    fig = plt.figure()
 
    with torch.no_grad():
        for i ,(inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)
 
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
 
            for j in range(inputs.size()[0]):
                image_so_far += 1
                ax = plt.subplot(num_images // 2, 3, image_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_name[preds[j]]))
                imshow(inputs.cpu().data[j])
 
                if image_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)
 
 
# if __name__ == '__main__':
#     # 获取一批训练数据
#     inputs, classes = next(iter(dataloaders['train']))
#     # 批量制作网络
#     out = torchvision.utils.make_grid(inputs)
#     imshow(out, title=[class_name[x] for x in classes])
#     plt.pause(0)  # 避免图像一闪而过

if __name__=='__main__': 
    # model_ft = models.resnet18(pretrained=True)
    # num_ftrs = model_ft.fc.in_features
    # model_ft.fc = nn.Linear(num_ftrs, 2)
 
    # model_ft = model_ft.to(device)
    # criterion = nn.CrossEntropyLoss()
 
    # # 观察所有参数都正在优化
    # optimizier_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
    # # 每7个epochs衰减LR通过设置gamma=0.1
    # exp_lr_scheduler = lr_scheduler.StepLR(optimizier_ft, step_size=7, gamma=0.1)
    # model_ft = train_model(model_ft, criterion, optimizier_ft, exp_lr_scheduler,     
    #            num_epochs=2)
    # # 保存模型
    # torch.save(model_ft, 'models/model')  # 此处暂时以保存整个模型结构和参数为例

    model_ft = torch.load('models/model_ft.pt')  # 加载模型
    visualize_model(model_ft)
    plt.pause(0)