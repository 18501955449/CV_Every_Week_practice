'''
1.初始化预训练模型
2.更改输出层，使其输出数量与新数据集中的类数相同
3.为优化算法定义我们在训练期间更新哪些参数
4.运行训练步骤
#hymenoptera_data 数据集包含蜜蜂和蚂蚁两类
'''
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
import time
import copy
import os
from torchvision import datasets,models,transforms

def train_model(model,dataloaders,criterion,optimizer,num_epoches=25):
    since = time.time()
    val_acc_history = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    for epoch in range(num_epoches):
        print('Epoch {}/{}'.format(epoch,num_epoches-1))
        #每个epoch都进行训练和验证
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            running_loss = 0.0  # 记录训练时的loss下降过程
            running_corrects = 0
            # 遍历数据
            for inputs, labels in dataloaders[phase]:
                # 梯度初始化
                optimizer.zero_grad()
                # 前向传播
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                # 得到预测结果
                _, preds = torch.max(outputs, 1)
                # 仅在训练时更新梯度，反向传播，backward + optimize
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # 将验证集上结果最好的一次训练存储下来
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history
def set_parameter_requires_grad(model,feature_extracting):
    '''将所有的层参数设置为param.requires_grad = False'''
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
# 模型微调
# 使用预训练好的分类模型，提取特征时，我们只想更新最后一层（进行分类层）的参数。因此，我们不需要计算的层的.requires_grad属性设置为False。
# 当我们初始化新层时，默认情况下，新参数为.requires_grad = True，因此仅新层的参数将被更新。当我们进行微调时，我们可以将所有.required_grad的保留为默认值True。
# 以 Alexnet 为例：
# 在ImageNet数据集上进行预训练后的最后一层模型为
# (classifier): Sequential(
#     ...
#     (6): Linear(in_features=4096, out_features=1000, bias=True)
#  )
# 要想使用本次数据集，需要将最后一层的输出更改为
# model.classifier[6] = nn.Linear(4096,num_classes)
def initialize_model(num_classes,feature_extract,use_pretrained = True):
    #加载预训练模型
    model_ft = models.alexnet(pretrained = use_pretrained)
    #更改输出层
    set_parameter_requires_grad(model_ft,feature_extract)
    #获取指定线性层的单元数,新层param.requires_grad 默认为True
    num_ftrs = model_ft.classifier[6].in_features
    model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
    return model_ft
def main():
    # 用于特征提取的标志。 如果为False，我们会微调整个模型，
    # 当为True时，我们仅更新reshape layer
    feature_extract = True
    data_dir = "hymenoptera_data"
    num_classes = 2
    batch_size = 4
    num_epoches = 15
    input_size = 224
    model_ft = initialize_model(num_classes, feature_extract, use_pretrained=True)
    #数据增强
    data_transforms = {
        'train':transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ]),
        'val':transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }
    #加载训练集和验证集
    #ImageFolder(root, transform=None, target_transform=None, loader=default_loader)
    #root:在root路径下寻找图片，transform：对PIL Image进行的转换操作
    image_datasets = {x:datasets.ImageFolder(os.path.join(data_dir,x), data_transforms[x]) for x in ['train','val']}
    dataloaders_dict = {x:torch.utils.data.DataLoader(image_datasets[x],batch_size = batch_size,shuffle=True, num_workers=4) for x in ['train','val']}
    #检测GPU是否可用
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_ft = model_ft.to(device)
    # 设置损失函数
    criterion = nn.CrossEntropyLoss()
    # 设置优化函数
    #params_to_update = model_ft.parameters()
    #optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
    optimizer_ft = optim.SGD(model_ft.parameters(),lr=0.001, momentum=0.9)
    # 训练
    model_ft, hist = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epoches=num_epoches)

if __name__ == '__main__':
    main()