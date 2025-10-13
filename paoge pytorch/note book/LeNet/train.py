import torch
import time
import copy
import pandas as pd
from torchvision.datasets import FashionMNIST
from torchvision import transforms
import torch.utils.data as Data
import numpy as np
import matplotlib.pyplot as plt
from model import LeNet
import torch.nn as nn


def train_val_data_process():
    train_data = FashionMNIST(root="./paoge pytorch/data/MNIST", train=True, 
                          transform=transforms.Compose([transforms.Resize(size = 28),
                                                          transforms.ToTensor()]), # 变为tensor后会增加一个维度1
                          download=True)
    
    train_data, val_data = Data.random_split(train_data, 
                                             [round(0.8*len(train_data)), round(0.2*len(train_data))])
    
    train_dataloader = Data.DataLoader(dataset = train_data,
                                      batch_size = 64, 
                                      shuffle = True,
                                      num_workers = 6) # 多线程读取数据
    
    val_dataloader = Data.DataLoader(dataset = val_data,
                                      batch_size = 64, 
                                      shuffle = True,
                                      num_workers = 6) 
    
    return train_dataloader, val_dataloader

# train_val_data_process

def train_model_process(model, train_dataloader, val_dataloader, num_epochs):
    device = torch.device("mps" if torch.backends.mps.is_built() else "cpu")
    print(f"using {device} device:")

    # optimizer可以有很多比如SGD, Adam等
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001) # 学习率

    # 交叉熵损失函数，分类任务
    loss_fn = nn.CrossEntropyLoss()

    # 将模型放到对应设备上
    model = model.to(device)
    
    # 复制当前模型参数
    best_model_wts = copy.deepcopy(model.state_dict())

    # 初始化参数
    # 最高精确度
    best_acc = 0.0
    # 训练集损失列表
    train_loss_all = []
    # 验证集损失列表
    val_loss_all = []
    # 训练集精确度列表
    train_acc_all = []
    # 验证集精确度列表
    val_acc_all = []

    # 当前时间
    since = time.time()

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch+1, num_epochs))
        print("-"*10)

        # 初始化参数
        train_loss = 0.0
        train_acc = 0.0
        val_loss = 0.0
        val_acc = 0.0

        # 样本数量
        train_num = 0
        val_num = 0


        # step 是当前 batch 的编号（比如第 0 个、第 1 个……）。
        # b_x 是当前 batch 的图片数据（通常是一个张量，形状如 (batch_size, 通道数, 高, 宽)）。
        #  b_y 是当前 batch 的标签（也是一个张量，形如 (batch_size,)）。
        for step, (b_x, b_y) in enumerate(train_dataloader): # 对每一轮的每一个batch进行训练
            b_x = b_x.to(device) # 64*1*28*28
            b_y = b_y.to(device) # 64*1
            model.train() # 模型设置为训练模式

            # 前向传播过程，输入为一个batch， 输出为一个batch中对应的预测
            output = model(b_x) # 64*10
            # print(output.shape)

            # softmax 查找最大值的索引
            pre_lab = torch.argmax(output, dim = 1) # 64*1
            # print(pre_lab.shape)

            # 每一个batch的损失
            loss = loss_fn(output, b_y) # 这是一个平均损失
            # print(loss.shape)

            # 将梯度初始化为0
            optimizer.zero_grad()

            # backward
            loss.backward()

            # 参数更新
            optimizer.step()

            train_loss += loss.item() * b_x.size(0) # 累加一个batch的损失，item()返回一个python number

            train_acc += torch.sum(pre_lab == b_y.data) # 累加一个batch中预测正确的数量

            train_num += b_x.size(0) # 累加样本数量

        for step, (b_x, b_y) in enumerate(val_dataloader): # 对每一轮的每一个batch进行验证
            b_x = b_x.to(device) # 64*1*28*28
            b_y = b_y.to(device) # 64*1
            model.eval() # 模型设置为验证模式

            # 前向传播过程，输入为一个batch， 输出为一个batch中对应的预测
            output = model(b_x) # 64*10    

            # softmax 查找最大值的索引
            pre_lab = torch.argmax(output, dim = 1) # 64*1

            # 每一个batch的损失
            loss = loss_fn(output, b_y)

            val_loss += loss.item() * b_x.size(0) # 累加一个batch的损失,item()返回一个python number

            val_acc += torch.sum(pre_lab == b_y.data) # 累加一个batch中预测正确的数量

            val_num += b_x.size(0) # 累加样本数量
    
        # 计算每一个epoch的平均损失和精确度,并保存
        train_loss_all.append(train_loss / train_num)
        val_loss_all.append((val_loss / val_num))

        train_acc_all.append((train_acc / train_num).item())
        val_acc_all.append((val_acc/ val_num).item())

        print("train_loss: {:.4f} val_loss: {:.4f}".format(train_loss_all[-1], val_loss_all[-1]))
        print("train_acc: {:.4f} val_acc: {:.4f}".format(train_acc_all[-1], val_acc_all[-1]))

        # 判断是否保存模型参数
        if val_acc_all[-1] > best_acc:
            best_acc = val_acc_all[-1]
            best_model_wts = copy.deepcopy(model.state_dict())

    # 找到最佳模型所用时间
    time_elapsed = time.time() - since
    # print(type(train_acc_all[1]))
    print("time uesd for training: {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))


    # 选择最优参数
    # load best model weights

    torch.save(best_model_wts, "./paoge pytorch/note book/LeNet/results/LeNet.pth")

    train_process = pd.DataFrame(data = {"epoch": range(num_epochs),
                                            "train_loss_all": train_loss_all,
                                            "val_loss_all": val_loss_all,
                                            "train_acc_all": train_acc_all,
                                            "val_acc_all": val_acc_all})
    print(train_process)

    return train_process

def matplot_acc_loss(train_process):
    plt.figure(figsize = (12, 4))
    plt.subplot(1, 2, 1) # 1行2列第1个图
    plt.plot(train_process["epoch"], train_process.train_loss_all, 'ro-', label = 'train loss')
    plt.plot(train_process["epoch"], train_process.val_loss_all, 'bs-', label = 'val loss')
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("train and val loss")
    plt.subplot(1, 2, 2) # 1行2列第2个图
    plt.plot(train_process["epoch"], train_process.train_acc_all, 'ro-', label = 'train acc')
    plt.plot(train_process["epoch"], train_process.val_acc_all, 'bs-', label = 'val acc')
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.title("train and val acc")
    plt.show()     

if __name__ == "__main__":
    # 实例化模型
    model = LeNet()
    # 处理数据  
    train_dataloader, val_dataloader = train_val_data_process()
    # 训练模型
    train_process = train_model_process(model, train_dataloader, val_dataloader, num_epochs = 20)
    # 画图
    matplot_acc_loss(train_process)



