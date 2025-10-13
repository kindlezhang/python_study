from torchvision.datasets import FashionMNIST
from torchvision import transforms
import torch.utils.data as Data
import numpy as np
import matplotlib.pyplot as plt

# 下载数据 
train_data = FashionMNIST(root="./paoge pytorch/data/MNIST", train=True, 
                          transform=transforms.Compose([transforms.Resize(size = 224),
                                                          transforms.ToTensor()]), # 变为tensor后会增加一个维度1
                          download=True)

# seal data into batches
train_loader = Data.DataLoader(dataset = train_data, 
                               batch_size = 64, 
                               shuffle = True,
                               num_workers = 0)

# 获得第一个Batch的数据
for step, (b_x, b_y) in enumerate(train_loader):
    if step > 0:
        break 
    batch_x = b_x.squeeze().numpy() # 将四维张量移除第一维，并转化为numpy数组, 方便画图
    batch_y = b_y.numpy() # 将tensor转化为numpy数组
    class_label = train_data.classes # 获得标签名称
    # print(class_label)
print("the shape of batch_x:", batch_x.shape) # (64, 224, 224 ) # 64张图片，每张图片224*224

# 可视化批次的数据
plt.figure(figsize=(12, 5))
for ii in np.arange(len(batch_y)):
    plt.subplot(4, 16, ii+1)
    plt.imshow(batch_x[ii,:,:], cmap=plt.cm.gray)
    plt.title(class_label[batch_y[ii]], size = 10)
    plt.axis('off')
    plt.subplots_adjust(wspace=0.05, hspace=0.5)
plt.show()


    