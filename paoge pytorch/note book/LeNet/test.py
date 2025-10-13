import torch
import torch.utils.data as Data
from torchvision.datasets import FashionMNIST
from torchvision import transforms
from model import LeNet
import torch.nn.functional as F


def test_data_process():
    test_data = FashionMNIST(root="./paoge pytorch/data/MNIST", 
                             train=False, 
                             transform=transforms.Compose([transforms.Resize(size = 28),
                                                          transforms.ToTensor()]), # 变为tensor后会增加一个维度1
                             download=True)
    
    test_dataloader = Data.DataLoader(dataset = test_data,
                                      batch_size = 1, 
                                      shuffle = True,
                                      num_workers = 6) # 多线程读取数据
    
    return test_dataloader

def test_model_process(model, test_dataloader):
    device = torch.device("mps" if torch.backends.mps.is_built() else "cpu")
    print(f"using {device} device:")

    # 将模型放到对应设备上
    model = model.to(device)
    
    # 初始化参数
    test_corrects = 0.0
    test_num = 0

    with torch.no_grad():  # 只进行前向传播，不进行反向传播
        for test_data_x, test_data_y in test_dataloader: 
            # 将数据放在设备上
            test_data_x = test_data_x.to(device)
            test_data_y = test_data_y.to(device)
            # 设置模型为评估模式
            model.eval()
            # 前向传播
            output = model(test_data_x)

            probs = F.softmax(output, dim = 1) # 按第一维(行)进行softmax
            pre_lab = torch.argmax(probs, dim = 1) # 按行取最大值的索引

            # 如果预测正确，则正确数加1
            test_corrects += torch.sum(pre_lab == test_data_y.data)
            test_num += test_data_x.size(0) # 累计样本数

    test_acc = test_corrects.item() / test_num
    print(f"test_acc: {test_acc:.4f}")

def test_model_process_det(model, test_dataloader):
    device = torch.device("mps" if torch.backends.mps.is_built() else "cpu")
    print(f"using {device} device:")

    # 将模型放到对应设备上
    model = model.to(device)

    classes = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    with torch.no_grad():
        for b_x, b_y in test_dataloader:
            b_x = b_x.to(device)
            b_y = b_y.to(device)
            model.eval()

            output = model(b_x)
            pre_lab = torch.argmax(output, dim = 1)
            result = pre_lab.item()
            label = b_y.item()
            print("the prediction is {:s}, the true value is {:s}".format(classes[result], classes[label]))


if __name__ == "__main__":
    # 加载模型
    model = LeNet()
    # 加载参数
    model.load_state_dict(torch.load("./paoge pytorch/note book/LeNet/results/LeNet.pth"))
    # 加载数据
    test_dataloader = test_data_process()
    # 测试数据
    test_model_process(model, test_dataloader)
    # test_model_process_det(model, test_dataloader)


    





            