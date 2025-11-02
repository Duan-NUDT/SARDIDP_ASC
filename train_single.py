import configparser
import os
import random
from datetime import datetime
#import cv
#import cv
import cv2
import torch
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from Unet_single import Unet_single
from unet_parts import inconv, down, up
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_root_dir = 'D:\\desk\\LabelMe\\LabelMe\\Labelme\\data\\Labelme\\Labelme\\LabelMe-12-50k\\test'
#train_root_dir = 'D:\\desk\\train2014_2'
train_img_dir = 'img'

test_root_dir = 'D:\\desk\\LabelMe\\LabelMe\\Labelme\\data\\Labelme\\Labelme\\test'
test_img_dir = 'img'


def plot(train_loss):
    plt.figure(figsize=(5,5))
    plt.plot(train_loss, label='train_loss')
    plt.title('train_loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.show()


tran_pose = torchvision.transforms.Compose([                                #对图片裁剪进行处理 此时输出为1x32x32
    # torchvision.transforms.Resize(size=(32,32)),
    torchvision.transforms.Grayscale(1),       # 将三通道图像转换为单通道灰度图
    torchvision.transforms.RandomCrop(size=(64,64)),
    torchvision.transforms.ToTensor(),
    # torchvision.transforms.Lambda(lambda x: x.repeat(3,1,1)),
    # torchvision.transforms.RandomHorizontalFlip(),
    # torchvision.transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])
])


# 对图像进行加噪
class add_gamma3:
    def add_gamma3(self, img):
        clean = (img*255.0+1)/256.0
        clean = clean * clean
        L_list = [1.0 ,2.0,4.0,6.0,8.0,10.0]
        L_s = random.sample(L_list,1)
        L   = L_s[0]
        # L = 1.0
        m = torch.distributions.gamma.Gamma(torch.tensor([L]), torch.tensor([L]))
        b = m.sample(sample_shape=img.size()).cuda()
        noise = b.view_as(img)
        # print(torch.max(c))
        intensity = noise * clean
        # noise_img =torch.sqrt(c * clean)
        # clean2 = torch.sqrt(clean)
        noise_img = torch.sqrt(intensity)
        return intensity, noise_img

add_gamma3 = add_gamma3()


# 对数据集路径进行处理
class MyDataset(Dataset):
    def __init__(self, root_dir, label_dir):
        super().__init__()
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir, self.label_dir)
        self.img_path = os.listdir(self.path)

    def __getitem__(self, idx):
        img_name = self.img_path[idx]
        img_item_path = os.path.join(self.root_dir, self.label_dir, img_name)  # 提取完整文件路径，并进行拼接

        #img = cv2.imread(img_item_path)
        img = Image.open(img_item_path)

        # img = random_crop(img)

        img = tran_pose(img)

        # label = self.label_dir
        # if label == 'img':
        #     label = torch.tensor(0)
        # else:
        #     label = torch.tensor(1)
        # return img, label
        return img

    def __len__(self):
        return len(self.img_path)

Unet = Unet_single()
Unet = Unet.to(device)


train_dataset = MyDataset(train_root_dir, train_img_dir)
test_dataset = MyDataset(test_root_dir, test_img_dir)

train_data_load = DataLoader(dataset=train_dataset, batch_size = 1,  shuffle=True, drop_last=True)   # shuffle 随机打乱数据集
test_data_load = DataLoader(dataset=test_dataset, batch_size = 1,  shuffle=True, drop_last=True)



# 假设模型和优化器已经定义

optim = torch.optim.SGD(Unet.parameters(), lr=1.0)  # 初始学习率设置为1e-4

# 定义学习率衰减函数
def lr_lambda(i):
    if i < 50:
        return 1e-4  # 前50个epoch学习率保持不变
    else:
        return max(1e-5, 1e-4 *(1+ (-0.9)*(i - 50) / 50) ) # 50个epoch之后线性衰减到1e-5

# 创建调度器
scheduler = LambdaLR(optim, lr_lambda)

# # 训练循环
# for epoch in range(100):  # 假设总训练100个epoch
#     # 训练模型...
#     optimizer.step()
#     scheduler.step()  # 更新学习率

train_dataset_size = len(train_dataset)
test_dataset_size  = len(test_dataset)

print(f'训练集长度为{train_dataset_size}')
print(f'测试集长度为{test_dataset_size}')

pixelwise_loss = nn.L1Loss(reduction='mean')  #平均绝对误差,L1-损失
loss_fn = nn.MSELoss()  #L2-损失 均方误差
loss_fn = loss_fn.to(device)
train_loss = list()

# loss_fn = nn.CrossEntropyLoss()
# loss_fn = loss_fn.to(device)

# criterionL1 = torch.nn.L1Loss()
# criterionL2 = torch.nn.MSELoss()


# learning_rate = 1e-3
# optim = torch.optim.Adam(Unet.parameters(), learning_rate)

train_step = 0   #训练次数
epoch = 100
#epochs = epoch * train_dataset_size
flag = 0
intensity = 0
lr_history = []
# 进入训练过程
if __name__ == '__main__':
    for i in range(epoch):
        print(f'-----第{i + 1}次训练开始-----')
        Unet.train()
        for data in train_data_load:
            imgs = data

            # imgs , targets = data
            imgs = imgs.to(device)
            # targets = targets.to(device)
            intensity, noise_img = add_gamma3.add_gamma3(imgs)
            #noise_img = add_gass.add_gaussian_noise(imgs)
            outputs = Unet(noise_img)
            # imgs = torch.argmax(imgs, dim=1)  对图像进行降维
            loss = loss_fn(outputs, imgs)


            optim.zero_grad()  # 梯度清零
            loss.backward()   #反向传播
            optim.step()   #更新梯度
                

            train_step += 1
            if train_step % 1000 == 0:
                train_loss.append(loss.item())
                print(f'训练次数{train_step}次, Loss={loss}')    # 每十次输出一次损失
        scheduler.step()      
        current_lr = optim.param_groups[0]['lr']
        lr_history.append(current_lr)
        print(f"End of Epoch {i + 1}: Current Learning Rate: {current_lr}")
        # Unet.eval()
        # accuracy = 0
        # accuracy_total = 0
        # with torch.no_grad():
        #     for data in test_data_load:
        #         imgs, targets = data
        #         imgs = imgs.to(device)
        #         targets = targets.to(device)
        #
        #         outputs = Unet(imgs)
        #         # accuracy = (outputs.argmax(axis=1) == imgs).sum()
        #         # accuracy_total += accuracy
        print(f'第{i + 1}轮训练结束')
        # print(f'第{i + 1}轮训练结束，准确率{accuracy_total/test_dataset_size}')
        flag = flag + 1
        if flag % (0.2 * epoch) == 0:
            # Img = Image.fromarray(cv.cvtColor(imgs, cv2.COLOR_BGR2RGB))
            torch.save(Unet, f'imgs_lr_gai_single_{i + 1}.pt')
        #torch.save(Unet, f'ants_bees_{i+1}_acc_{accuracy_total/test_dataset_size}')
        # # 保存
        # torch.save(the_model, PATH)
        # # 读取
        # the_model = torch.load(PATH)

    plot(train_loss)
    # print(datetime.datetime.now())












