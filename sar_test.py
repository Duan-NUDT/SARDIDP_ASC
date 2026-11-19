import os
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms
from matplotlib import image as mping
from scipy.io import loadmat
import numpy as np
import cv2
import random
import scipy.io as io
# from sar_train import downsample_tensor
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tran_pose1 = torchvision.transforms.Compose([                                #对图片裁剪进行处理
    torchvision.transforms.Resize(size=(1024, 1024)),
    # torchvision.transforms.RandomCrop(size=32,padding=4),    # 随机裁剪 32,32
    torchvision.transforms.ToTensor(),   # 转换为tensor
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),         # 将单通道改为三通道
    # torchvision.transforms.Lambda(lambda x: x.repeat(3,1 ,1)),
    # torchvision.transforms.RandomHorizontalFlip(),             # 水平随机翻转
    torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    # transforms.Normalize((0.1307,), (0.3081,))   # 如果不修改通道数，就更改标准化代码
])
# 从文件夹中将SAR图像进行读取
# SAR图像成像后的数据一般保存的格式是uint16 像素值取值范围是0-65535 日常所见到的图像一般是uint8格式的，像素值范围是0-255

Unet = torch.load('Sar_model/Sar_complex.pt', map_location=torch.device('cpu'))
Unet = Unet.cuda()


# 第一部分内容，读取npy文件  shape 128*128 单通道灰度图
depthmap = np.load(r'C:\Users\huang\Desktop\t.npy')
data = np.sqrt(np.square(depthmap[:,:,0]) + np.square(depthmap[:,:,1]))# 得到幅度图
data = np.uint8(data)

# data = np.clip(data, 1024*7, 1024*7+255)
# data -= (1024*7)
data = data/255
# 对数据进行随机降采样，随机降采样
height = data.shape[0]
weight = data.shape[1]
flag = 2
num_h = int(height/flag)
num_w = int(weight/flag)
new_data = np.zeros((num_h, num_w))
for i in range(num_h):
    for j in range(num_w):
        temp = random.randint(0,flag)
        y = i*flag - temp
        x = j*flag - temp
        new_data[j, i] = data[x, y]
downsampled = new_data



def test_plot(noise_img):
    test_img = noise_img
    test_img = test_img.squeeze(0)
    test_img = test_img.permute(1, 2, 0)
    array = test_img.cpu().numpy()
    plt.imshow(array, cmap='gray')
    # plt.show()
    plt.ylabel(f'tu')
    plt.title('yuantu')
def test_plot1(noise_img):
    test_img = noise_img
    test_img = test_img.squeeze(0)
    test_img = test_img.permute(1, 2, 0)
    array = test_img.cpu().numpy()
    plt.imshow(array, cmap='gray')
    plt.title('qvzao')
    plt.show()
    plt.ylabel(f'tu')

if __name__ == '__main__':
    Unet.eval()
    accuracy = 0
    accuracy_total = 0
    total_psnr = 0
    total_ssim = 0
    with torch.no_grad():
        # 输入图像shape要求是tensor （b, c, h, w）
            img = Image.open('data/test.JPG')
            # img = Image.open('BSD300/gt_uint8/0.png')
            transform = torchvision.transforms.Compose([
                torchvision.transforms.Grayscale(1),
                torchvision.transforms.ToTensor(),
            ])
            flag = 2
            img = transform(img)
            imgs = torch.unsqueeze(img,dim=0)
            imgs = imgs.to(device)
            outputs = Unet(imgs)
            plt.figure(2)
            test_plot(imgs)
            # test_plot(noise_img)
            plt.figure(3)
            test_plot1(outputs)



