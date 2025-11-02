import cv2
import random
import numpy as np
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def add_gamma3(img):

    clean = (img*255.0+1)/256.0
    clean = clean * clean
    # L_list = [1.0 ,2.0,4.0,6.0,8.0,10.0]
    # L_s = random.sample(L_list,1)
    # L   = L_s[0]
    L = 1.0
    m = torch.distributions.gamma.Gamma(torch.tensor([L]), torch.tensor([L]))
    b = m.sample(sample_shape=img.size()).cuda()
    noise = b.view_as(img)
    # print(torch.max(c))
    intensity = noise * clean
    # noise_img =torch.sqrt(c * clean)
    # clean2 = torch.sqrt(clean)
    noise_img = torch.sqrt(intensity)


    return noise_img
#
# imgs = cv2.imread('D:\\desk\\LabelMe\\LabelMe\\Labelme\\data\\Labelme\\Labelme\\test\\img\\009000.jpg')
# img2222 = add_gamma3(imgs)
#
# # 产生高斯随机数
# # noise = np.random.normal(0,50,size=img.size).reshape(img.shape[0],img.shape[1],img.shape[2])
# # # 加上噪声
# # img = img + noise
# # img = np.clip(img,0,255)
# # img = img/255
# #
# # img1 = cv2.GaussianBlur(img,(9,9),0)
#
#
#
# cv2.imshow('Gauss noise',img2222)
# cv2.waitKey(0)
# cv2.imshow('Gaussblur noise',img1)
# cv2.waitKey(0)


# 给图像加入噪声


# 读取图像
# def read_image(data_path):
#     with open(data_path, "rb") as f:
#         data1 = np.fromfile(f, dtype=np.uint8)
#         # 塑形成[batch, c, h, w]
#         images = np.reshape(data1, [-1, 3, 96, 96])
#         # 图像转化为RGB(即最后一个维度是通道维度)的形式，方便使用matplotlib进行可视化
#         images = np.transpose(images, [0, 3, 2, 1])
#     return images / 255
#
# data_path = "../数据集/STL10/stl10_binary/train_X.bin"
# images = read_image(data_path)
# images.shape
