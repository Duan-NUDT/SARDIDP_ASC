import os
import random
import cv2
import torch
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from Unet_single import Unet_single
from tqdm import tqdm
import cv2
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
# import torchvision.metrics as metrics

def calc_psnr(im1, im2):

    return compare_psnr(im1, im2)

def calc_ssim(im1, im2):

    return compare_ssim(im1, im2)




tran_pose = torchvision.transforms.Compose([  # 对图片裁剪进行处理 此时输出为1x32x32

    torchvision.transforms.Grayscale(1),  # 将三通道图像转换为单通道灰度图
    torchvision.transforms.RandomCrop(size=(64, 64)),
    torchvision.transforms.ToTensor(),

])

# 对图像进行加噪
class add_gamma3:
    def add_gamma3(self, img):
        clean = (img * 255.0 + 1) / 256.0
        clean = clean * clean
        L_list = [1.0, 2.0, 4.0, 6.0, 8.0, 10.0]
        L_s = random.sample(L_list, 1)
        L = L_s[0]
        L = 10.0
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

data_dir = {'train': 'data/train', 'test': 'data/test'}

val_dataset = torchvision.datasets.ImageFolder(data_dir['test'], transform=tran_pose)
val_dataloader = torch.utils.data.DataLoader(val_dataset, 1, shuffle=False, num_workers=4)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

Unet = torch.load('imgs_lr_gai2_single_100.pt', map_location=torch.device('cpu'))
Unet = Unet.to(device)

ssim=0
psnr=0
# b=[135,137,140,141,142]1
total_num=len(val_dataset)
# 进入训练过程
if __name__ == '__main__':

    for img, _ in iter(val_dataloader):
        img = img.to(device)
        # label = label.to(device)
        intensity, noise_img = add_gamma3.add_gamma3(img)
        outputs = Unet(noise_img)
        # logits = model(img)
        img = img.cpu().numpy()
        outputs = outputs.cpu().numpy()
        b = calc_psnr(img, outputs)
        a = calc_ssim(img, outputs)

        print(a, b)
        ssim += a
        psnr += b

    ssim = ssim / total_num
    psnr = psnr / total_num
    print("ssim=", ssim)
    print("psnr=", psnr)
    # print(f'In epoch {e}, average training loss is {avg_train_loss}, average training acc is {avg_train_acc}.')






