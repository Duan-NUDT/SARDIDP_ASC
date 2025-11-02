import numpy as np
from PIL.Image import Image
from torch.utils.data import Dataset, DataLoader
from add_noise import *
import os

class MyDataset(Dataset):
    def __init__(self, path, transform, sigma=30, ex=1):
        self.transform = transform
        self.sigma = sigma

        for _, _, files in os.walk(path):
            self.imgs = [path + file for file in files if Image.open(path + file).size >= (96,96)] * ex
            #上面的意思是仅读取大小大于或等于96*96的图片，ex是数据增广系数，即把同一张图片复制多份以达到扩充数据量的目的
            #由于COCO2014数据集训练图片有八万多张，数据量足够大不需要增广，因此ex设置为1
        np.random.shuffle(self.imgs) #随机打乱顺序

    def __getitem__(self, index):
        tempImg = self.imgs[index]
        tempImg = Image.open(tempImg).convert('RGB') #数据集中有部分图片为灰度图，将所有图片转换为RGB格式
        Img = np.array(self.transform(tempImg))/255 #像素归一化至[0,1]
        nImg = add_gamma3(Img, self.sigma)  # 添加高斯噪声
        #nImg = addGaussNoise(Img, self.sigma) #添加高斯噪声
        Img = torch.tensor(Img.transpose(2,0,1)) #由于Image.open加载的图片是H*W*C的格式，因此转换成C*H*W的格式
        nImg = torch.tensor(nImg.transpose(2,0,1))
        return Img, nImg

    def __len__(self):
        return len(self.imgs)
Datasets = MyDataset()

def get_data(batch_size, train_path, val_path, transform, sigma, ex=1):
    train_dataset = MyDataset(train_path, transform, sigma, ex)
    val_dataset = MyDataset(val_path, transform, sigma, ex)
    train_iter = DataLoader(train_dataset, batch_size, drop_last=True, num_workers=6)
    val_iter = DataLoader(val_dataset, batch_size, drop_last=True, num_workers=6)

    return train_iter, val_iter

# train_iter, val_iter = get_data(batch_size, train_path, val_path, randomcrop, 30, ex=1)