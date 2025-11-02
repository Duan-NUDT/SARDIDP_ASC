import os
import random
import cv2
import torch
import torchvision
from PIL import Image
# from d2l.tensorflow import transpose
from matplotlib import pyplot as plt
from scipy import io
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from Unet_single import Unet_single
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

# from sar_test import num_w



def plot(train_loss):
    plt.figure(figsize=(5, 5))
    plt.plot(train_loss, label='train_loss')
    plt.title('train_loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.show()

def downsample_tensor(img, flag):
    img = torch.squeeze(img)
    img = torch.squeeze(img)
    height = img.shape[0]
    weight = img.shape[1]
    num_h = int(height / flag)
    num_w = int(weight / flag)

    # 存储降采样后的结果
    tensor_downsampled = torch.zeros(num_h, num_w)
    for i in range(num_h):
        for j in range(num_w):
            start_row = i * flag
            start_col = j * flag
            random_row = start_row + torch.randint(0, flag, (1,)).item()
            random_col = start_col + torch.randint(0, flag, (1,)).item()
            # 将选取的像素赋值给降采样后的张量
            tensor_downsampled[i, j] = img[random_row, random_col]
    tensor_downsampled = torch.unsqueeze(tensor_downsampled,dim=0)
    tensor_downsampled = torch.unsqueeze(tensor_downsampled,dim=0)
    return tensor_downsampled

def downsample_tensor2(img, img2, flag): # x, y , radio
    img = torch.squeeze(img)
    img = torch.squeeze(img)
    img2 = torch.squeeze(img2)
    img2 = torch.squeeze(img2)
    height = img.shape[0]
    weight = img.shape[1]
    num_h = int(height / flag)
    num_w = int(weight / flag)

    # 存储降采样后的结果
    tensor_downsampled = torch.zeros(num_h, num_w)
    tensor_downsampled2 = torch.zeros(num_h, num_w)
    for i in range(num_h):
        for j in range(num_w):
            start_row = i * flag
            start_col = j * flag
            random_row = start_row + torch.randint(0, flag, (1,)).item()
            random_col = start_col + torch.randint(0, flag, (1,)).item()
            # 将选取的像素赋值给降采样后的张量
            tensor_downsampled[i, j] = img[random_row, random_col]
            tensor_downsampled2[i, j] = img2[random_row, random_col]

    tensor_downsampled = torch.unsqueeze(tensor_downsampled,dim=0)
    tensor_downsampled = torch.unsqueeze(tensor_downsampled,dim=0)
    tensor_downsampled2 = torch.unsqueeze(tensor_downsampled2,dim=0)
    tensor_downsampled2 = torch.unsqueeze(tensor_downsampled2,dim=0)
    return tensor_downsampled, tensor_downsampled2  # x, y

def train_img_save(test_img, i, name,save_path):
    save_file_path = os.path.join(save_path, str(i))
    if not os.path.exists(save_file_path):
        os.makedirs(save_file_path)
    a = torch.squeeze(test_img, dim=0)
    a = torch.squeeze(a, dim=0)
    a[a > 1.0] = 1.0
    a[a < 0.0] = 0.0
    b = a.detach().cpu().numpy()
    cv2.imwrite(save_file_path + '/'  +  f'{name}.png', (b/0.6) * 255)

def random_crop_pair(array1, array2, crop_size=(64, 64)):

    # assert array1.shape == array2.shape == (128, 128), "输入数组的大小必须为 (128, 128)"
    crop_height, crop_width = crop_size

    # 随机生成裁剪的起始坐标，确保裁剪不会超出边界
    max_x = array1.shape[0] - crop_height - 2
    max_y = array1.shape[1] - crop_width - 2

    x_start = np.random.randint(0, max_x + 1)
    y_start = np.random.randint(0, max_y + 1)

    # 裁剪出两个 (64, 64) 的子数组
    cropped_array1 = array1[x_start:x_start + crop_height, y_start:y_start + crop_width]
    cropped_array2 = array2[x_start:x_start + crop_height, y_start:y_start + crop_width]

    return cropped_array1, cropped_array2

tran_pose = torchvision.transforms.Compose([  # 对图片裁剪进行处理 此时输出为1x32x32

    torchvision.transforms.Grayscale(1),  # 将三通道图像转换为单通道灰度图
    torchvision.transforms.RandomCrop(size=(64, 64)),
    torchvision.transforms.ToTensor(),
])

trans_pose1 = torchvision.transforms.Compose([
    # torchvision.transforms.Grayscale(1),
    torchvision.transforms.ToTensor(),
    # torchvision.transforms.RandomCrop(size=(64, 64)),
])

transform = torchvision.transforms.Compose([
    torchvision.transforms.Grayscale(1),
    torchvision.transforms.ToTensor(),
])


class Mstar_Complex(Dataset):
    def __init__(self, list_dir, transform=None):
        self.data_list = {'mat_list': [], 'ASC_list': []}
        self.transform = transform

        f = open(list_dir, 'r')
        for i in f.readlines():
            self.data_list['mat_list'].append(i.strip().split()[0])
            self.data_list['ASC_list'].append(i.strip().split()[1])

    def __getitem__(self, idx):

        mat_file = self.data_list['mat_list'][idx]
        asc_file = self.data_list['ASC_list'][idx]
        depthmap = io.loadmat(mat_file)
        # A = depthmap['Img']
        data = depthmap['data']
        ABS = abs(data).astype(np.float32)
        ASC_map = io.loadmat(asc_file)
        Asc = ASC_map['image_bi']
        ASC = abs(Asc).astype(np.float32)
        A_crop, ASC_crop = random_crop_pair(ABS, ASC)
        # zero_matrix = np.zeros((64, 64))
        # zero_matrix[0:ASC_crop.shape[0], 0:ASC_crop.shape[1]] = ASC_crop
        if self.transform:
            img = self.transform(A_crop)
            asc = self.transform(ASC_crop)
        else:
            img = A_crop
            asc = ASC_crop

        return img, asc
    def __len__(self):
        return len(self.data_list['mat_list'])

data_dir = {'train': 'data/Mstar128', 'png': 'data/png', 'test': 'data/test'}
dataset = 'WJ_datalist_12.txt'
train_dataset = Mstar_Complex(dataset, transform=trans_pose1)
# train_dataset = torchvision.datasets.ImageFolder(data_dir['train'], transform=tran_pose)
train_png_dataset = torchvision.datasets.ImageFolder(data_dir['png'], transform=tran_pose)
val_dataset = torchvision.datasets.ImageFolder(data_dir['test'], transform=tran_pose)

train_dataloader = torch.utils.data.DataLoader(train_dataset, 1, shuffle=True, num_workers=4)
train_png_dataloader = torch.utils.data.DataLoader(train_png_dataset, batch_size=1, shuffle=False, num_workers=4)
val_dataloader = torch.utils.data.DataLoader(val_dataset, 1, shuffle=False, num_workers=4)

# dataloader2 = DataLoader(dataset2, batch_size=len(dataset2), shuffle=False)
# 对数据集路径进行处理


def shuffle_img(img, patch_size):
    img2 = torch.zeros_like(img)
    height, width = img2.shape[2], img2.shape[3]
    num_h = int(height / patch_size)
    num_w = int(width / patch_size)
    num = num_h * num_w
    lista = range(num)
    list1 = random.sample(lista, num)
    for i in range(num):
        h_ind1 = int(i / num_h)
        w_ind1 = i % num_w
        j = list1[i]
        h_ind2 = int(j / num_h)
        w_ind2 = j % num_w

        img2[:,:,h_ind1*patch_size:(h_ind1+1)*patch_size,w_ind1*patch_size:(w_ind1+1)*patch_size] = img[:,:,h_ind2*patch_size:(h_ind2+1)*patch_size,w_ind2*patch_size:(w_ind2+1)*patch_size]

    return img2


device = 'cuda' if torch.cuda.is_available() else 'cpu'

Unet = Unet_single()
Unet = Unet.to(device)

pretrained_model = torch.load('imgs_crop=64_new_adam_single_100.pt', map_location=torch.device('cpu'))
# pretrained_model = pretrained_model.to(device)
# 创建优化器，初始学习率设置为1e-4
# 定义学习率衰减函数
initial_lr = 1e-4
optim = torch.optim.Adam(Unet.parameters(), lr=initial_lr)

# 定义学习率调整函数
def lr_lambda(epoch):
    if epoch < 50:
        return 1.0
    else:
        return 1.0 - (epoch - 50) / 50 * (1 - 1e-5 / 1e-4)

# 初始化学习率调度器
scheduler = LambdaLR(optim, lr_lambda=lr_lambda)


loss_fn1 = nn.MSELoss()  # L1-损失
loss_fn1 = loss_fn1.to(device)
loss_fn2 = nn.MSELoss()  # L2-损失
loss_fn2 = loss_fn2.to(device)
# a = 0.1 # 超参数
train_loss = list()
radio = 2
train_step = 0  # 训练次数
epoch = 100
flag = 0

intensity = 0
lr_history = []

# 进入训练过程
if __name__ == '__main__':
    all_png_data = [data for data in train_png_dataloader]
    for i in range(epoch):
        print(f'\n-----第{i + 1}次训练开始-----')
        Unet.train()
        train_step = len(train_dataloader)
        losses = []
        m = 0
        alpha = 1
        save_path = 'result_from_WJdata/WJdata12'
        with tqdm(total=train_step, desc=f'Train Epoch {i + 1}/{epoch}', postfix=dict,mininterval=0.3) as pbar:
                for img, ASC in iter(train_dataloader):
                    # img[img>100000] =100000
                    # img[img<0] = 0
                    # img = img / 2000000
                    img[img>1000000]=1000000
                    img[img<0]=0
                    img = img / 1000000
                    img = img.to(device)
                    # img[img>1] = 1
                    # img[img<0.2] = 0.2
                    ASC = ASC.to(device)
                    random_index = random.randint(0, len(all_png_data) - 1)
                    image2, label2 = all_png_data[random_index]
                    image2 = image2.to(device)  # 1
                    outputs = Unet(img)# 1
                    N = (img / outputs).square()
                    M = N / torch.mean(N)
                    new1 = torch.sqrt(M)
                    new = shuffle_img(new1, 8)
                    D = image2 * new  # 1
                    # D[D>1.0] = 1.0
                    D[D < 0.0] = 0.0
                    out_from_D = Unet(D.detach())  # 1
                    img_down, out_down = downsample_tensor2(img, outputs, radio)
                    out_label = pretrained_model(img_down)
                    # out_pre = pretrained_model(img)
                    loss_1 = loss_fn1(outputs*ASC, img*ASC)
                    loss_3 = loss_fn1(out_from_D, image2)
                    loss_2 = loss_fn2(out_down, out_label)

                    loss =   alpha * loss_1 + loss_2 + loss_3
                    loss.backward()
                    optim.step()
                    optim.zero_grad()

                    losses.append(loss.item())

                    pbar.set_postfix(**{'Train Loss': np.mean(losses)},**{'alpha': alpha} )
                    pbar.update(1)

                    # train_img_save(img, i, 'img', save_path)
                    # train_img_save(image2, i, 'image2', save_path)
                    # train_img_save(img_down, i, 'img_down', save_path)
                    # train_img_save(D, i, 'D', save_path)
                    # train_img_save(10*outputs, i, 'outputs', save_path)
                    # train_img_save(out_down, i, 'out_down', save_path)
                    # # train_img_save(N, i, 'N',save_path)
                    # train_img_save(out_label, i, 'out_label', save_path)
                    # train_img_save(out_from_D, i, 'out_from_D', save_path)
                    # train_img_save(out_pre, i, 'out_pre', save_path)
                    # print('done')
                scheduler.step()
                # avg_train_loss = sum(losses) / len(losses)

        # print(f'In epoch {e}, average training loss is {avg_train_loss}, average training acc is {avg_train_acc}.')
                current_lr = scheduler.get_last_lr()[0]
        # current_lr = optim.param_groups[0]['lr']
                lr_history.append(current_lr)
                print(f" \n End of Epoch {i + 1}: Current Learning Rate: {current_lr:.6f}")
                print(f'第{i + 1}轮训练结束')

        flag = flag + 1
        train_img_save(img, i, 'img',save_path)
        train_img_save(image2, i, 'image2', save_path)
        train_img_save(img_down, i, 'img_down',save_path)
        train_img_save(D, i, 'D', save_path)
        train_img_save(outputs, i, 'outputs',save_path)
        train_img_save(out_down, i, 'out_down',save_path)
        # train_img_save(N, i, 'N',save_path)
        train_img_save(out_label, i, 'out_label',save_path)
        train_img_save(out_from_D, i, 'out_from_D', save_path)

        depthmap = io.loadmat('data/Mstar_fushu/BMP2/HB03787.mat')
        wj_data = io.loadmat('data/WJ_sandstone/car1/KU_HH_15_0_650808.mat')
        data = depthmap['Img'] # Mstar
        data[data>1]=1
        data[data<0]=0
        WJ = wj_data['data']  # WJ data
        A = abs(WJ)
        A[A > 1000000] = 1000000
        A[A < 0] = 0
        A = A/1000000
        mod_data = data.reshape(1, 128, 128) # Mstar
        mod = A.reshape(1, 128, 128)   # wj data
        # mod = A.unsqueeze(0)
        out = torch.from_numpy(mod_data).float()
        img_wj = torch.from_numpy(mod).float()
        imgs = out
        imgs = imgs.unsqueeze(0)
        imgs_wj = img_wj.unsqueeze(0)
        # imgs = imgs.unsqueeze(0)
        # imgs, targets = data
        imgs = imgs.to(device)
        imgs_wj = imgs_wj.to(device)
        outputs = Unet(imgs)
        outputs_wj = Unet(imgs_wj)
        outputs_wj = outputs_wj[0,0,:,:]
        a = torch.squeeze(outputs, dim=0)
        a = torch.squeeze(a, dim=0)
        a[a>1.0] = 1.0
        a[a<0.0] = 0.0
        b = a.detach().cpu().numpy()
        cv2.imwrite(f'result_from_WJdata/WJdata12/complex_mstar_{i+1}.png', (b/0.6)*255)
        outputs_wj[outputs_wj>1.0] = 1.0
        outputs_wj[outputs_wj<0.0] = 0.0
        wj_out = outputs_wj.detach().cpu().numpy()
        cv2.imwrite(f'result_from_WJdata/WJdata12/complex_wj_{i+1}.png', (wj_out/0.6)*255)
        if flag % (0.1 * epoch) == 0:
            torch.save(Unet, f'Sar_Detach_model/Sar_wj12_guang_alpha_{alpha}_shuffle_D.detach_{i + 1}.pt')


