import os
from utils import Trainer, StepLR
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torchvision.models as models
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import random
from PIL import Image
from unet import UNet


def default_loader(path, is_img):
    if is_img:
        img = Image.open(path).convert('RGB')
    else:
        img = Image.open(path).convert('1')
    img = img.resize((224, 224), Image.ANTIALIAS)
    return img


class MyDataset(Dataset):
    def __init__(self, mode,
                 txt,
                 transform=None,
                 loader=default_loader):
        fh = open(txt, 'r')
        imgs = []
        for line in fh:
            line = line.strip('\n')  # 移除字符串头尾指定的字符（默认为空格或换行符）或字符序列
            line = line.rstrip()  # 删除 string 字符串末尾的指定字符（默认为空格）
            words = line.split()  # 通过指定分隔符对字符串进行切片，如果参数 num 有指定值，则分隔 num+1 个子字符串
            imgs.append((words[0], int(words[1])))
        self.imgs = imgs
        self.transform = transform
        self.loader = loader
        self.mode = mode

    def __getitem__(self, index):  # 相当于字典操作，返回index对应的item
        fn, label = self.imgs[index]
        img = self.loader('D:/code/thyroid_nodule/' + 'data/img/' + fn, True)
        tar = self.loader('D:/code/thyroid_nodule/' + 'data/mask/' + fn, False)
        if random.randint(0, 1) and self.mode == 'train':
            img = img.transpose(Image.FLIP_LEFT_RIGHT)  # 随机左右翻转
            tar = tar.transpose(Image.FLIP_LEFT_RIGHT)
        img = self.transform(img)
        tar = torch.from_numpy(np.array(tar, np.int64, copy=False))
        if label:
            tar[tar == 1] += 1
        return img, tar

    def __len__(self):
        return len(self.imgs)  # 返回对象（字符、列表、元组等）长度或项目个数。


def main(batch_size, data_root):
    train_data = MyDataset(
        mode='train',
        txt=data_root + 'train_label_balance.txt',
        transform=transforms.Compose([
            # transforms.RandomResizedCrop(224),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.3301, 0.3301, 0.3301],
                                 std=[0.1938, 0.1938, 0.1938])
        ]))
    test_data = MyDataset(
        mode='test',
        txt=data_root + 'test_label_balance.txt',
        transform=transforms.Compose([
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.3301, 0.3301, 0.3301],
                                 std=[0.1938, 0.1938, 0.1938])
        ]))

    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True)
    test_loader = DataLoader(
        test_data, batch_size=batch_size, shuffle=False, num_workers=8)
    model = UNet(n_channels=3, n_classes=3)
    print(model)
    model = nn.DataParallel(model.cuda(), device_ids=[0])
    optimizer = optim.SGD(params=model.parameters(),
                          lr=0.01, momentum=0.9, weight_decay=1e-5)
    # optimizer = optim.Adam(params=model.parameters(), lr=0.01)
    scheduler = StepLR(optimizer, 10, gamma=0.1)
    trainer = Trainer(model, optimizer, F.cross_entropy, save_dir=".")
    trainer.loop(50, train_loader, test_loader, scheduler)


if __name__ == '__main__':
    batch_size = 6
    read_path = 'D:/code/thyroid_nodule/'
    main(batch_size, read_path)
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
