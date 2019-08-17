import numpy as np
import scipy.io as sio
import torch
import os
from utils import Trainer, StepLR
import torch
import torchvision.models as models
from torch.autograd import Variable
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from unet import UNet

os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3'

path = './temp/'
n_class = 3
model = UNet(n_channels=3, n_classes=n_class)
# print(model)
# exit()
model = nn.DataParallel(model, device_ids=[0])
for epoch in range(28, 29, 1):
    model.load_state_dict(torch.load('{}train{}models.pth'.format(path, epoch)))
    # model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.3301, 0.3301, 0.3301],
                             std=[0.1938, 0.1938, 0.1938])
    ])
    loop_iou, loop_dice = [], []
    read_path = './result/test/'
    txt = './test_label_balance.txt'
    fh = open(txt, 'r')
    l = 1
    for line in fh:
        line = line.strip('\n')
        line = line.rstrip()
        words = line.split()

        img = Image.open('./data/img/' + words[0]).convert('RGB')
        img = img.resize((224, 224), Image.ANTIALIAS)
        img.save('{}{}_ori.jpg'.format(read_path, l))
        img = transform(img)
        n = img.size()
        tar = Image.open('./data/mask/' + words[0]).convert('1')
        tar = tar.resize((224, 224), Image.ANTIALIAS)
        tar = torch.from_numpy(np.array(tar, np.int64, copy=False))
        if int(words[1]):
            tar[tar == 1] += 1
        tar_imgtensor = torch.zeros(3, n[1], n[2])
        tar_imgtensor[0, tar == 2] = 1
        tar_imgtensor[1, tar == 1] = 1
        piltrans = transforms.ToPILImage('RGB')
        tar_img = piltrans(tar_imgtensor)
        tar_img.save('{}{}_0.jpg'.format(read_path, l))
        data = torch.ones(1, 3, n[1], n[2])
        data[0] = img
        data = Variable(data.cuda(0))
        with torch.no_grad():
            output = model(data)
        out = torch.reshape(output.cpu(), (n_class, n[1], n[2]))
        pre = torch.max(out, 0)[1]
        # add = pre + tar
        # mul = pre * tar
        # # add=
        # # mul=
        # iou = (len(mul[mul == 1])) / len(add[add > 0])
        # dice = 2 * len(mul[mul == 1]) / (len(pre[pre == 1]) + len(tar[tar == 1]))
        pre[pre == 2] += 1
        pre[pre == 1] += 1
        tar[tar == 2] += 1
        tar[tar == 1] += 1
        ad = pre + tar
        iou1, iou2, iou3, dice1, dice2，dice = 0, 0, 0, 0, 0，0
        if len(ad[ad == 4]) != 0:
            iou1 = len(ad[ad == 4]) / (len(ad[ad == 2]) + len(ad[ad == 4]) + len(ad[ad == 5]))
            dice1 = 2 * len(ad[ad == 4]) / (len(pre[pre == 2]) + len(tar[tar == 2]))
        if len(ad[ad == 6]) != 0:
            iou2 = len(ad[ad == 6]) / (len(ad[ad == 3]) + len(ad[ad == 5]) + len(ad[ad == 6]))
            dice2 = 2 * len(ad[ad == 6]) / (len(pre[pre == 3]) + len(tar[tar == 3]))
        if len(ad[ad == 4]) + len(ad[ad == 6]) != 0:
            iou3 = len(ad[ad >= 4]) / len(ad[ad >= 2])
            dice = 2 * len(ad[ad >= 4]) / (len(pre[pre > 0]) + len(tar[tar > 0]))
        # dice = max([dice1, dice2])
        # iou = max([iou1, iou2])
        iou = iou3
        out_imgtensor = torch.zeros(3, n[1], n[2])
        out_imgtensor[0, pre == 3] = 1
        out_imgtensor[1, pre == 2] = 1
        out_img = piltrans(out_imgtensor)
        out_img.save('{}{}_{}_{:.4f}.jpg'.format(read_path, l, epoch, dice))
        loop_iou.append(iou)
        loop_dice.append(dice)
        l += 1
        print('epoch : {}, l : {}'.format(epoch, l))
        print('平均IOU:{}'.format(sum(loop_iou) / len(loop_iou)))
        print('平均DICE:{}'.format(sum(loop_dice) / len(loop_dice)))
    # print(sum(loop_iou) / len(loop_iou))
    # print(sum(loop_dice) / len(loop_dice))
    sio.savemat(read_path + 'train_result.mat', {'iou': loop_iou})
