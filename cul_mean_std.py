import os
import torch
import numpy as np
import glob
from PIL import Image
from tqdm import tqdm
from torchvision import transforms


txt = './train_label.txt'
content_txt = open(txt, 'r')
name = []

for line in content_txt:
    line = line.strip('\n')
    line = line.rstrip()
    words = line.split()
    name.append(words[0])
    
im_path = './data/img/'
len = len(name)
print(len)
total_im = np.zeros([512, 512, len])
m, n = 0, 0

for i in tqdm(range(len - 1)):
    im = Image.open(im_path + name[i]).convert('L')
    im = im.resize((512, 512), Image.ANTIALIAS)
    im = np.array(im)
    total_im[:, :, i] = im

tensor_im = transforms.ToTensor()(total_im)
print(torch.mean(tensor_im))
print(torch.std(tensor_im))

img_mean = 0.3231054213495149
img_std = 0.19280945014066775