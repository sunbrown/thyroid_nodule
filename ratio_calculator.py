import numpy as np
import os
import torch
from torch.autograd import Variable
from torch import nn
from PIL import Image
from torchvision import transforms
from unet import UNet
import pickle
import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
txt = r'D:\code\thyroid_nodule\label\train_label_balance1.txt'
img_dir = './data/img/'
n_class = 3
num = 21088#5399
loop_iou, loop_dice = [], []
fh = open(txt, 'r')
ratio = []
labels = []

model = UNet(n_channels=3, n_classes=n_class)
model = nn.DataParallel(model, device_ids=[0])
model.load_state_dict(torch.load('trainmodels.pth'))
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.3301, 0.3301, 0.3301],
                         std=[0.1938, 0.1938, 0.1938])
])
for line in fh:
    line = line.strip('\n')
    line = line.rstrip()
    words = line.split()
    path = img_dir + words[0]
    labels.append(int(words[1]))
    image = Image.open(path).convert('RGB')
    image = image.resize((224, 224), Image.ANTIALIAS)
    in_img_tensor = transform(image)
    n = in_img_tensor.size()
    data = torch.ones(1, 3, n[1], n[2])
    data[0] = in_img_tensor
    data = Variable(data.cuda(0))
    with torch.no_grad():
        output = model(data)  # 把图片输入模型处理
    out = torch.reshape(output.cpu(), (n_class, n[1], n[2]))
    pre = torch.max(out, 0)[1]
    out_img_tensor = torch.zeros(3, n[1], n[2])
    out_img_tensor[0, pre == 2] = 1
    out_img_tensor[1, pre == 1] = 1
    dots_sum = torch.sum(out_img_tensor)
    ratio_tmp = torch.sum(out_img_tensor[0]) / dots_sum
    ratio.append(ratio_tmp.numpy())
with open('r1.txt', 'wb') as f:
    pickle.dump(ratio, f)
with open('l1.txt', 'wb') as f:
    pickle.dump(labels, f)
with open('r1.txt', 'rb') as f:
    ratio = pickle.load(f)
with open('l1.txt', 'rb') as f:
    labels = pickle.load(f)

accuracy = {}
predict = np.empty(num)
threshold = 0.1
while threshold <= 0.9:
    for i in range(num):
        if ratio[i] >= threshold:
            predict[i] = 1
        else:
            predict[i] = 0
    predict_right = 0.0
    for j in range(num):
        if predict[j] == int(labels[j]):
            predict_right += 1.0
    current_accuracy = (predict_right / num)
    accuracy[str(threshold)] = current_accuracy
    threshold = threshold + 0.01
# 将字典按照value也就是准确率排序
temp = sorted(accuracy.items(), key=lambda d: d[1], reverse=True)
# [(threshold1, acc1),(threshold2，acc2).....]
highestAccuracy = temp[0][1]
thres = temp[0][0]
print(highestAccuracy, thres)
