import os
import torch
import scipy.io as sio
import numpy as np
from torch.autograd import Variable
from torch.optim import Optimizer
from tqdm import tqdm
import torch.nn.functional as F
import torch.nn as nn
import pdb

train_mat = []
test_mat = []


class Trainer(object):

    def __init__(self, model, optimizer, loss_f, save_dir=None, save_freq=1):
        self.model = model
        self.optimizer = optimizer
        self.loss_f = loss_f
        self.save_dir = save_dir
        self.save_freq = save_freq

    def _loop(self, data_loader, is_train=True):
        loop_loss, loop_iou= [], []
        num = [np.array([21088, 10544, 10544, 21088, 21088]), np.array([5399, 1885, 3514, 5399, 5399])]
        for data, target in tqdm(data_loader):
            self.optimizer.zero_grad()
            data, target = data.cuda(), target.cuda()
            data, target = Variable(data, requires_grad=True), Variable(target)
            if is_train:
                torch.set_grad_enabled(True)
                output = self.model(data)
            else:
                torch.set_grad_enabled(False)
                output = self.model(data)
            n = output.size()
            softmax = nn.Softmax(dim=1)
            output = softmax(output)
            out_flat = torch.reshape(output, (n[0], n[1], n[2]*n[3]))
            tar_flat = torch.reshape(target, (n[0], n[2]*n[3]))
            loss = F.nll_loss(torch.log(out_flat), tar_flat)
            # loss = self.loss_f(out_flat, tar_flat)
            loop_loss.append(loss.detach() / len(data_loader))
            for i in range(n[0]):
                iou = np.zeros(5)
                out = out_flat[i].clone().data
                tar = tar_flat[i].clone().data
                pre = torch.max(out, 0)[1]
                pre[pre == 2] += 1
                pre[pre == 1] += 1
                tar[tar == 2] += 1
                tar[tar == 1] += 1
                ad = pre + tar
                if len(ad[ad == 0]) != 0:
                    iou[0] = len(ad[ad == 0])/(len(ad[ad == 0]) + len(ad[ad == 2]) + len(ad[ad == 3]))
                if len(ad[ad == 4]) != 0:
                    iou[1] = len(ad[ad == 4])/(len(ad[ad == 2]) + len(ad[ad == 4]) + len(ad[ad == 5]))
                if len(ad[ad == 6]) != 0:
                    iou[2] = len(ad[ad == 6])/(len(ad[ad == 3]) + len(ad[ad == 5]) + len(ad[ad == 6]))
                if len(ad[ad == 4]) + len(ad[ad == 6]) != 0:
                    iou[3] = len(ad[ad >= 4]) / len(ad[ad >= 2])
                iou[4] = (iou[0] + iou[3]) / 2
                loop_iou.append(iou)
            if is_train:
                loss.backward()
                self.optimizer.step()
        mode = "train" if is_train else "test"
        d = num[0] if is_train else num[1]
        # print(f">>>[{mode}] loss: {sum(loop_loss):.2f}/accuracy: {sum(correct):.2%}")
        print(mode + ' loss: {:.6f}, iou: {}'.format(
            sum(loop_loss), sum(loop_iou)/d))
        return sum(loop_loss), sum(loop_iou/d)

    def train(self, data_loader):
        self.model.train()
        loss, iou = self._loop(data_loader)
        return loss, iou

    def test(self, data_loader):
        self.model.eval()
        loss, iou = self._loop(data_loader, is_train=False)
        return loss, iou

    def loop(self, epochs, train_data, test_data, scheduler=None):
        for ep in range(1, epochs + 1):
            if scheduler is not None:
                scheduler.step()
            # print(f"epochs: {ep}")
            print('epoch {}'.format(ep))
            train_loss, train_iou = self.train(train_data)
            test_loss, test_iou = self.test(test_data)
            train_mat.append((train_loss, train_iou))
            test_mat.append((test_loss, test_iou))
            self.save(ep)
            sio.savemat('./temp/loss.mat', {'train_mat': train_mat, 'test_mat': test_mat})

    def save(self, epoch, **kwargs):
        if self.save_dir:
            # name = f"weight-{epoch}-" + "-".join([f"{k}_{v}" for k, v in kwargs.items()]) + ".pkl"
            # torch.save({"weight": self.model.state_dict()},
            #            os.path.join(self.save_dir, name))
            name = './temp/train' + str(epoch) + 'models.pth'
            torch.save(self.model.state_dict(), name)


# copied from pytorch's master
class _LRScheduler(object):
    def __init__(self, optimizer, last_epoch=-1):
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer
        if last_epoch == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError("param 'initial_lr' is not specified "
                                   "in param_groups[{}] when resuming an optimizer".format(i))
        self.base_lrs = list(
            map(lambda group: group['initial_lr'], optimizer.param_groups))
        self.step(last_epoch + 1)
        self.last_epoch = last_epoch

    def get_lr(self):
        raise NotImplementedError

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


class StepLR(_LRScheduler):
    """Sets the learning rate of each parameter group to the initial lr
    decayed by gamma every step_size epochs. When last_epoch=-1, sets
    initial lr as lr.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        step_size (int): Period of learning rate decay.
        gamma (float): Multiplicative factor of learning rate decay.
            Default: 0.1.
        last_epoch (int): The index of last epoch. Default: -1.
    Example:
        >>> # Assuming optimizer uses lr = 0.5 for all groups
        >>> # lr = 0.05     if epoch < 30
        >>> # lr = 0.005    if 30 <= epoch < 60
        >>> # lr = 0.0005   if 60 <= epoch < 90
        >>> # ...
        >>> scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
        >>> for epoch in range(100):
        >>>     scheduler.step()
        >>>     train(...)
        >>>     validate(...)
    """

    def __init__(self, optimizer, step_size, gamma=0.1, last_epoch=-1):
        self.step_size = step_size
        self.gamma = gamma
        super(StepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr * self.gamma ** (self.last_epoch // self.step_size)
                for base_lr in self.base_lrs]
