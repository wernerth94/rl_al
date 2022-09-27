import sys
import getpass
print(F"The user is: {getpass.getuser()}")
print(F"The virtualenv is: {sys.prefix}")

# path additions for the cluster
sys.path.append("core")
sys.path.append("evaluation")
sys.path.append("config")
print(F"updated path is {sys.path}")

from torchvision.models import resnet18
import torchvision.transforms

from Data import load_cifar10_pytorch
import numpy as np
from tqdm import tqdm
from time import sleep
import random, os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from torchvision.transforms import *
import argparse


class AugDataset(TensorDataset):
    transform = Compose([
        RandomHorizontalFlip(p=0.3),
        RandomVerticalFlip(p=0.3),
        RandomCrop(32, padding=10),
        RandomRotation(20)
    ])
    def __getitem__(self, idx):
        x, y = super().__getitem__(idx)
        x = AugDataset.transform(x)
        return (x, y)

class EarlyStop:
    def __init__(self, threshold=0.001, patience=5):
        self.threshold = threshold
        self.loss_multiplier = 1.0 + threshold
        self.patience = patience
        self.best_loss = np.inf
        self.counter = 0

    def check_stop(self, loss_value):
        if loss_value * self.loss_multiplier < self.best_loss:
            self.best_loss = loss_value
            self.counter = 0
        else:
            if self.counter >= self.patience:
                return True
            self.counter += 1
        return False


CLUSTER = False
if sys.prefix.startswith('/home/werner/miniconda3'):
    CLUSTER = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

x_train, y_train, x_test, y_test = load_cifar10_pytorch(return_tensors=False, channelFirst=True)
x_train = x_train.to(device)
y_train = y_train.to(device)
x_test = x_test.to(device)
y_test = y_test.to(device)

model = resnet18(pretrained=True)
optimizer_class = optim.Adam(model.parameters(), lr=0.1)
loss_ce = nn.CrossEntropyLoss()
model.to(device)

BATCH_SIZE = 128
train_dataloader = DataLoader(AugDataset(x_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(AugDataset(x_test, y_test), batch_size=100, shuffle=False)

scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer_class, [63, 86], gamma=0.1)
early_stop = EarlyStop()
class_errors = []
recon_errors = []
val_accs = []
try:
    for epoch in range(100):
        for batch_x, batch_y in tqdm(train_dataloader, disable=CLUSTER):
            yHat = model(batch_x)
            loss_ce(yHat, batch_y)
            optimizer_class.zero_grad()
            loss_ce.backward()
            optimizer_class.step()

        sum_class = 0.0
        sum_recon = 0.0
        acc = 0.0
        counter = 0
        for batch_x, batch_y in val_dataloader:
            z = encoder(batch_x)
            yHat = class_head(z)
            recon = recon_head(z)
            acc += accuracy(yHat.detach(), batch_y)
            class_loss = loss_ce(yHat, batch_y)
            recon_loss = loss_recon(recon, batch_x)
            sum_class += class_loss.detach().cpu().numpy()
            sum_recon += recon_loss.detach().cpu().numpy()
            counter += 1
        sum_class /= counter; class_errors.append(sum_class)
        sum_recon /= counter; recon_errors.append(sum_recon)
        acc /= counter; val_accs.append(acc)
        print("%d: classification loss %1.3f reconstruction loss %1.1f - Acc %1.3f"%(epoch, sum_class, sum_recon, acc))
        sleep(0.1) # to fix some printing ugliness with tqdm
        if early_stop.check_stop(sum_class + args.reconmult * sum_recon):
            print("early stop")
            break
except KeyboardInterrupt as ex:
    pass