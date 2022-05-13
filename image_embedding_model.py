import sys
import getpass
print(F"The user is: {getpass.getuser()}")
print(F"The virtualenv is: {sys.prefix}")

# path additions for the cluster
sys.path.append("core")
sys.path.append("evaluation")
sys.path.append("config")
print(F"updated path is {sys.path}")


import torchvision.transforms

from Data import load_cifar10_pytorch
import numpy as np
import random, os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from torchvision.transforms import *
from Misc import accuracy
import argparse

arg_parse = argparse.ArgumentParser()
arg_parse.add_argument("--reconmult", "-r", type=float, default=0.0)
arg_parse.add_argument("--triplet", "-t", type=bool, default=False)
args = arg_parse.parse_args()

print(f"--triplet training: {args.triplet}")
print(f"--reconmult: {args.reconmult}")

FOLDER = "encoder_gs"
os.makedirs(FOLDER, exist_ok=True)
MODEL_FILE = f"{FOLDER}/encoder_{args.triplet}_{args.reconmult}.pt"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

encoder = nn.Sequential(
    nn.Conv2d(3, 32, (3,3), stride=2),
    nn.BatchNorm2d(32),
    nn.LeakyReLU(),
    nn.Conv2d(32, 64, (3,3)),
    nn.BatchNorm2d(64),
    nn.Dropout2d(p=0.05),
    nn.LeakyReLU(),
    nn.Conv2d(64, 128, (3,3)),
    nn.LeakyReLU(),
    nn.Conv2d(128, 64, (3,3)),
    nn.BatchNorm2d(64),
    nn.Dropout2d(p=0.05),
    nn.LeakyReLU(),
    nn.Conv2d(64, 12, (2,2)),
    nn.MaxPool2d(3, stride=2),
    nn.Flatten()
).to(device)
HIDDEN_DIM = 108
class_head = nn.Sequential(
    nn.Linear(HIDDEN_DIM, 10),
    nn.Softmax(dim=1)
).to(device)
recon_head = nn.Sequential(
    nn.Unflatten(1, (12, 3, 3)),
    nn.ConvTranspose2d(12, 24, (3,3)),
    nn.BatchNorm2d(24),
    nn.ConvTranspose2d(24, 32, (2,2)),
    nn.BatchNorm2d(32),
    nn.ConvTranspose2d(32, 64, (3,3), stride=2),
    nn.ConvTranspose2d(64, 64, (3,3)),
    nn.ConvTranspose2d(64, 32, (3,3), stride=2),
    nn.ConvTranspose2d(32, 3, (3,3)),
    torchvision.transforms.Resize((32, 32))
).to(device)

x_train, y_train, x_test, y_test = load_cifar10_pytorch(return_tensors=True)
x_train = x_train.to(device)
y_train = y_train.to(device)
x_test = x_test.to(device)
y_test = y_test.to(device)

class AugDataset(TensorDataset):
    transform = Compose([
        RandomHorizontalFlip(p=0.3),
        RandomVerticalFlip(p=0.3),
        #RandomCrop(25, padding=0),
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


def convert_dataset():
    x_train, y_train, x_test, y_test = load_cifar10_pytorch(return_tensors=False, channelFirst=True)
    encoder = torch.load(MODEL_FILE)
    encoder.eval()
    with torch.no_grad():
        train_result = []
        for i in range(len(x_train)):
            train_result.append(encoder(torch.from_numpy(x_train[i:i+1]).float()).cpu().numpy()[0])
        x_train = np.array(train_result)

        test_result = []
        for i in range(len(x_test)):
            test_result.append(encoder(torch.from_numpy(x_test[i:i+1]).float()).cpu().numpy()[0])
        x_test = np.array(test_result)

    OUT_FILE = "../datasets/cifar10_custom.npz"
    if os.path.exists(OUT_FILE):
        os.remove(OUT_FILE)
    np.savez(OUT_FILE,
             x_train=x_train, y_train=y_train,
             x_test = x_test, y_test = y_test)


def train_new_triplet(x_train, y_train, x_test, y_test, train_epochs=100):
    lr = 0.0001
    optimizer = optim.Adam(encoder.parameters(), lr=lr)
    loss_func = nn.TripletMarginLoss()
    img_size = x_train.shape[1:]

    BATCH_SIZE = 128
    train_dataloader = DataLoader(AugDataset(x_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    class_loaders = {}
    class_ids = {}
    class_datasets = {}
    for i, label in enumerate(y_train):
        label = int(torch.argmax(label))
        if label not in class_ids:
            class_ids[label] = list()
        class_ids[label].append(i)
    for label, ids in class_ids.items():
        class_datasets[label] = TensorDataset(x_train[ids], y_train[ids])
        class_loaders[label] = RandomSampler(class_datasets[label])

    samplers = {}
    for k, l in class_loaders.items():
        samplers[k] = iter(l)
    def sample_point(label):
        try:
            point = class_datasets[label][next(samplers[label])][0].unsqueeze(0)
        except StopIteration:
            samplers[label] = iter(class_loaders[label])
            point = class_datasets[label][next(samplers[label])][0].unsqueeze(0)
        return point

    early_stop = EarlyStop()
    errors = []
    try:
        for epoch in range(train_epochs):
            epoch_error = 0
            for batch_x, batch_y in train_dataloader:
                z = encoder(batch_x)
                pos_sample = torch.zeros(torch.Size([0]) + img_size).to(device)
                neg_sample = torch.zeros(torch.Size([0]) + img_size).to(device)
                for i, label in enumerate(batch_y):
                    label = int(torch.argmax(label))
                    s = sample_point(label)
                    pos_sample = torch.cat([pos_sample, s], dim=0)
                    l = list(range(10))
                    del l[label]
                    neg_class = np.random.choice(l)
                    s = sample_point(neg_class)
                    neg_sample = torch.cat([neg_sample, s], dim=0)
                loss_val = loss_func(z, encoder(pos_sample), encoder(neg_sample))
                optimizer.zero_grad()
                loss_val.backward()
                optimizer.step()
                epoch_error += loss_val.detach().cpu().numpy()

            epoch_error /= len(train_dataloader)
            errors.append(epoch_error)
            print("epoch: %d \tloss %1.4f"%(epoch, epoch_error))
            if early_stop.check_stop(epoch_error):
                print("early stop")
                break
    except KeyboardInterrupt:
        pass


def train_new_model(x_train, y_train, x_test, y_test, train_epochs=100):
    lr = 0.0001
    class_model = nn.Sequential(encoder, class_head).to(device)
    recon_model = nn.Sequential(encoder, recon_head).to(device)
    optimizer_class = optim.Adam(class_model.parameters(), lr=lr)
    optimizer_recon = optim.Adam(recon_model.parameters(), lr=lr)
    loss_ce = nn.CrossEntropyLoss()
    loss_recon = nn.MSELoss()

    VAL_SIZE = int(len(x_train) * 0.2)
    val_ids = np.random.choice(len(x_train), size=VAL_SIZE, replace=False)
    val_mask = [ v not in val_ids for v in np.arange(len(x_train)) ]
    x_val = x_train[val_ids]
    y_val = y_train[val_ids]
    x_train = x_train[val_mask]
    y_train = y_train[val_mask]

    BATCH_SIZE = 128
    train_dataloader = DataLoader(AugDataset(x_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(AugDataset(x_val, y_val), batch_size=BATCH_SIZE, shuffle=True)

    early_stop = EarlyStop()
    class_errors = []
    recon_errors = []
    val_accs = []
    try:
        for epoch in range(train_epochs):
            if epoch == 20:
                for param_group in optimizer_recon.param_groups:
                    param_group['lr'] = lr / 2.0
                for param_group in optimizer_class.param_groups:
                    param_group['lr'] = lr / 2.0
            elif epoch == 40:
                for param_group in optimizer_recon.param_groups:
                    param_group['lr'] = lr / 4.0
                for param_group in optimizer_class.param_groups:
                    param_group['lr'] = lr / 4.0

            for batch_x, batch_y in train_dataloader:
                z = encoder(batch_x)
                yHat = class_head(z)
                recon = recon_head(z)
                loss_val = loss_ce(yHat, batch_y) + args.reconmult * loss_recon(recon, batch_x)
                optimizer_class.zero_grad()
                optimizer_recon.zero_grad()
                loss_val.backward()
                optimizer_class.step()
                optimizer_recon.step()

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
            print("%d: class %1.4f recon %1.4f - Acc %1.3f"%(epoch, sum_class, sum_recon, acc))
            if early_stop.check_stop(sum_class):
                print("early stop")
                break
    except KeyboardInterrupt as ex:
        pass

    if os.path.exists(MODEL_FILE):
        os.remove(MODEL_FILE)
    torch.save(encoder, MODEL_FILE)

    # import matplotlib.pyplot as plt
    # plt.plot(val_accs, label="Accuracy")
    # plt.plot(class_errors, label="Class. Loss")
    # plt.plot(recon_errors, label="Recon. Loss")
    # plt.legend(); plt.grid()
    # plt.show()


if __name__ == '__main__':
    #convert_dataset()
    if args.triplet:
        train_new_triplet(x_train, y_train, x_test, y_test)
        encoder.requires_grad_(False)
    train_new_model(x_train, y_train, x_test, y_test)
    pass