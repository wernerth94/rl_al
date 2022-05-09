import torchvision.transforms

from Data import load_cifar10_pytorch
import numpy as np
import matplotlib.pyplot as plt
import random, os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torchvision.transforms import *
from Misc import accuracy
import argparse

arg_parse = argparse.ArgumentParser()
arg_parse.add_argument("--reconmult", "-r", type=float, default=0.00025)
args = arg_parse.parse_args()

FOLDER = "encoder_gs"
os.makedirs(FOLDER, exist_ok=True)
MODEL_FILE = f"{FOLDER}/encoder_{args.reconmult}.pt"

TRAIN_EPOCHS = 1

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
)
HIDDEN_DIM = 108
class_head = nn.Sequential(
    nn.Linear(HIDDEN_DIM, 10),
    nn.Softmax(dim=1)
)
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
)



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



def train_new_model():
    lr = 0.0001
    class_model = nn.Sequential(encoder, class_head)
    recon_model = nn.Sequential(encoder, recon_head)
    optimizer_class = optim.Adam(class_model.parameters(), lr=lr)
    optimizer_recon = optim.Adam(recon_model.parameters(), lr=lr)
    loss_ce = nn.CrossEntropyLoss()
    loss_recon = nn.MSELoss()

    x_train, y_train, x_test, y_test = load_cifar10_pytorch(return_tensors=True)
    VAL_SIZE = int(len(x_train) * 0.2)
    val_ids = np.random.choice(len(x_train), size=VAL_SIZE, replace=False)
    val_mask = [ v not in val_ids for v in np.arange(len(x_train)) ]
    x_val = x_train[val_ids]
    y_val = y_train[val_ids]
    x_train = x_train[val_mask]
    y_train = y_train[val_mask]

    BATCH_SIZE = 128
    train_dataloader = DataLoader(AugDataset(x_train, y_train), batch_size=BATCH_SIZE)
    val_dataloader = DataLoader(AugDataset(x_val, y_val), batch_size=BATCH_SIZE)

    class_errors = []
    recon_errors = []
    val_accs = []
    try:
        for epoch in range(TRAIN_EPOCHS):
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

    except KeyboardInterrupt as ex:
        pass

    if os.path.exists(MODEL_FILE):
        os.remove(MODEL_FILE)
    torch.save(encoder, MODEL_FILE)

    # plt.plot(val_accs, label="Accuracy")
    # plt.plot(class_errors, label="Class. Loss")
    # plt.plot(recon_errors, label="Recon. Loss")
    # plt.legend(); plt.grid()
    # plt.show()


if __name__ == '__main__':
    #convert_dataset()
    train_new_model()
    pass