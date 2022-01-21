from Data import loadCifar
import numpy as np
import matplotlib.pyplot as plt
import random, os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

def accuracy(yHat, labels):
    yHat = torch.argmax(yHat, dim=1)
    labels = torch.argmax(labels, dim=1)
    correct = yHat == labels
    acc = torch.sum(correct) / len(yHat)
    return acc.numpy()

encoder = nn.Sequential(
    nn.Conv2d(3, 32, (3,3)),
    nn.BatchNorm2d(32),
    nn.LeakyReLU(),
    nn.Conv2d(32, 64, (3,3)),
    nn.BatchNorm2d(64),
    nn.Dropout2d(p=0.05),
    nn.LeakyReLU(),
    nn.Conv2d(64, 128, (3,3), stride=2),
    nn.LeakyReLU(),
    nn.Conv2d(128, 64, (3,3), stride=2),
    nn.BatchNorm2d(64),
    nn.Dropout2d(p=0.05),
    nn.LeakyReLU(),
    nn.Conv2d(64, 24, (3,3)),
    nn.Flatten()
)
HIDDEN_DIM = 384
head = nn.Sequential(
    # nn.Linear(400, 64),
    # nn.LeakyReLU(),
    # nn.Linear(64, 10),
    nn.Linear(HIDDEN_DIM, 10),
    nn.Softmax(dim=1)
)

MODEL_FILE = "encoder_model/encoder.pt"


def convert_dataset():
    x_train, y_train, x_test, y_test = loadCifar(return_tensors=False, channelFirst=True)
    encoder = torch.load(MODEL_FILE)
    encoder.eval()
    with torch.no_grad():
        train_result = []
        for i in range(len(x_train)):
            train_result.append(encoder(torch.from_numpy(x_train[i:i+1]).float()).numpy()[0])
        x_train = np.array(train_result)

        test_result = []
        for i in range(len(x_test)):
            test_result.append(encoder(torch.from_numpy(x_test[i:i+1]).float()).numpy()[0])
        x_test = np.array(test_result)

    OUT_FILE = "../datasets/cifar10_custom.npz"
    if os.path.exists(OUT_FILE):
        os.remove(OUT_FILE)
    np.savez(OUT_FILE,
             x_train=x_train, y_train=y_train,
             x_test = x_test, y_test = y_test)



def train_new_model():
    model = nn.Sequential(encoder, head)
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    loss = nn.CrossEntropyLoss()

    x_train, y_train, x_test, y_test = loadCifar(return_tensors=True, channelFirst=True)
    VAL_SIZE = int(len(x_train) * 0.2)
    val_ids = np.random.choice(len(x_train), size=VAL_SIZE, replace=False)
    val_mask = [ v not in val_ids for v in np.arange(len(x_train)) ]
    x_val = x_train[val_ids]
    y_val = y_train[val_ids]
    x_train = x_train[val_mask]
    y_train = y_train[val_mask]

    train_dataset = TensorDataset(x_train, y_train) # create your datset
    train_dataloader = DataLoader(train_dataset, batch_size=128) # create your dataloader
    val_dataloader = DataLoader(TensorDataset(x_val, y_val), batch_size=128) # create your dataloader

    EPOCHS = 100
    BATCH_SIZE = 128

    val_errors = []
    val_accs = []
    best_val = np.inf
    try:
        for epoch in range(EPOCHS):
            for batch_x, batch_y in train_dataloader:
                yHat = model(batch_x)
                loss_val = loss(yHat, batch_y)
                optimizer.zero_grad()
                loss_val.backward()
                optimizer.step()

            sum = 0.0
            acc = 0.0
            counter = 0
            for batch_x, batch_y in val_dataloader:
                yHat = model(batch_x)
                acc += accuracy(yHat.detach(), batch_y)
                loss_val = loss(yHat, batch_y)
                sum += loss_val.detach().numpy()
                counter += 1
            sum /= counter; val_errors.append(sum)
            acc /= counter; val_accs.append(acc)
            print("%d: %1.4f - %1.3f"%(epoch, sum, acc))
            if sum < best_val:
                best_val = sum
            else:
                pass
                # print("Early stop")
                # break
    except KeyboardInterrupt as ex:
        pass

    if os.path.exists(MODEL_FILE):
        os.remove(MODEL_FILE)
    torch.save(encoder, MODEL_FILE)

    plt.plot(val_accs, label="Accuracy")
    plt.plot(val_errors, label="Loss")
    plt.legend(); plt.grid()
    plt.show()


if __name__ == '__main__':
    #convert_dataset()
    pass