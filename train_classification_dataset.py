import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import make_classification
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

X, y = make_blobs(n_samples=600, n_features=2, centers=3)

plt.figure(figsize=(8, 8))
plt.scatter(X[:, 0], X[:, 1], marker="o", c=y, s=25, edgecolor="k")
plt.show()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)

train_x = torch.Tensor(X_train)  # transform to torch tensor
train_y = torch.Tensor(y_train)

train_x = torch.Tensor(X_train)  # transform to torch tensor
train_y = torch.Tensor(y_train)

test_x = torch.Tensor(X_test)  # transform to torch tensor
test_y = torch.Tensor(y_test)

train_dataset = TensorDataset(train_x, train_y)  # create your datset
train_loader = DataLoader(train_dataset, batch_size=32)  # create your dataloader

test_dataset = TensorDataset(test_x, test_y)  # create your datset
test_loader = DataLoader(test_dataset, batch_size=10)  # create your dataloader

class Net(nn.Module):
    def __init__(self, input_features, classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_features, 5*input_features)  # 5*5 from image dimension
        self.fc2 = nn.Linear(5*input_features, classes)
        self.relu = nn.ReLU(inplace=True)
        self.net = nn.Sequential(self.fc1,self.relu,self.fc2)
    def forward(self,x):
        return F.softmax( self.net(x))

def configure_loss_function():
    return torch.nn.CrossEntropyLoss()


def configure_optimizer(model):
    return torch.optim.Adam(model.parameters())


def full_gd(model, criterion, optimizer, loader, n_epochs=100):
    train_losses = []
    test_losses = []
    for epoch in range(n_epochs):
        for it, (X, y) in enumerate(loader):

            outputs = model(X)
            loss = criterion(outputs, y.long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # outputs_test = model(X_test)
            # loss_test = criterion(outputs_test, y_test)

            train_losses.append(loss.item())
            # test_losses.append(loss_test.item())

            if (it + 1) % 10 == 0:
                print(
                    f'In this epoch {epoch*it + 1}/{len(loader)*n_epochs}, Training loss: {loss.item():.4f}, ')

    return train_losses, test_losses

model = Net(2,3)
print(model)
criterion = configure_loss_function()
optimizer = configure_optimizer(model)
train_losses, test_losses = full_gd(model, criterion, optimizer, train_loader)

plt.plot(train_losses, label='train loss')
plt.plot(test_losses, label='test loss')
plt.legend()
plt.show()
