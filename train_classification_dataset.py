import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import make_classification
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from classifier_factory import create_classifier

#
# X, y = make_blobs(n_samples=60000, n_features=780, centers=10)
# model = create_classifier(X,y)
# print(model)
# plt.figure(figsize=(8, 8))
# plt.scatter(X[:, 0], X[:, 1], marker="o", c=y, s=25, edgecolor="k")
# plt.show()
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.33, random_state=42)
#
# train_x = torch.Tensor(X_train)  # transform to torch tensor
# train_y = torch.Tensor(y_train)
#
# train_x = torch.Tensor(X_train)  # transform to torch tensor
# train_y = torch.Tensor(y_train)
#
# test_x = torch.Tensor(X_test)  # transform to torch tensor
# test_y = torch.Tensor(y_test)
#
# train_dataset = TensorDataset(train_x, train_y)  # create your datset
# train_loader = DataLoader(train_dataset, batch_size=32)  # create your dataloader
#
# test_dataset = TensorDataset(test_x, test_y)  # create your datset
# test_loader = DataLoader(test_dataset, batch_size=10,drop_last=True)  # create your dataloader


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train = datasets.MNIST('../data', train=True, download=True,
                       transform=transform)
test = datasets.MNIST('../data', train=False,
                      transform=transform)
BATCHSIZE = 100
train_loader = torch.utils.data.DataLoader(train,batch_size=BATCHSIZE,drop_last=True)
test_loader = torch.utils.data.DataLoader(test,batch_size=100)

# x = np.random.rand(len(train_loader)*BATCHSIZE,784)
# y = np.random.randint(0,9,len(train_loader)*100)

class Net(nn.Module):
    def __init__(self, input_features, classes):
        super(Net, self).__init__()
        self.expansion = 2
        self.fc1 = nn.Linear(input_features, self.expansion*input_features)  # 5*5 from image dimension
        self.fc2 = nn.Linear(self.expansion*input_features, classes)
        self.relu = nn.ReLU(inplace=True)
        self.net = nn.Sequential(self.fc1,self.relu,self.fc2)
    def forward(self,x):
        return self.net(x)

model = Net(784,10)
def configure_loss_function():
    return torch.nn.CrossEntropyLoss()


def configure_optimizer(model):
    return torch.optim.Adam(model.parameters())


def full_gd(model, criterion, optimizer, loader, val_loader, n_epochs=1):
    train_losses = []
    test_losses = []
    for epoch in range(n_epochs):
        for it, (X, y) in enumerate(loader):

            outputs = model(X.view(BATCHSIZE,-1))
            loss = criterion(outputs, y.long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # outputs_test = model(X_test)
            # loss_test = criterion(outputs_test, y_test)

            train_losses.append(loss.item())
            # test_losses.append(loss_test.item())

            if (it + 1) % 1000 == 0:
                print(
                    f'In this epoch {epoch * it + 1}/{len(loader) * n_epochs}, Training loss: {loss.item():.4f}, ')
    total = 0
    correct = 0
    with torch.no_grad():
        for it, (X, y) in enumerate(val_loader):
            outputs_test = model(X.view(BATCHSIZE,-1))
            loss_test = criterion(outputs_test, y.long())

            test_losses.append(loss_test.item())
            _, predicted = torch.max(outputs_test.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

        print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')
        # if (it + 1) % 10 == 0:
        #     print(
        #         f'In this epoch { it + 1}/{len(val_loader) }, test loss: {loss_test.item():.4f}, ')
    return train_losses, test_losses


criterion = configure_loss_function()
optimizer = configure_optimizer(model)
train_losses, test_losses = full_gd(model, criterion, optimizer, train_loader, test_loader)
plt.subplot(211)
plt.plot(train_losses, label='train loss')
plt.subplot(212)
plt.plot(test_losses, label='test loss')
plt.legend()
plt.show()
