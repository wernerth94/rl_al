import random
import Classifier
import config.cifarConfig as c
from Data import load_cifar10_mobilenet
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import bottleneck as bn

def _to_advantage(inpt):
    m = np.mean(inpt, axis=0)
    return inpt - m

def _get_state(classifier, x):
    # eps = 1e-7
    # # prediction metrics
    # pred = classifier(x).detach()
    # part = (-bn.partition(-pred, 4, axis=1))[:, :4]  # collects the two highest entries
    # struct = np.sort(part, axis=1)
    # bVsSB = 1 - (struct[:, -1] - struct[:, -2])
    # bVsSB = _to_advantage(bVsSB)
    # bVsSB = torch.from_numpy(bVsSB).float()
    #
    # entropy = -np.average(pred * np.log(eps + pred) + (1 + eps - pred) * np.log(1 + eps - pred), axis=1)
    # entropy = _to_advantage(entropy)
    # entropy = torch.from_numpy(entropy).float()

    # hist_data = []
    # for m in range(x.size()[1]):
    #     hist = np.histogram(x[:, m], density=True, bins=[0.0, 0.6, 1.2, 1.8, 2.4, 3.0, 3.6, 4.2, 4.8, 5.4, 6.0])
    #     hist_data.append(hist[0])
    latent_mean = torch.mean(x, dim=0)
    #latent_mean = torch.stack([latent_mean]*len(bVsSB))
    latent_std = torch.std(x, dim=0)
    #latent_std = torch.stack([latent_std]*len(bVsSB))

    return torch.cat([latent_mean, latent_std])
    #return torch.cat([torch.unsqueeze(bVsSB, 1), torch.unsqueeze(entropy, 1), latent_mean, latent_std], dim=-1)


x_train, y_train, x_test, y_test = load_cifar10_mobilenet(return_tensors=True)
classifierFactory = Classifier.EmbeddingClassifierFactory(c.EMBEDDING_SIZE)
loss = nn.CrossEntropyLoss()

meta_x = []
meta_y = []

LEN_META = 2000
BATCH_SIZE = 64

for epoch in range(LEN_META):
    labeled_size = random.randint(100, 500)
    ids = np.random.choice(len(x_train), size=labeled_size, replace=False)
    x_labeled = x_train[ids]
    y_labeled = y_train[ids]

    model = classifierFactory()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    lastLoss = torch.inf
    for e in range(30):
        permutation = torch.randperm(len(x_labeled))
        for i in range(0, len(x_labeled), BATCH_SIZE):
            indices = permutation[i:i + BATCH_SIZE]
            batch_x, batch_y = x_labeled[indices], y_labeled[indices]
            yHat = model(batch_x)
            loss_val = loss(yHat, batch_y)
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
        # early stopping on test
        with torch.no_grad():
            yHat_test = model(x_test)
            test_loss = loss(yHat_test, y_test)
            if test_loss > lastLoss:
                break
            lastLoss = test_loss
    if epoch % 50 == 0:
        print(epoch, '/', LEN_META)
    meta_x.append(_get_state(model, x_labeled))
    meta_y.append(lastLoss)

random.shuffle(meta_x)
random.shuffle(meta_y)
cut = int(len(meta_x) * 0.8)
meta_x_train = torch.stack(meta_x[:cut])
meta_y_train = torch.stack(meta_y[:cut])
meta_x_test = torch.stack(meta_x[cut:])
meta_y_test = torch.stack(meta_y[cut:])

model = nn.Sequential(nn.Linear(1280*2, 300), nn.Tanh(),
                      nn.Linear(300, 10), nn.Tanh(),
                      nn.Linear(10, 1))
loss = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

loss_curve = []
lastLoss = torch.inf
for e in range(100):
    permutation = torch.randperm(len(meta_x_train))
    for i in range(0, len(meta_x_train), BATCH_SIZE):
        indices = permutation[i:i + BATCH_SIZE]
        batch_x, batch_y = meta_x_train[indices], meta_y_train[indices]
        yHat = model(batch_x)
        loss_val = loss(yHat.squeeze(), batch_y)
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()
    # early stopping on test
    with torch.no_grad():
        yHat_test = model(meta_x_test)
        test_loss = loss(yHat_test.squeeze(), meta_y_test)
        if test_loss > lastLoss:
            print("early stopping")
            break
        lastLoss = test_loss
    print(e, test_loss)
    loss_curve.append(test_loss)

import matplotlib.pyplot as plt
plt.plot(loss_curve)
plt.show()