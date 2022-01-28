import numpy as np
import gc
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def resetALPool(dataset, init_points_per_class=5):
    x_train, y_train, x_test, y_test = dataset
    nClasses = y_train.shape[1]
    xLabeled, yLabeled = [], []

    ids = np.arange(x_train.shape[0], dtype=int)
    np.random.shuffle(ids)
    perClassIntances = [0 for _ in range(nClasses)]
    usedIds = []
    for i in ids:
        label = np.argmax(y_train[i])
        if perClassIntances[label] < init_points_per_class:
            xLabeled.append(i)
            yLabeled.append(i)
            # xLabeled.append(x_train[i])
            # yLabeled.append(y_train[i])
            usedIds.append(i)
            perClassIntances[label] += 1
        if sum(perClassIntances) >= init_points_per_class * nClasses:
            break
    unusedIds = [i for i in np.arange(x_train.shape[0]) if i not in usedIds]
    xLabeled = x_train[xLabeled]
    yLabeled = y_train[yLabeled]
    xUnlabeled = x_train[unusedIds]
    yUnlabeled = y_train[unusedIds]

    xLabeled = xLabeled.to(device)
    yLabeled = yLabeled.to(device)
    xUnlabeled = xUnlabeled.to(device)
    yUnlabeled = yUnlabeled.to(device)
    gc.collect()
    return xLabeled, yLabeled, xUnlabeled, yUnlabeled, perClassIntances


def addDatapointToPool(xLabeled, yLabeled, xUnlabeled, yUnlabeled, perClassIntances, dpId: int):
    # add images
    perClassIntances[int(torch.argmax(yUnlabeled[dpId]).cpu())] += 1  # keep track of the added images
    xLabeled = torch.cat([ xLabeled, xUnlabeled[dpId:dpId + 1] ], dim=0)
    yLabeled = torch.cat([ yLabeled, yUnlabeled[dpId:dpId + 1] ], dim=0)
    xUnlabeled = torch.cat([ xUnlabeled[:dpId], xUnlabeled[dpId+1:] ], dim=0)
    yUnlabeled = torch.cat([ yUnlabeled[:dpId], yUnlabeled[dpId+1:] ], dim=0)
    return xLabeled, yLabeled, xUnlabeled, yUnlabeled, perClassIntances


def addPoolInformation(xUnlabeled, xLabeled, stateIds, alFeatures):
    presentedImg = xUnlabeled[stateIds]
    labeledPool = np.mean(xLabeled, axis=0)
    poolFeat = np.tile(labeledPool, (len(alFeatures), 1))
    return np.concatenate([alFeatures, presentedImg, poolFeat], axis=1)


def sampleNewBatch(xUnlabeled, sampleSize):
    return np.random.choice(len(xUnlabeled), sampleSize)

