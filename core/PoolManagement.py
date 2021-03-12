import numpy as np
import mnistConfig as c
import gc

def resetALPool(dataset):
    x_train, y_train, x_test, y_test = dataset
    nClasses = y_train.shape[1]
    xLabeled, yLabeled = [], []

    ids = np.arange(x_train.shape[0], dtype=int)
    np.random.shuffle(ids)
    perClassIntances = [0 for _ in range(nClasses)]
    usedIds = []
    for i in ids:
        label = np.argmax(y_train[i])
        if perClassIntances[label] < c.INIT_POINTS_PER_CLASS:
            xLabeled.append(x_train[i])
            yLabeled.append(y_train[i])
            usedIds.append(i)
            perClassIntances[label] += 1
        if sum(perClassIntances) >= c.INIT_POINTS_PER_CLASS * nClasses:
            break
    unusedIds = [i for i in np.arange(x_train.shape[0]) if i not in usedIds]
    xLabeled = np.array(xLabeled)
    yLabeled = np.array(yLabeled)
    xUnlabeled = np.array(x_train[unusedIds])
    yUnlabeled = np.array(y_train[unusedIds])
    gc.collect()
    return xLabeled, yLabeled, xUnlabeled, yUnlabeled, perClassIntances


def addDatapointToPool(xLabeled, yLabeled, xUnlabeled, yUnlabeled, perClassIntances, dpId: int):
    # add images
    perClassIntances[int(np.argmax(yUnlabeled[dpId]))] += 1  # keep track of the added images
    xLabeled = np.append(xLabeled, xUnlabeled[dpId:dpId + 1], axis=0)
    yLabeled = np.append(yLabeled, yUnlabeled[dpId:dpId + 1], axis=0)
    xUnlabeled = np.delete(xUnlabeled, dpId, axis=0)
    yUnlabeled = np.delete(yUnlabeled, dpId, axis=0)
    return xLabeled, yLabeled, xUnlabeled, yUnlabeled, perClassIntances


def addPoolInformation(xUnlabeled, xLabeled, stateIds, alFeatures):
    presentedImg = xUnlabeled[stateIds]
    labeledPool = np.mean(xLabeled, axis=0)
    poolFeat = np.tile(labeledPool, (len(alFeatures), 1))
    return np.concatenate([alFeatures, presentedImg, poolFeat], axis=1)


def sampleNewBatch(xUnlabeled):
    return np.random.choice(xUnlabeled.shape[0], c.SAMPLE_SIZE)

