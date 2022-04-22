import numpy as np
import sys, os
import torch
from scipy.io import arff
import pandas as pd
from tensorflow import keras
from torchvision import transforms
import torchvision

CLUSTER = False
if sys.prefix.startswith('/home/werner/miniconda3'):
    CLUSTER = True

def to_categorical(labels, num_classes=None):
    if not num_classes:
        num_classes = max(labels)
    return np.eye(num_classes, dtype='uint8')[labels]

def loadTrafficSigns():
    print('loading traffic signs')
    data = arff.loadarff('../datasets/traffic_signs.arff')
    df = pd.DataFrame(data[0])
    data = df.values.astype('float32')
    allIds = np.arange(len(data))
    np.random.shuffle(allIds)
    cutoff = int(len(data) * 0.8)
    trainIds, testIds = allIds[:cutoff], allIds[cutoff:]

    x, y = data[:, :-1], to_categorical(data[:, -1:])
    return (x[trainIds], y[trainIds], x[testIds], y[testIds])


def loadWine():
    x = []; y = []
    with open('../datasets/wine.data', 'r') as reader:
        while (line := reader.readline()) != '':
            split = line.split(',')
            x.append(split[:4])
            y.append(split[-1])
    x = np.array(x, dtype=float);  y = np.array(y)
    y = to_categorical(y, num_classes=10)
    idx = np.arange(len(x))
    np.random.shuffle(idx)
    trainIdx = idx[:100]
    testIdx = idx[100:]
    return (x[trainIdx], y[trainIdx], x[testIdx], y[testIdx])



def loadIRIS():
    labels = {
        'Iris-setosa': 0,
        'Iris-versicolor': 1,
        'Iris-virginica': 2
    }
    x = []; y = []
    with open('../datasets/iris.data', 'r') as reader:
        while (line := reader.readline()) != '':
            split = line.split(',')
            x.append(split[:4])
            y.append([ labels[split[-1].strip()] ])
    x = np.array(x, dtype=float); y = np.array(y)
    y = to_categorical(y, num_classes=3)
    idx = np.arange(len(x))
    np.random.shuffle(idx)
    trainIdx = idx[:100]
    testIdx = idx[100:]
    return (x[trainIdx], y[trainIdx], x[testIdx], y[testIdx])


def load_mnist_embedded(embedding, numTest=2000, prefix=''):
    if embedding == 'mnist_mobileNet':
        file = 'mnist_mobileNetV2.npz'
    elif embedding == 'mnist_embedSmall':
        file = 'mnist_embedSmall.npz'
    with np.load(os.path.join(prefix, '../datasets', file), allow_pickle=True) as f:
        x_train, y_train = f['x_train'].astype(np.float32), f['y_train'].astype(np.float32)
        x_test, y_test = f['x_test'][:numTest].astype(np.float32), f['y_test'][:numTest].astype(np.float32)

    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    return (x_train, y_train, x_test, y_test)


def loadMNIST(color=False, numTest=4000, prefix=''):
    with np.load(os.path.join(prefix, '../datasets/mnist.npz'), allow_pickle=True) as f:
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'][:numTest], f['y_test'][:numTest]

    x_train = x_train.reshape(len(x_train), 28, 28, 1)
    x_train = np.array(x_train, dtype=float) / 255
    if color:
        x_train = np.repeat(x_train, 3, axis=3)
    y_train = to_categorical(y_train, num_classes=10)

    x_test = x_test.reshape(len(x_test), 28, 28, 1)
    x_test = np.array(x_test, dtype=float)[:numTest] / 255
    if color:
        x_test = np.repeat(x_test, 3, axis=3)
    y_test = to_categorical(y_test[:numTest], num_classes=10)
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    return (x_train, y_train, x_test, y_test)


def _post_process(x_train, y_train, x_test, y_test, return_tensors=False, channelFirst=False):
    if return_tensors:
        x_train = torch.from_numpy(x_train).float()
        y_train = torch.from_numpy(y_train).float()
        x_test = torch.from_numpy(x_test).float()
        y_test = torch.from_numpy(y_test).float()
        if channelFirst:
            x_train = x_train.permute(0, 3, 1, 2)
            x_test = x_test.permute(0, 3, 1, 2)
    else:
        if channelFirst:
            x_train = np.moveaxis(x_train, -1, 1)
            x_test = np.moveaxis(x_test, -1, 1)
    return (x_train, y_train, x_test, y_test)


def load_cifar10_pytorch(numTest=1000, img_size=32, prefix="", return_tensors=True):
    trans = transforms.Compose([transforms.Resize([img_size, img_size]),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225])])
    folder = os.path.join(prefix, "../datasets")
    trainset = torchvision.datasets.CIFAR10(root=folder, train=True, download=True, transform=trans)
    testset = torchvision.datasets.CIFAR10(root=folder, train=False, download=True, transform=trans)
    return _post_process(trainset.data, to_categorical(np.array(trainset.targets), 10),
                         testset.data, to_categorical(np.array(testset.targets), 10),
                         return_tensors=return_tensors, channelFirst=True)

def load_cifar10(numTest=1000, numLabels=10, return_tensors=False, channelFirst=False):
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    y_train = to_categorical(y_train, numLabels)
    x_train = np.array(x_train, dtype=float) / 255
    y_test = to_categorical(y_test[:numTest], numLabels)
    x_test = np.array(x_test, dtype=float)[:numTest] / 255
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    return _post_process(x_train, y_train, x_test, y_test,
                         return_tensors=return_tensors, channelFirst=channelFirst)


def load_cifar10_mobilenet(numTest=1000, prefix='', return_tensors=False, channelFirst=False):
    with np.load(os.path.join(prefix, '../datasets/cifar10_mobileNetV2.npz'), allow_pickle=True) as f:
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'][:numTest], f['y_test'][:numTest]
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    return _post_process(x_train, y_train, x_test, y_test,
                         return_tensors=return_tensors, channelFirst=channelFirst)

def load_cifar10_custom(numTest=1000, prefix='', return_tensors=False, channelFirst=False):
    with np.load(os.path.join(prefix, '../datasets/cifar10_custom.npz'), allow_pickle=True) as f:
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'][:numTest], f['y_test'][:numTest]
    x_train = (x_train - np.mean(x_train, axis=0)) / np.std(x_train, axis=0)
    x_test = (x_test - np.mean(x_test, axis=0)) / np.std(x_test, axis=0)
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    return _post_process(x_train, y_train, x_test, y_test,
                         return_tensors=return_tensors, channelFirst=channelFirst)
