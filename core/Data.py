import numpy as np
import sys, os
import tensorflow.keras as keras
from tensorflow.keras.utils import to_categorical

CLUSTER = False
if sys.prefix.startswith('/home/werner/miniconda3'):
    CLUSTER = True


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


def load_mnist_mobilenet(numTest=2000, prefix=''):
    with np.load(os.path.join(prefix, '../datasets/mnist_mobileNetV2.npz'), allow_pickle=True) as f:
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'][:numTest], f['y_test'][:numTest]

    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    return (x_train, y_train, x_test, y_test)


def loadMNIST(color=False, numTest=2000, prefix=''):
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




def loadCifar(numTest=1000, numLabels=10):
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    y_train = to_categorical(y_train, numLabels)
    x_train = np.array(x_train, dtype=float) / 255
    y_test = to_categorical(y_test[:numTest], numLabels)
    x_test = np.array(x_test, dtype=float)[:numTest] / 255
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    return (x_train, y_train, x_test, y_test)


def load_cifar10_mobilenet(numTest=1000, prefix=''):
    with np.load(os.path.join(prefix, '../datasets/cifar10_mobileNetV2.npz'), allow_pickle=True) as f:
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'][:numTest], f['y_test'][:numTest]

    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    return (x_train, y_train, x_test, y_test)
