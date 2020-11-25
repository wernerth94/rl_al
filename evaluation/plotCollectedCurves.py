import matplotlib.pyplot as plt
import numpy as np
import os

import convConfig as c

LINE_WIDTH = 2
plt.figure(dpi=200)
plt.ylabel('F1-Score')
plt.xlabel('Datapoints')
plt.grid()

def avrg(curve, window):
    avrgCurve = []
    for i in range(1, len(curve)):
        avrgCurve.append(np.mean(curve[max(0, i - window):i]))
    return np.array(avrgCurve)


def plot(collection, window=5):
    means = list()
    for curves in collection:
        means.append(np.mean(curves, axis=0))
        plt.clf()
        for curve in curves:
            avrgCurve = avrg(curve, window)
            x = np.arange(len(avrgCurve))
            plt.plot(x, avrgCurve, linewidth=LINE_WIDTH)
        plt.grid()
        plt.show()

    plt.clf()
    for m in means:
        plt.plot(m)
    plt.grid()
    plt.show()


def collect(folder):
    folder = os.path.join(folder, 'curves')
    collections = list()
    for file in os.listdir(folder):
        curves = np.load(os.path.join(folder, file))
        collections.append(curves)

    return collections

folder = '..'

plot(collect(os.path.join(folder, 'outDDQN_MNIST_BATCH')), window=1)
#plot('output/DDQN_Iris', 'red', displayName='DDQN')

plt.legend(fontsize='x-small')
plt.show()