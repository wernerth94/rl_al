import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns

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


def plot(collection, labels, window=5, thresh=0.0):
    means = list()
    sns.set()  # plt.grid()
    fig, axes = plt.subplots(1, len(collection), figsize=(6*len(collections), 4))
    if len(collection) < 2: axes = [axes]
    for curves, ax, label in zip(collection, axes, labels):
        mask = list(np.mean(curves[:, -100:], axis=1) > thresh)
        curves = curves[mask]
        means.append(np.mean(curves, axis=0))
        for curve in curves:
            avrgCurve = avrg(curve, window)
            x = np.arange(len(avrgCurve))
            ax.plot(x, avrgCurve, linewidth=LINE_WIDTH)
        ax.set_xlabel(label)
        ax.set_ylim(0.6, 1)
    plt.show()

    print('means')
    plt.clf()
    sns.set() #plt.grid()
    for m, l in zip(means, labels):
        plt.plot(m, label=l)
    plt.ylim(0.6, 1)
    plt.legend()
    plt.show()


def collect(folder, curvesFolder='curves'):
    folder = os.path.join(folder, curvesFolder)
    collections = list()
    labels = list()
    for file in os.listdir(folder):
        if not os.path.isdir(os.path.join(folder, file)):
            curves = np.load(os.path.join(folder, file))
            collections.append(curves)
            labels.append(file)

    return collections, labels

folder = '..'

collections, labels = collect(os.path.join(folder, 'out_PROC_MNIST'), curvesFolder='curves')
plot(collections, labels, window=1)
#plot('output/DDQN_Iris', 'red', displayName='DDQN')