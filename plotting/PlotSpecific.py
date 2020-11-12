import matplotlib.pyplot as plt
import numpy as np
import os

import convConfig as c

LINE_WIDTH = 2
plt.figure(dpi=200)
#plt.title('Experiment 3 Comparison of the Agents')
plt.ylabel('F1-Score')
plt.xlabel('Datapoints')
#plt.ylim(0.5, 1)
plt.grid()

def avrg(curve, window):
    end = min(c.BUDGET, curve.shape[1])
    stdCurve = curve[1, :end]; curve = curve[0, :end]
    avrgCurve = []
    for i in range(1, len(curve)):
        avrgCurve.append(np.mean(curve[max(0, i - window):i]))
    return np.array(avrgCurve), stdCurve[1:]


def plot(curve, color, displayName, window=5):
    avrgCurve, stdCurve = avrg(curve, window)
    x = np.arange(len(avrgCurve))
    plt.fill_between(x, avrgCurve-stdCurve, avrgCurve+stdCurve, alpha=0.15, facecolor=color)
    plt.plot(x, avrgCurve, label=displayName, linewidth=LINE_WIDTH, c=color)


def collect(folder):
    folder = os.path.join(folder, 'curves')
    curves = None
    for file in os.listdir(folder):
        curve = np.load(os.path.join(folder, file))
        if not curves:
            curves = curve
        else:
            curves = np.concatenate([curves, curve], axis=0)
    curves = np.array(curves)
    result = [np.mean(curves, axis=0), np.std(curves, axis=0)]
    # plt.hist(result[1])
    # plt.show()
    return np.array(result)

folder = '..'

plot(np.load(os.path.join(folder, 'baselines/random_mnist_f1.npy')), 'black', displayName='random', window=1)
plot(np.load(os.path.join(folder, 'baselines/BvsSB_mnist_f1.npy')), 'blue', displayName='BvsSB', window=1)
plot(collect(os.path.join(folder, 'outDDQN_MNIST_CONV')), 'red', displayName='ddqn', window=1)
#plot('output/DDQN_Iris', 'red', displayName='DDQN')

suffix = input("suffix?")
if suffix != '' and not suffix.startswith('_'):
    suffix = '_'+suffix

plt.legend(fontsize='x-small')
plt.savefig('plot'+suffix+'.png')
plt.show()