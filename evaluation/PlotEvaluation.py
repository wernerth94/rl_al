import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

from config import batchConfig as c

LINE_WIDTH = 2
plt.figure(dpi=200)
#plt.title('Experiment 3 Comparison of the Agents')
plt.ylabel('F1-Score')
plt.xlabel('Datapoints')
#plt.ylim(0.5, 1)
plt.grid()

def avrg(curve, window):
    end = curve.shape[1] # min(c.BUDGET, curve.shape[1])
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


def collect(folder, maskingThreshold=0.0, curvesFolder='curves'):
    folder = os.path.join(folder, curvesFolder)
    curves = None
    for file in os.listdir(folder):
        if not os.path.isdir(os.path.join(folder, file)):
            curve = np.load(os.path.join(folder, file))
            mask = list(np.mean(curve[:, -100:], axis=1) > maskingThreshold)
            curve = curve[mask]
            if curves is None:
                curves = curve
            else:
                curves = np.concatenate([curves, curve], axis=0)
    curves = np.array(curves)
    result = [np.mean(curves, axis=0), np.std(curves, axis=0)]
    return np.array(result)

folder = '..'
sns.set()


plot(np.load(os.path.join(folder, 'baselines/emb_random_1000.npy')), 'gray', displayName='emb_random', window=1)
plot(np.load(os.path.join(folder, 'baselines/emb_bvssb_1000.npy')), 'navy', displayName='emb_BvsSB', window=1)
plot(np.load(os.path.join(folder, 'baselines/emb_entropy_1000.npy')), 'darkgreen', displayName='emb_Entropy', window=1)

plot(np.load(os.path.join(folder, 'baselines/random_1000.npy')), 'black', displayName='random', window=1)
plot(np.load(os.path.join(folder, 'baselines/bvssb_1000.npy')), 'blue', displayName='BvsSB', window=1)
plot(np.load(os.path.join(folder, 'baselines/entropy_1000.npy')), 'green', displayName='Entropy', window=1)

plot(collect(os.path.join(folder, 'out_backup_PROC_MNIST_BATCH_RS'), curvesFolder='curves', maskingThreshold=0.0), 'red', displayName='ddqn', window=1)
plot(collect(os.path.join(folder, 'out_backup_PROC_MNIST_BATCH_RS'), curvesFolder='curves_0.1', maskingThreshold=0.0), 'orange', displayName='ddqn', window=1)
plot(collect(os.path.join(folder, 'out_backup_PROC_MNIST_BATCH_RS'), curvesFolder='curves_0.01', maskingThreshold=0.0), 'pink', displayName='ddqn', window=1)
# plot(collect(os.path.join(folder, 'goodRuns/MNIST_BATCH_2')), 'green', displayName='ddqn_2', window=1)

plt.ylim(0.5, 1)
plt.yticks(np.arange(0.5, 1, 0.1))
plt.axhline(y=0.965, label='upper bound', c='black', linestyle='dashed')
plt.legend(fontsize='x-small')
plt.savefig('plot_'+c.MODEL_NAME+'.png')
plt.show()