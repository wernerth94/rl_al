import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

from config import cifarConfig as c

LINE_WIDTH = 2
plt.figure(dpi=200)
#plt.title('Experiment 3 Comparison of the Agents')
plt.ylabel('F1-Score')
plt.xlabel('Datapoints')
#plt.ylim(0.5, 1)
plt.grid()

def avrg(curve, weight):
    end = curve.shape[1] # min(c.BUDGET, curve.shape[1])
    stdCurve = curve[1, :end]; curve = curve[0, :end]
    avrgCurve = []
    moving_avrg = curve[0]
    for i in range(1, len(curve)):
        moving_avrg = weight * moving_avrg + (1 - weight) * curve[i]
        avrgCurve.append(moving_avrg)
    return np.array(avrgCurve), stdCurve[1:]


def plot(curve_file, color, displayName, weight=0.9, alpha=0.8):
    curve = np.load(curve_file)
    if len(curve) > 2:
        # of the file contains many runs instead of mean and std
        curve = np.concatenate( [np.expand_dims(np.mean(curve, axis=0), axis=0),
                                 np.expand_dims(np.std(curve, axis=0), axis=0)],  axis=0)
    auc = round(sum(curve[0]) / len(curve[0]), 3)
    avrgCurve, stdCurve = avrg(curve, weight)

    improvement = round(avrgCurve[-1] - avrgCurve[0], 3)
    fullName = f"{displayName} improv. {improvement} auc {auc}"
    x = np.arange(len(avrgCurve))
    plt.fill_between(x, avrgCurve-stdCurve, avrgCurve+stdCurve, alpha=0.5, facecolor=color)
    plt.plot(x, avrgCurve, label=fullName, linewidth=LINE_WIDTH, c=color, alpha=alpha)


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

plot('../baselines/cifar10_custom/random.npy', 'gray', displayName='random', weight=0.8)
plot('../baselines/cifar10_custom/bvssb.npy', 'navy', displayName='BvsSB', weight=0.8)
plot('../baselines/cifar10_custom/ensemble.npy', 'red', displayName='Ensemble', weight=0.97)
plot('../baselines/cifar10_custom/ensemble_b2.npy', 'purple', displayName='Ensemble B2', weight=0.97)
plot('../baselines/cifar10_custom/agent_b2000_s20.npy', 'orange', displayName='Agent', weight=0.8, alpha=1.0)

# plot('../baselines/mock/random.npy', 'gray', displayName='random', weight=0.0)
# plot('../baselines/mock/entropy.npy', 'green', displayName='Entropy', weight=0.0)
# plot('../baselines/mock/bvssb.npy', 'navy', displayName='BvsSB', weight=0.0)

plt.legend(fontsize='x-small')
plt.ylim(0.55, 0.8)
plt.savefig('plot_'+c.MODEL_NAME+'.png')
plt.show()