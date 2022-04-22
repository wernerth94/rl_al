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

def avrg(curve, window):
    end = curve.shape[1] # min(c.BUDGET, curve.shape[1])
    stdCurve = curve[1, :end]; curve = curve[0, :end]
    avrgCurve = []
    for i in range(1, len(curve)):
        avrgCurve.append(np.mean(curve[max(0, i - window):i]))
    return np.array(avrgCurve), stdCurve[1:]


def plot(curve, color, displayName, window=5):
    if len(curve) > 2:
        curve = np.concatenate( [np.expand_dims(np.mean(curve, axis=0), axis=0),
                                 np.expand_dims(np.std(curve, axis=0), axis=0)],  axis=0)
    avrgCurve, stdCurve = avrg(curve, window)

    improvement = round(avrgCurve[-1] - avrgCurve[0], 3)
    auc = round(sum(avrgCurve) / len(avrgCurve), 3)
    fullName = f"{displayName} improv. {improvement} auc {auc}"
    x = np.arange(len(avrgCurve))
    plt.fill_between(x, avrgCurve-stdCurve, avrgCurve+stdCurve, alpha=0.3, facecolor=color)
    plt.plot(x, avrgCurve, label=fullName, linewidth=LINE_WIDTH, c=color)


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

# plot(np.load('../baselines/cifar10_custom/random.npy'), 'gray', displayName='random', window=5)
# plot(np.load('../baselines/cifar10_custom/bvssb.npy'), 'navy', displayName='BvsSB', window=5)
# plot(np.load('../baselines/cifar10_custom/agent_b2000_s20.npy'), 'red', displayName='Agent', window=5)
plot(np.load('../baselines/mock/random.npy'), 'gray', displayName='random', window=5)
plot(np.load('../baselines/mock/bvssb.npy'), 'navy', displayName='BvsSB', window=5)
#plot(np.load('../baselines/cifar10_custom/agent_b2000_s20.npy'), 'red', displayName='Agent', window=5)

# plot(np.load('../baselines/bvssb_b5000_s10.npy'), 'red', displayName='BvsSB_10', window=5)
# plot(np.load('../baselines/bvssb_b5000_s100.npy'), 'orange', displayName='BvsSB_100', window=5)
# plot(np.load('../baselines/bvssb_b5000_s10000.npy'), 'blue', displayName='BvsSB_10000', window=5)
# plot(np.load('../baselines/random_b5000_s10.npy'), 'black', displayName='random', window=5)

plt.legend(fontsize='x-small')
plt.savefig('plot_'+c.MODEL_NAME+'.png')
plt.show()