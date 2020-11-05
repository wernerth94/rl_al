import matplotlib.pyplot as plt
import numpy as np
import os

LINE_WIDTH = 2
plt.figure(dpi=200)
#plt.title('Experiment 3 Comparison of the Agents')
plt.ylabel('F1-Score')
plt.xlabel('Datapoints')
#plt.ylim(0.5, 1)
plt.grid()

def avrg(file, window):
    curve = np.load(os.path.join(folder, file))
    stdCurve = curve[1]; curve = curve[0]
    avrgCurve = []
    for i in range(1, len(curve)):
        avrgCurve.append(np.mean(curve[max(0, i - window):i]))
    return np.array(avrgCurve), stdCurve[1:]


def plot(name, color, window=5, displayName=None):
    if displayName is None:
        displayName = name
    avrgCurve, stdCurve = avrg(name+'_f1.npy', window)
    x = np.arange(len(avrgCurve))
    plt.fill_between(x, avrgCurve-stdCurve, avrgCurve+stdCurve, alpha=0.15, facecolor=color)
    plt.plot(x, avrgCurve, label=displayName, linewidth=LINE_WIDTH, c=color)


folder = '..'

plot('baselines/random_mnist', 'black', displayName='random', window=1)
plot('baselines/BvsSB_mnist', 'blue', displayName='BvsSB', window=1)
#plot('outDDQN_IRIS_CONV/DDQN_IRIS_CONV', 'red', displayName='ddqn', window=1)
#plot('output/DDQN_Iris', 'red', displayName='DDQN')

suffix = input("suffix?")
if suffix != '' and not suffix.startswith('_'):
    suffix = '_'+suffix

plt.legend(fontsize='x-small')
plt.savefig('plot'+suffix+'.png')
plt.show()