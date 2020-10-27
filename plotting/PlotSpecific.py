import matplotlib.pyplot as plt
import numpy as np
import os

LINE_WIDTH = 2
plt.figure(dpi=200)
#plt.title('Experiment 3 Comparison of the Agents')
plt.ylabel('F1-Score')
plt.xlabel('Images')
#plt.ylim(0.5, 1)
plt.grid()

def avrg(file, window):
    curve = np.load(os.path.join(folder, file))
    avrgCurve = []
    for i in range(1, len(curve)):
        avrgCurve.append(np.mean(curve[max(0, i - window):i]))
    return avrgCurve

def plot(name, color, window=5, displayName=None):
    if displayName is None:
        displayName = name
    avrgCurve = avrg(name+'_f1.npy', window)
    plt.plot(np.arange(len(avrgCurve)), avrgCurve, label=displayName, linewidth=LINE_WIDTH, c=color)

folder = '..'

plot('output/random', 'black', displayName='random')
#plot('output/DDQN_Iris', 'red', displayName='DDQN')

plt.legend(fontsize='x-small')
plt.savefig('plot.png')
plt.show()