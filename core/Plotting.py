import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
import os
import Misc
import config.mnistConfig as c

cwd = os.getcwd()
if cwd.endswith('core'):
    os.chdir('..')

def baselineMean(curve):
    return np.mean(curve[0, c.BUDGET-1 : c.BUDGET+1])

baseline_random = None
baseline_bvssb = None
if c.DATASET == 'mnist_mobileNet':
    baseline_random = np.load('baselines/mobilenet/random.npy')
    baseline_bvssb = np.load('baselines/mobilenet/bvssb_1000.npy')
else:
    baseline_random = np.load('baselines/random.npy')
    baseline_bvssb = np.load('baselines/bvssb_1000.npy')

if baseline_random is not None: baseline_random = baselineMean(baseline_random)
if baseline_bvssb is not None: baseline_bvssb = baselineMean(baseline_bvssb)



def gameLengthToEpochCurve(ts, c):
    steps = ts['stepCurve']
    length = []
    currentStep = 0
    for s in steps:
        currentStep += s
        length.append(int(c.GL[np.clip(currentStep, 0, len(c.GL)-1)]))
    return length


lossWindow = 20
rewardWindow = 200
qWindow = 20
stepWindow = 10

def plot(trainState, config, outDir=None, showPlots=False):
    plt.clf()
    sns.set()
    fig, axes = plt.subplots(2, 2, figsize=(15, 7))
    ax1 = axes[0,0]; ax2 = axes[0,1]; ax3 = axes[1,0]; ax4 = axes[1,1]

    ##########################################
    # Top Left
    # loss
    avrgCurve = Misc.avrg(trainState['lossCurve'], lossWindow)
    mean = max(1e-7, np.average(avrgCurve))
    ax1.set_ylim(0, 2 * mean)
    ax1.plot(np.arange(len(avrgCurve)), avrgCurve, label='loss')
    ax1.set_ylabel('loss')
    ax1.set_title('loss and reward')

    # rewards
    avrgCurve = Misc.avrg(trainState['rewardCurve'], rewardWindow)
    ax12 = ax1.twinx()
    ax12.plot(np.arange(len(avrgCurve)), avrgCurve, c='red', label='reward')
    ax12.set_ylabel('reward')

    loss_patch = mpatches.Patch(color='blue', label='loss')
    rew_patch = mpatches.Patch(color='red', label='reward')
    ax12.legend(fontsize='small', handles=[loss_patch, rew_patch])

    ##########################################
    # Top Right
    # Q values
    avrgCurve = Misc.avrg(trainState['qCurve'], qWindow)
    ax2.plot(np.arange(len(avrgCurve)), avrgCurve, c='purple', label='Q')
    #ax2.axhline(y=0, color='k')
    ax2.set_ylabel('Q')
    ax2.set_title('average Q value')

    # game length
    # ax22 = ax2.twinx()
    # ax22.plot(trainState['glCurve'], c='green', label='game length')
    # ax22.set_ylabel('game length')
    # ax22.set_ylim(bottom=0)

    q_patch = mpatches.Patch(color='purple', label='avrg Q')
    gl_patch = mpatches.Patch(color='green', label='game length')
    ax2.legend(fontsize='small', handles=[q_patch, gl_patch])

    ##########################################
    # Bottom Left
    ax3.plot(trainState['lrCurve'], c='green', label='learning rate')
    ax3.set_ylabel('learning rate')
    ax3.set_ylim(bottom=0, top=np.max(trainState['lrCurve'])+0.01)

    ax32 = ax3.twinx()
    ax32.plot(trainState['greedCurve'], c='red', label='greed')
    ax32.set_ylabel('greed')
    ax32.set_ylim(bottom=0.0, top=np.max(trainState['greedCurve'])+0.1)

    lr_patch = mpatches.Patch(color='green', label='learning rate')
    greed_patch = mpatches.Patch(color='red', label='greed')
    ax3.legend(fontsize='small', handles=[lr_patch, greed_patch])

    ##########################################
    # Bottom Right
    window = 100
    offset = 100
    plots = 5
    cm = plt.get_cmap('OrRd')
    alphas = np.linspace(0.65, 0.8, num=plots)
    colorIndices = np.linspace(0.2, 1, num=plots)

    if baseline_random is not None: ax4.axvline(x=baseline_random, color='k', linestyle='--', linewidth=1)
    if baseline_bvssb is not None:  ax4.axvline(x=baseline_bvssb, color='b', linestyle='--', linewidth=1)
    data = []
    for i in range(plots):
        high = len(trainState['f1Curve']) - (offset * i)
        low = max(high - window, -len(trainState['f1Curve']))
        data.append(trainState['f1Curve'][low:high])
    data.reverse()

    for i, d, a, cId in zip(np.flip(np.arange(len(data))), data, alphas, colorIndices):
        c = cm(cId)
        c = tuple(list(c[:3]) + [a])
        sns.kdeplot(d, ax=ax4, c=c, label=str(-i*offset))

    ax4.legend(fontsize='small')

    totalSteps = trainState.get('totalSteps', 0)
    etaH = trainState.get('eta', 0)
    gl = 0 #trainState['glCurve'][-1]
    lr = trainState['lrCurve'][-1]
    greed = trainState['greedCurve'][-1]
    fig.suptitle('Eta %3.1f h  Current Step %d  GameLength %d  LR: %0.4f  Greed: %0.3f'%(etaH, totalSteps, gl, lr, greed), fontsize=16)
    fig.tight_layout()
    os.makedirs(outDir, exist_ok=True)

    plt.savefig(os.path.join(outDir, 'prog.png'), dpi=200)
    if showPlots:
        plt.show()
    plt.close('all')


if __name__ == "__main__":
    from config import mnistConfig as c

    c.OUTPUT_FOLDER = 'out_MNIST'
    c.MODEL_NAME = 'MNIST'
    tS = Misc.loadTrainState(c)
    plot(tS, c, outDir=os.path.join('..', c.OUTPUT_FOLDER), showPlots=True)