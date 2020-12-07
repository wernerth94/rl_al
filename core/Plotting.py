import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
import pandas as pd
import os
import Misc

def avrg(curve, window):
    if len(curve) <= 0:
        return [0]
    if len(curve) < 2:
        return [curve[0]]
    avrgCurve = []
    for i in range(1, len(curve)):
        avrgCurve.append(np.mean(curve[max(0, i - window):i]))
    return avrgCurve


def gameLengthToEpochCurve(ts, c):
    steps = ts['stepCurve']
    length = []
    currentStep = 0
    for s in steps:
        currentStep += s
        length.append(int(c.GL[np.clip(currentStep, 0, len(c.GL)-1)]))
    return length


lossWindow = 30
rewardWindow = 30
qWindow = 30
stepWindow = 10

def plot(trainState, config, outDir=None, showPlots=False):
    plt.clf()
    sns.set()
    fig, axes = plt.subplots(2, 2, figsize=(15, 7))
    ax1 = axes[0,0]; ax2 = axes[0,1]; ax3 = axes[1,0]; ax4 = axes[1,1]

    #gameLengthCurve = trainState['stepCurve'] #gameLengthToEpochCurve(trainState, config)
    vertLines = []
    for i in range(len(trainState['glCurve'])-1):
        if trainState['glCurve'][i] != trainState['glCurve'][i+1]:
            vertLines.append(i)

    ##########################################
    # Top Left
    for l in vertLines:
        ax1.axvline(x=l, color='k', linestyle='--', linewidth=0.3)
    # loss
    avrgCurve = avrg(trainState['lossCurve'], lossWindow)
    mean = max(1e-7, np.average(avrgCurve))
    ax1.set_ylim(0, 2 * mean)
    ax1.plot(np.arange(len(avrgCurve)), avrgCurve, label='loss')
    ax1.set_ylabel('loss')
    ax1.set_title('loss and reward')

    # rewards
    avrgCurve = avrg(trainState['rewardCurve'], rewardWindow)
    ax12 = ax1.twinx()
    ax12.plot(np.arange(len(avrgCurve)), avrgCurve, c='red', label='reward')
    ax12.set_ylabel('reward')

    loss_patch = mpatches.Patch(color='blue', label='loss')
    rew_patch = mpatches.Patch(color='red', label='reward')
    ax12.legend(fontsize='small', handles=[loss_patch, rew_patch])

    ##########################################
    # Top Right
    for l in vertLines:
        ax2.axvline(x=l, color='k', linestyle='--', linewidth=0.3)
    # Q values
    avrgCurve = avrg(trainState['qCurve'], qWindow)
    ax2.plot(np.arange(len(avrgCurve)), avrgCurve, c='purple', label='Q')
    ax2.axhline(y=0, color='k')
    ax2.set_ylabel('Q')
    ax2.set_title('average Q value')

    q_patch = mpatches.Patch(color='purple', label='avrg Q')
    ax2.legend(fontsize='small', handles=[q_patch])

    ##########################################
    # Bottom Left
    # for l in vertLines:
    #     ax3.axvline(x=l, color='k', linestyle='--', linewidth=0.3)
    # game length
    #ax3.plot(np.arange(len(trainState['glCurve'])), trainState['glCurve'], c='green', label='game length')
    ax3.plot(trainState['lrCurve'], c='green', label='learning rate')
    ax3.set_ylabel('learning rate')
    ax3.set_ylim(bottom=0)

    ax32 = ax3.twinx()
    ax32.plot(trainState['greedCurve'], c='red', label='greed')
    ax32.set_ylabel('greed')
    ax32.set_ylim(bottom=0.1, top=0.9)

    lr_patch = mpatches.Patch(color='green', label='learning rate')
    greed_patch = mpatches.Patch(color='red', label='greed')
    ax3.legend(fontsize='small', handles=[lr_patch, greed_patch])

    ##########################################
    # Bottom Right
    window = 100
    offset = 20
    plots = 5
    cm = plt.get_cmap('OrRd')
    alphas = np.linspace(0.0, 0.99, num=plots)
    colorIndices = np.linspace(0, 1, num=plots)
    #data = sns.load_dataset("penguins")

    data = []
    for i in range(plots):
        high = len(trainState['rewardCurve']) - (offset * i)
        low = max(high - window, -len(trainState['rewardCurve']))
        data.append(trainState['rewardCurve'][low:high])
    data.reverse()
    #df = pd.DataFrame(data=np.array(data).T)
    for d, a, cId in zip(data, alphas, colorIndices):
        c = cm(cId)
        c = tuple(list(c[:3]) + [a])
        sns.kdeplot(d, ax=ax4, c=c)


    totalSteps = trainState.get('totalSteps', 0)
    etaH = trainState.get('eta', 0)
    gl = config.GL[np.clip(totalSteps, 0, len(config.GL)-1)]
    lr = config.LR[np.clip(totalSteps, 0, len(config.LR)-1)]
    greed = config.GREED[np.clip(totalSteps, 0, len(config.GREED)-1)]
    fig.suptitle('Eta %3.1f h  Current Step %d  GameLength %d  LR: %0.4f  Greed: %0.3f'%(etaH, totalSteps, gl, lr, greed), fontsize=16)
    fig.tight_layout()
    os.makedirs(outDir, exist_ok=True)

    if showPlots:
        plt.show()
    plt.savefig(os.path.join(outDir, 'prog.png'), dpi=200)
    plt.close('all')


if __name__ == "__main__":
    import batchConfig as c
    tS = Misc.loadTrainState(c, path_prefix='..')
    plot(tS, c, outDir=os.path.join('..', c.OUTPUT_FOLDER), showPlots=True)