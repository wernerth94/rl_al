import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
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

def plot(trainState, config, outDir=None, compCurves=[], showPlots=False):
    plt.clf()
    sns.set()
    fig, axes = plt.subplots(2, 2, figsize=(15, 7))
    ax1 = axes[0,0]; ax2 = axes[0,1]; ax3 = axes[1,0]; ax4 = axes[1,1]

    gameLengthCurve = trainState['stepCurve'] #gameLengthToEpochCurve(trainState, config)
    vertLines = []
    for i in range(len(gameLengthCurve)-1):
        if gameLengthCurve[i] != gameLengthCurve[i+1]:
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
    for l in vertLines:
        ax3.axvline(x=l, color='k', linestyle='--', linewidth=0.3)
    # game length
    ax3.plot(np.arange(len(gameLengthCurve)), gameLengthCurve, c='green', label='game length')

    gl_patch = mpatches.Patch(color='green', label='game length')
    ax3.legend(fontsize='small', handles=[gl_patch])

    ##########################################
    # Bottom Right
    if len(compCurves) > 0:
        for curve in compCurves:
            mean, std = curve
            baseValue = np.full_like(mean, mean[0])
            mean -= baseValue
            x = np.arange(len(mean))
            #adjustedMean = mean - c.LABEL_COST * x
            ax4.fill_between(x, mean-std, mean+std, alpha=0.15)
            ax4.plot(x, mean, linewidth=1)

        compX = np.arange(1, 11)*100
        MnS = []
        for i in compX:
            points = []
            for j in range(len(trainState['stepCurve'])):
                if i == trainState['stepCurve'][j]:
                    points.append(trainState['rewardCurve'][j] + i * c.LABEL_COST)
                if j > i: break
            if len(points) > 0:
                MnS.append([np.mean(points), np.std(points)])

        MnS = np.array(MnS)
        ax4.errorbar(compX[:len(MnS)], MnS[:,0], MnS[:,1], linestyle=None)



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
    rndm = np.load(os.path.join('../baselines', 'random_mnist_f1.npy'))
    BvsSB = np.load(os.path.join('../baselines', 'BvsSB_mnist_f1.npy'))
    plot(tS, c, outDir=os.path.join('..', c.OUTPUT_FOLDER), compCurves=[], showPlots=True)