import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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
        length.append(c.GL[np.clip(currentStep, 0, len(c.GL)-1)])
    return length


lossWindow = 30
rewardWindow = 30
qWindow = 30
stepWindow = 10

def plot(trainState, config, outDir=None, showPlots=False):
    plt.clf()
    fig, axes = plt.subplots(2, 2, figsize=(15, 7))
    ax1 = axes[0,0]; ax2 = axes[0,1]; ax3 = axes[1,0]; ax4 = axes[1,1]

    ##########################################
    # Top Left
    # loss
    avrgCurve = avrg(trainState['lossCurve'], lossWindow)
    mean = max(1e-7, np.average(avrgCurve))
    ax1.axhline(y=0, color='k')
    ax1.set_ylim(0, 2 * mean)
    ax1.plot(np.arange(len(avrgCurve)), avrgCurve, label='loss')
    ax1.set_ylabel('loss')
    #ax1.legend(fontsize='small')
    ax1.set_title('loss and reward')

    # rewards
    avrgCurve = avrg(trainState['rewardCurve'], rewardWindow)
    ax12 = ax1.twinx()
    ax12.plot(np.arange(len(avrgCurve)), avrgCurve, c='red', label='reward')
    ax12.axhline(y=0, color='k', linestyle='--', linewidth=1)
    ax12.set_ylabel('reward')

    loss_patch = mpatches.Patch(color='blue', label='loss')
    rew_patch = mpatches.Patch(color='red', label='reward')
    ax12.legend(fontsize='small', handles=[loss_patch, rew_patch])

    ##########################################
    # Top Right
    # Q values
    avrgCurve = avrg(trainState['qCurve'], qWindow)
    ax2.plot(np.arange(len(avrgCurve)), avrgCurve, c='purple', label='Q')
    ax2.set_ylabel('Q')
    ax2.axhline(y=0, color='k', linestyle='--', linewidth=1)
    ax2.set_title('average Q value')

    q_patch = mpatches.Patch(color='purple', label='avrg Q')
    ax2.legend(fontsize='small', handles=[q_patch])

    ##########################################
    # Bottom Left
    # game length
    ax3.plot(np.arange(len(trainState['stepCurve'])), gameLengthToEpochCurve(trainState, config), c='green', label='game length')
    #ax3.set_ylabel('game length')
    avrgCurve = avrg(trainState['stepCurve'], stepWindow)
    ax3.plot(np.arange(len(avrgCurve)), avrgCurve, c='yellow', label='steps')

    # step curve
    #avrgCurve = avrg(trainState['stepCurve'], stepWindow)
    #mx = max(1e-7, np.max(avrgCurve))
    #ax32 = ax3.twinx()
    #ax32.plot(np.arange(len(avrgCurve)), avrgCurve, c='yellow', label='steps')
    #ax32.set_ylim(0, mx)
    #ax32.set_ylabel('steps')
    #ax32.axhline(y=0, color='k', linestyle='--', linewidth=1)
    #ax3.set_title('game length and steps per epoch')

    steps_patch = mpatches.Patch(color='yellow', label='steps')
    gl_patch = mpatches.Patch(color='green', label='game length')
    ax3.legend(fontsize='small', handles=[gl_patch, steps_patch])

    ##########################################
    # Bottom Right

    fig.tight_layout()
    if showPlots:
        plt.show()
    else:
        os.makedirs(outDir, exist_ok=True)
        plt.savefig(os.path.join(outDir, 'prog.png'), dpi=200)
    plt.close('all')



if __name__ == "__main__":
    import config as c
    tS = Misc.loadTrainState(c, path_prefix='..')
    plot(tS, c, outDir='../output', showPlots=False)