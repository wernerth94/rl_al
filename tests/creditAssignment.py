import Agent, Environment, Memory, Misc
import numpy as np
import matplotlib.pyplot as plt
from time import time

maxEpochs = 1000
gameLength = 500
nSteps = 3

env = Environment.CreditAssignmentGame(gameLength)
agent = Agent.DDVN(env.stateSpace, n_hidden=10, lr=0.002, gamma=1)
memory = Memory.NStepVMemory(env.stateSpace, nSteps)

startTime = time()
totalRewards, totalVStarts = [], []
try:
    for i in range(maxEpochs):
        stateBuffer = []; rewardBuffer = []
        state = env.reset()

        epochRewards = 0
        steps, done, vStart = 0, False, 0
        while not done:
            for n in range(nSteps):
                V, a = agent.predict(state, greed=0.1)
                a = a[0]
                if steps == 0:
                    vStart = V[a]

                reward, newState, done = env.step(a)
                stateBuffer.append(state[a])
                rewardBuffer.append(reward)
                epochRewards += reward

                if len(stateBuffer) >= nSteps:
                    memory.addMemory(stateBuffer.pop(0), rewardBuffer, np.mean(newState, axis=0), done)
                    rewardBuffer.pop(0)

                    memSample = memory.sampleMemory(16)
                    agent.fit(memSample)
                    if i % 10 == 0:
                        agent.copyWeights()

                if done:
                    break
                state = newState

        totalRewards.append(epochRewards)
        totalVStarts.append(vStart)

        secPerEpoch = (time() - startTime) / (i+1)
        print(i, 'rewards: %1.4f \t V: %1.4f \t eta (h): %1.1f'%(np.mean(totalRewards), np.mean(totalVStarts), secPerEpoch*(maxEpochs-i)/60/60))
except KeyboardInterrupt:
    pass

cont = True
window = 30
while cont:
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(Misc.avrg(totalVStarts, window=window), label='V', c='blue')
    ax1.plot(Misc.avrg(totalRewards, window=window), label='reward', c='red')
    fig.tight_layout()
    plt.show()
    print('window size', window)
    w = input('try with different window?  size:')
    try:
        window = int(w)
    except:
        break