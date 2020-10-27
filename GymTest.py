import numpy as np
import os
from core.Agent import DDQN
from core.Memory import Memory
import matplotlib.pyplot as plt
import config as c
import Misc
import gym, cv2
import rank_based

avrgWindow = 100
envName = 'FrozenLake-v0'
modelFile = "output/"+envName+".h5"
warmup     = 1000
totalSteps = 100000
imgShape = (84, 84, 1)

env = gym.make(envName, is_slippery=False)
#env.stateSpace = list(imgShape)
env.stateSpace = env.observation_space.n
env.actionSpace = env.action_space.n

agent = DDQN(env, gamma=0.99, lr=0.0001, fromCheckpoints=modelFile)

conf = {'size': 10000,
        'learn_start': warmup,
        'total_step': totalSteps,
        'batch_size': c.BATCH_SIZE,
        'partition_num': 100}
#experience = rank_based.Experience(conf)
memory = Memory(env, maxLength=30000)


def fixState(s):
    # frame = s.astype(np.uint8)  # cv2 requires np.uint8, other dtypes will not work
    # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    # frame = frame[34:34 + 160, :160]  # crop image
    # frame = cv2.resize(frame, imgShape[:2], interpolation=cv2.INTER_NEAREST)
    # return frame.reshape(imgShape)
    return (s.reshape(1, -1))
    #return labelToOneHot(np.array([s]), 16)

def eval(render=False):
    rew = 0
    done = False
    state = env.reset()
    state = fixState(state)
    while not done:
        if render:
            env.render()
        Q, a = agent.predict(state, greedParameter=0)
        statePrime, reward, done, _ = env.step(int(a[0]))
        statePrime = fixState(statePrime)
        rew += reward
        state = statePrime
    return rew

# env.is_slippery=False
# while True:
#     eval(True)

Misc.createStopSwitch()

trainState = Misc.loadTrainState(envName)
if trainState is not None:
    steps, greed, epoch = trainState['steps'], trainState['greed'], trainState['epoch']
    rewards, losses, evalRewards = trainState['rewards'], trainState['losses'], trainState['evalRewards']
else:
    steps = 0
    greed = 2
    epoch = 0
    rewards, losses, evalRewards = [],[],[]

while steps < totalSteps:
    epoch += 1
    epochSteps = 0
    state = fixState(env.reset())
    QEpoch = np.zeros(env.actionSpace)
    epochLoss = 0
    epochReward = 0
    done = False
    while not done:
        steps += 1
        epochSteps += 1
        Q, a = agent.predict(state)#np.expand_dims(state, 0))
        statePrime, reward, done, _ = env.step(int(a[0][0]))
        statePrime = fixState(statePrime)

        QEpoch += np.abs(Q[0])
        epochReward += reward

        #experience.store((state, a, reward, statePrime, done))
        memory.addMemory(state, a, reward, statePrime, done)

        if steps % 5000 == 0:
            agent.copyWeights()
        if steps > warmup and steps % 5000 == 0:
            greed = max(greed * 0.8, 0.1)
            agent.model1.save_weights(modelFile, overwrite=True, save_format="h5")

        if steps > warmup and steps % 4 == 0:
            #experience.rebalance()
            #memSample, w, e_id = experience.sample(step)
            memSample = memory.sampleMemory(c.BATCH_SIZE)
            epochLoss += agent.fit(memSample)

        state = statePrime

    evalRewards.append(eval())
    rewards.append(epochReward)
    losses.append(epochLoss)
    if epoch % 10 == 0:
        qAvrg = np.average(QEpoch / epochSteps)
        avrgR = np.average(rewards[max(-avrgWindow, -len(rewards)):])
        avrgER = np.average(evalRewards[max(-avrgWindow, -len(evalRewards)):])
        avrgL = np.average(losses[max(-avrgWindow, -len(losses)):])
        print('epoch %d  steps %d\t avrg reward %1.2f \t avrg loss %1.4f \t '
              'greed %1.2f \t avrg Q %1.3f \t evalReward %1.2f'%(epoch, steps,
               avrgR, avrgL, greed, qAvrg, avrgER))

    if not os.path.exists("stopSwitch"):
        break


agent.model1.save_weights(modelFile, overwrite=True, save_format="h5")
trainState = {
    'greed': greed,
    'rewards': rewards,
    'losses': losses,
    'evalRewards': evalRewards,
    'epoch': epoch,
    'steps': steps }
Misc.saveTrainState(envName, trainState)

plt.plot(range(len(rewards)), rewards, c='red')
plt.title("Reward")
plt.show()
plt.plot(range(len(losses)), losses, c='dodgerblue')
plt.title("Loss")
plt.show()


while True:
    eval(render=True)
    input('')