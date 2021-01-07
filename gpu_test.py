import sys
import getpass
print(F"The user is: {getpass.getuser()}")
print(F"The virtualenv is: {sys.prefix}")

# path additions for the cluster
sys.path.append("core")
sys.path.append("evaluation")
sys.path.append("config")
print(F"updated path is {sys.path}")

import tensorflow as tf
import tensorflow.keras as keras
import numpy as np

import Data
import Classifier
import Environment
import Agent
import Memory
import config.batchConfig as c

nSteps = 5
dataset = Data.loadMNIST()

def printTime(totalSec):
    minutes = int(totalSec / 60)
    seconds = int(totalSec % 60)
    print('took %dm %ds' % (minutes, seconds))

def test():
    try:
        from time import  time
        startTime = time()
        env = Environment.BatchALGame(dataset=dataset, modelFunction=Classifier.DenseClassifierMNIST, config=c, verbose=0)
        memory = Memory.NStepMemory(env, nSteps, maxLength=c.MEMORY_CAP)

        agent = Agent.NStepBatchAgent(env, nSteps)

        for _ in range(5):
            stateBuffer = []
            rewardBuffer = []
            state = env.reset()
            steps, done = 0, False
            while not done:
                for n in range(nSteps):
                    Q, a = agent.predict(state, greedParameter=0.5)
                    a = a[0]

                    statePrime, reward, done, _ = env.step(a)
                    stateBuffer.append(state[a])
                    rewardBuffer.append(reward)

                    if len(stateBuffer) >= nSteps:
                        memory.addMemory(stateBuffer.pop(0), rewardBuffer, np.mean(statePrime, axis=0), done)
                        rewardBuffer.pop(0)
                        # agent training
                        lr = 0.001
                        memSample = memory.sampleMemory(c.BATCH_SIZE)
                        agent.fit(memSample, lr=lr)

                    state = statePrime
                    if done:
                        break
            print('.', end='')
        print()
        totalSec = time()-startTime
        printTime(totalSec)
        return totalSec

    except RuntimeError as e:
        print('failed')


all_devices = tf.config.experimental.list_physical_devices('GPU') + tf.config.experimental.list_physical_devices('CPU')
all_devices = [d[0][-5:] for d in all_devices]
print('available devices')
print(all_devices)
#device = '/device:' + all_devices[0]

reps = 5
print('trying with cpu')
with tf.device('/device:CPU:0'):
    t = 0
    for _ in range(reps):
        t += test()
    print('average')
    printTime(t/reps)


print('\n\n', 'trying with gpu')
with tf.device('/device:GPU:0'):
    t = 0
    for _ in range(reps):
        t += test()
    print('average')
    printTime(t/reps)


print('\n\n', 'trying without device')
t = 0
for _ in range(reps):
    t += test()
print('average')
printTime(t/reps)