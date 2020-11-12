import sys
import getpass
print(F"The user is: {getpass.getuser()}")
print(F"The virtualenv is: {sys.prefix}")

# path additions for the cluster
sys.path.append("core")
sys.path.append("plotting")
print(F"updated path is {sys.path}")

import numpy as np
import os
from time import time
import tensorflow.keras as keras
import tensorflow as tf

import Data
import Classifier
import Environment
import Agent
from core.Memory import Memory
from core.Plotting import plot
import Misc
import DataAugmentation

all_datasets = ['mnist', 'iris']
all_setups = ['conv', 'dense']
dataset = str(sys.argv[-1])
setup = str(sys.argv[-2])
if dataset not in all_datasets: raise ValueError('dataset not in all_datasets;  given: ' + dataset)
if setup not in all_setups: raise ValueError('setup not in all_setups;  given: ' + setup)

if setup == 'dense':
    import config as c
    envFunc = Environment.ImageClassificationGame
    agentFunc = Agent.DenseAgent
elif setup == 'conv':
    import convConfig as c
    envFunc = Environment.ConvALGame
    agentFunc = Agent.ConvAgent

print('#########################################################')
print('loaded config', c.MODEL_NAME, '\t DATASET', dataset)

print('planned interactions', c.MIN_INTERACTIONS)
print(c.N_EXPLORE, 'exploration', c.N_CONVERSION, 'conversion')
print('#########################################################')
print('#########################################################')

if dataset == 'iris':
    dataset = Data.loadIRIS()
    classifier = Classifier.SimpleClassifier
elif dataset == 'mnist':
    dataset = Data.loadMNIST()
    classifier = Classifier.DenseClassifierMNIST

env = envFunc(dataset=dataset, modelFunction=classifier, config=c, verbose=0)

memory = Memory(env, maxLength=c.MEMORY_CAP, dataAugmentors=[DataAugmentation.GaussianNoise(0, 0.01)])
memory.loadFromDisk(c.memDir)
#memory = Memory(env, buildFromBacklog=True)

cp_callback = keras.callbacks.ModelCheckpoint(c.ckptDir, verbose=0, save_freq=c.C,
                                              save_weights_only=True)
agent = agentFunc(env, fromCheckpoints=c.ckptDir, callbacks=[cp_callback])

if c.USE_STOPSWITCH:
    Misc.createStopSwitch()

# Load Past Train State
tS = Misc.loadTrainState(c)
if tS is None:
    tS = {
        'eta':0,
        'totalSteps': 0,
        'lossCurve': [],
        'stepCurve': [],
        'rewardCurve': [],
        'imgCurve': [],
        'qCurve': []
    }

sessionSteps = 0
startTime = time()
printCounter = 0
seed = int(str(startTime)[-5:])
try:
    while tS['totalSteps'] < c.MIN_INTERACTIONS:
        np.random.seed(seed+len(tS['rewardCurve']))
        tf.random.set_seed(seed+len(tS['rewardCurve']))
        state = env.reset()
        epochLoss, epochRewards = 0, 0
        steps, done = 0, False
        qAvrg = 0
        while not done:
            gl = c.GL[np.clip(tS['totalSteps'], 0, len(c.GL)-1)]
            env.gameLength = int(gl)

            g = c.GREED[np.clip(tS['totalSteps'], 0, len(c.GREED)-1)]
            Q, a = agent.predict(state, greedParameter=g)
            a = a[0]
            qAvrg += Q[0, a]

            statePrime, reward, done, _ = env.step(a)
            epochRewards += reward

            memory.addMemory(state, a, reward, statePrime, done)

            # agent training
            lr = c.LR[np.clip(tS['totalSteps'], 0, len(c.LR)-1)]
            if tS['totalSteps'] > c.WARMUP:
                for _ in range(c.RL_UPDATES_PER_ENV_UPDATE):
                    memSample = memory.sampleMemory(c.BATCH_SIZE)
                    epochLoss += agent.fit(memSample, lr=lr)

            # Update target network
            if tS['totalSteps'] % c.C == 0:
                agent.copyWeights()

            state = statePrime
            tS['totalSteps'] += 1
            steps += 1
        sessionSteps += steps
        printCounter += steps

        # logging ##################
        lossPerStep = epochLoss / steps
        tS['lossCurve'].append(lossPerStep)

        tS['rewardCurve'].append(epochRewards)
        tS['imgCurve'].append(env.xLabeled.shape[0])
        tS['stepCurve'].append(steps)
        tS['qCurve'].append(qAvrg / steps)

        if printCounter >= c.PRINT_FREQ:
            printCounter = 0
            timePerStep = (time()-startTime) / float(sessionSteps)
            etaSec = timePerStep * (c.MIN_INTERACTIONS - tS['totalSteps'] - 1)
            etaH = etaSec / 60 / 60
            print('ETA %1.2f h | game length %1.0f \t steps %d \t total steps %d \t loss/step %1.6f \t '
                  'epoch reward %1.3f \t greed %1.3f \t lr %1.5f'%(
                  etaH, gl, steps, tS['totalSteps'],
                  lossPerStep, epochRewards, g, lr))

            tS['eta'] = etaH
            plot(tS, c, c.OUTPUT_FOLDER)
            memory.writeToDisk(c.memDir)
            Misc.saveTrainState(c, tS)

        if c.USE_STOPSWITCH and not os.path.exists('stopSwitch'):
            break
except KeyboardInterrupt:
    print('training stopped by user')

#######################################
# Finalization
memory.writeToDisk(c.memDir)
Misc.saveTrainState(c, tS)