import sys
import getpass
print(F"The user is: {getpass.getuser()}")
print(F"The virtualenv is: {sys.prefix}")

# path additions for the cluster
sys.path.append("core")
sys.path.append("evaluation")
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
import Memory
from core.Plotting import plot
import Misc

all_datasets = ['mnist', 'iris']
all_setups = ['conv', 'dense', 'batch']
setup = str(sys.argv[1])
dataset = str(sys.argv[2])
nSteps = int(sys.argv[3])
if dataset not in all_datasets: raise ValueError('dataset not in all_datasets;  given: ' + dataset)
if setup not in all_setups: raise ValueError('setup not in all_setups;  given: ' + setup)

if setup == 'dense':
    import config.config as c
    envFunc = Environment.ImageClassificationGame
    agentFunc = Agent.DenseAgent
elif setup == 'conv':
    import config.convConfig as c
    envFunc = Environment.ConvALGame
    agentFunc = Agent.ConvAgent
elif setup == 'batch':
    import config.batchConfig as c
    envFunc = Environment.BatchALGame
    agentFunc = Agent.NStepBatchAgent

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

memory = Memory.NStepMemory(env, nSteps, maxLength=c.MEMORY_CAP)
memory.loadFromDisk(c.memDir)

cp_callback = keras.callbacks.ModelCheckpoint(c.ckptDir, verbose=0, save_freq=c.C,
                                              save_weights_only=False)
agent = agentFunc(env, nSteps, fromCheckpoints=c.ckptDir, callbacks=[cp_callback])

if c.USE_STOPSWITCH:
    Misc.createStopSwitch()

# Load Past Train State
trainState = Misc.loadTrainState(c)
if trainState is None:
    trainState = {
        'eta':0, 'totalSteps': 0,
        'lossCurve': [], 'stepCurve': [], 'rewardCurve': [],
        'imgCurve': [], 'qCurve': [],
        'lrCurve': [], 'greedCurve': [], 'glCurve': []
    }

sessionSteps = 0
startTime = time()
printCounter = 0
seed = int(str(startTime)[-5:])
nextUpdate = c.C + c.WARMUP
try:
    while trainState['totalSteps'] < c.MIN_INTERACTIONS:
        np.random.seed(seed + len(trainState['rewardCurve']))
        tf.random.set_seed(int(seed/2) + len(trainState['rewardCurve']))
        state = env.reset()
        epochLoss, epochRewards = 0, 0
        steps, done = 0, False
        qAvrg = 0
        while not done:
            startingState = None
            rewards = list()
            for n in range(nSteps):
                gl = c.GL[np.clip(trainState['totalSteps'], 0, len(c.GL) - 1)]
                env.gameLength = int(gl)

                greed = c.GREED[np.clip(trainState['totalSteps'], 0, len(c.GREED) - 1)]
                Q, a = agent.predict(state, greedParameter=greed)
                a = a[0]
                qAvrg += Q[0]
                if startingState is None:
                    startingState = state[a]

                statePrime, reward, done, _ = env.step(a)
                epochRewards += reward
                rewards.append(reward)

                state = statePrime
                trainState['totalSteps'] += 1
                steps += 1

                if done:
                    break

            memory.addMemory(startingState, rewards, np.mean(statePrime, axis=0), done)
            #memory.addMemory(state, a, reward, statePrime, done)

            # agent training
            lr = c.LR[np.clip(trainState['totalSteps'], 0, len(c.LR) - 1)]
            if trainState['totalSteps'] > c.WARMUP:
                for _ in range(c.RL_UPDATES_PER_ENV_UPDATE):
                    memSample = memory.sampleMemory(c.BATCH_SIZE)
                    epochLoss += agent.fit(memSample, lr=lr)

            # Update target network
            if trainState['totalSteps'] >= nextUpdate:
                agent.copyWeights()
                nextUpdate += c.C


        sessionSteps += steps
        printCounter += steps

        # logging ##################
        lossPerStep = epochLoss / steps
        trainState['lossCurve'].append(lossPerStep)

        trainState['rewardCurve'].append(epochRewards)
        trainState['imgCurve'].append(env.xLabeled.shape[0])
        trainState['stepCurve'].append(steps)
        trainState['qCurve'].append(qAvrg / steps)
        trainState['lrCurve'].append(lr)
        trainState['greedCurve'].append(greed)
        trainState['glCurve'].append(gl)

        if printCounter >= c.PRINT_FREQ:
            printCounter = 0
            timePerStep = (time()-startTime) / float(sessionSteps)
            etaSec = timePerStep * (c.MIN_INTERACTIONS - trainState['totalSteps'] - 1)
            etaH = etaSec / 60 / 60
            print('ETA %1.2f h | game length %1.0f \t steps %d \t total steps %d \t loss/step %1.6f \t '
                  'epoch reward %1.3f \t greed %1.3f \t lr %1.5f' % (
                      etaH, gl, steps, trainState['totalSteps'], lossPerStep, epochRewards, greed, lr))
            trainState['eta'] = etaH
            plot(trainState, c, c.OUTPUT_FOLDER)
            memory.writeToDisk(c.memDir)
            Misc.saveTrainState(c, trainState)

        if c.USE_STOPSWITCH and not os.path.exists('stopSwitch'):
            break
except KeyboardInterrupt:
    print('training stopped by user')

#######################################
# Finalization
memory.writeToDisk(c.memDir)
Misc.saveTrainState(c, trainState)

print('took', (time()-startTime)/60.0/60.0, 'hours')