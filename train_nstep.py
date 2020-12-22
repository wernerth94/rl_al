import sys
import getpass
print(F"The user is: {getpass.getuser()}")
print(F"The virtualenv is: {sys.prefix}")

# path additions for the cluster
sys.path.append("core")
sys.path.append("evaluation")
sys.path.append("config")
print(F"updated path is {sys.path}")

import numpy as np
import os, gc
from time import time
import tensorflow.keras as keras
import tensorflow as tf

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


import config.batchConfig as c

print('#########################################################')
print('loaded config', c.MODEL_NAME, '\t DATASET', dataset)

print('planned interactions', c.MIN_INTERACTIONS)
print(c.N_EXPLORE, 'exploration', c.N_CONVERSION, 'conversion')
print('#########################################################')
print('#########################################################')

if dataset == 'cifar':
    from Data import load_cifar10_mobilenet
    dataset = load_cifar10_mobilenet()
elif dataset == 'mnist':
    from Data import load_mnist_mobilenet
    dataset = load_mnist_mobilenet()

classifier = Classifier.EmbeddingClassifier(1280)


xLabeled = np.array(0)
yLabeled = np.array(0)
xUnlabeled = np.array(0)
yUnlabeled = np.array(0)
perClassIntances = []

def resetALPool():
    global xLabeled
    global yLabeled
    global xUnlabeled
    global yUnlabeled
    global perClassIntances
    x_train, y_train, x_test, y_test = dataset
    nClasses = y_train.shape[1]
    xLabeled, yLabeled = [], []

    ids = np.arange(x_train.shape[0], dtype=int)
    np.random.shuffle(ids)
    perClassIntances = [0 for _ in range(nClasses)]
    usedIds = []
    for i in ids:
        label = np.argmax(y_train[i])
        if perClassIntances[label] < c.INIT_POINTS_PER_CLASS:
            xLabeled.append(x_train[i])
            yLabeled.append(y_train[i])
            usedIds.append(i)
            perClassIntances[label] += 1
        if sum(perClassIntances) >= c.INIT_POINTS_PER_CLASS * nClasses:
            break
    unusedIds = [i for i in np.arange(x_train.shape[0]) if i not in usedIds]
    xLabeled = np.array(xLabeled)
    yLabeled = np.array(yLabeled)
    xUnlabeled = np.array(x_train[unusedIds])
    yUnlabeled = np.array(y_train[unusedIds])
    gc.collect()


def addDatapointToPool(dpId:int):
    global xLabeled
    global yLabeled
    global xUnlabeled
    global yUnlabeled
    global perClassIntances
    # add images
    perClassIntances[int(np.argmax(yUnlabeled[dpId]))] += 1 # keep track of the added images
    xLabeled = np.append(xLabeled, xUnlabeled[dpId:dpId + 1], axis=0)
    yLabeled = np.append(yLabeled, yUnlabeled[dpId:dpId + 1], axis=0)
    xUnlabeled = np.delete(xUnlabeled, dpId, axis=0)
    yUnlabeled = np.delete(yUnlabeled, dpId, axis=0)


def addPoolInformation(stateIds, alFeatures):
    global xUnlabeled
    presentedImg = xUnlabeled[stateIds]
    labeledPool = np.mean(xLabeled, axis=0)
    poolFeat = np.tile(labeledPool, (len(alFeatures),1))
    return np.concatenate([alFeatures, presentedImg, poolFeat], axis=1)


def createState(stateIds):
    global env
    alFeatures = env.createState(xUnlabeled[stateIds])
    state = addPoolInformation(stateIds, alFeatures)
    return state


def sampleNewBatch():
    global xUnlabeled
    return np.random.choice(xUnlabeled.shape[0], c.SAMPLE_SIZE)


def checkDone():
    global xUnlabeled
    global yUnlabeled
    return len(xLabeled) - (c.INIT_POINTS_PER_CLASS * yUnlabeled.shape[1]) >= c.BUDGET


env = Environment.ALGame(dataset=dataset, modelFunction=classifier, config=c, verbose=0)
STATE_SPACE = env.stateSpace + 2*dataset[0].shape[1]

memory = Memory.NStepMemory(STATE_SPACE, nSteps, maxLength=c.MEMORY_CAP)
memory.loadFromDisk(c.memDir)

cp_callback = keras.callbacks.ModelCheckpoint(c.stateValueDir, verbose=0, save_freq=c.C,
                                              save_weights_only=False)
agent = Agent.DDVN(STATE_SPACE, nSteps, fromCheckpoints=c.stateValueDir, callbacks=[cp_callback])

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

sessionSteps = 0; printCounter = 0
startTime = time()
seed = int(str(startTime)[-5:])
stateBuffer = []; rewardBuffer = []
try:
    while trainState['totalSteps'] < c.MIN_INTERACTIONS:
        np.random.seed(seed + len(trainState['rewardCurve']))
        tf.random.set_seed(int(seed/2) + len(trainState['rewardCurve']))

        resetALPool()
        env.reset(xLabeled, yLabeled)
        stateIds = sampleNewBatch()
        state = createState(stateIds)

        epochLoss, epochRewards = 0, 0
        steps, done, qAvrg = 0, False, 0
        while not done:
            for n in range(nSteps):
                #gl = c.GL[np.clip(trainState['totalSteps'], 0, len(c.GL) - 1)]

                greed = c.GREED[np.clip(trainState['totalSteps'], 0, len(c.GREED) - 1)]
                V, a = agent.predict(state, greedParameter=greed)
                a = a[0]
                if steps == 0:
                    qAvrg = V[a]

                addDatapointToPool(a)
                reward = env.fitClassifier(xLabeled, yLabeled)
                stateIds = sampleNewBatch()
                statePrime = createState(stateIds)
                done = checkDone()

                epochRewards += reward
                stateBuffer.append(state[a])
                rewardBuffer.append(reward)

                if len(stateBuffer) >= nSteps:
                    memory.addMemory(stateBuffer.pop(0), rewardBuffer, np.mean(statePrime, axis=0), done)
                    rewardBuffer.pop(0)

                    # agent training
                    lr = c.LR[np.clip(trainState['totalSteps'], 0, len(c.LR) - 1)]
                    if trainState['totalSteps'] > c.WARMUP:
                        for _ in range(c.RL_UPDATES_PER_ENV_UPDATE):
                            memSample = memory.sampleMemory(c.BATCH_SIZE)
                            epochLoss += agent.fit(memSample, lr=lr)

                        # Update target network
                        if trainState['totalSteps'] % c.C == 0:
                            agent.copyWeights()

                state = statePrime
                trainState['totalSteps'] += 1
                steps += 1

                if done:
                    break


        sessionSteps += steps
        printCounter += steps

        # logging ##################
        lossPerStep = epochLoss / steps
        trainState['lossCurve'].append(lossPerStep)

        trainState['rewardCurve'].append(epochRewards)
        trainState['imgCurve'].append(xLabeled.shape[0])
        trainState['stepCurve'].append(steps)
        trainState['qCurve'].append(qAvrg / steps)
        trainState['lrCurve'].append(lr)
        trainState['greedCurve'].append(greed)
        #trainState['glCurve'].append(gl)

        if printCounter >= c.PRINT_FREQ:
            printCounter = 0
            timePerStep = (time()-startTime) / float(sessionSteps)
            etaSec = timePerStep * (c.MIN_INTERACTIONS - trainState['totalSteps'] - 1)
            etaH = etaSec / 60 / 60
            print('ETA %1.2f h | steps %d \t total steps %d \t loss/step %1.6f \t '
                  'epoch reward %1.3f \t greed %1.3f \t lr %1.5f' % (
                      etaH, steps, trainState['totalSteps'], lossPerStep, epochRewards, greed, lr))
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