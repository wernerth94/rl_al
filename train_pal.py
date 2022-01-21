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
from time import time
from importlib import reload
from tensorflow.keras.callbacks import ModelCheckpoint
import Classifier
import Environment
import Agent
import Memory
import Plotting
import Misc

import config.palConfig as c

if c.DATASET == 'mnist':
    from Data import loadMNIST
    dataset = loadMNIST()
    classifier = Classifier.DenseClassifierMNIST
else:
    from Data import load_mnist_embedded
    dataset = load_mnist_embedded(c.DATASET)
    classifier = Classifier.EmbeddingClassifierFactory(c.EMBEDDING_SIZE)

env = Environment.ALStreamingGame(dataset=dataset, modelFunction=classifier, config=c, verbose=0)
memory = Memory.NStepQMemory(env.stateSpace, c.N_STEPS, maxLength=c.MEMORY_CAP)

cp_callback = ModelCheckpoint(c.stateValueDir, verbose=0, save_freq=c.C, save_weights_only=False)
agent = Agent.DDQN(env.stateSpace, env.actionSpace, gamma=c.AGENT_GAMMA, nHidden=c.AGENT_NHIDDEN, callbacks=[cp_callback])

# Load Past Train State
trainState = Misc.loadTrainState(c)
if trainState is None:
    trainState = {
        'eta':0, 'totalSteps': 0,
        'lossCurve': [], 'rewardCurve': [],
        'f1Curve': [], 'qCurve': [],
        'lrCurve': [], 'greedCurve': []
    }

sessionSteps = 0; printCounter = 0
startTime = time()
seed = startTime
nextTargetNetUpdate = c.WARMUP + c.C
lr = 0.001

try:
    while trainState['totalSteps'] < c.MIN_INTERACTIONS:
        epochLoss, epochRewards = 0, 0
        steps, vStart = 0, 0
        done = False
        stateBuffer, rewardBuffer, actionBuffer = [], [], []
        state = env.reset()
        while not done:
            greed = c.GREED[np.clip(trainState['totalSteps'], 0, len(c.GREED) - 1)]
            Q, action = agent.predict(state, greed=greed)
            action = action[0]
            if steps == 0:
                vStart = Q[0, action]

            statePrime, reward, done = env.step(action)

            epochRewards += reward
            stateBuffer.append(state[0])
            rewardBuffer.append(reward)
            actionBuffer.append(action)

            if len(stateBuffer) >= c.N_STEPS:
                memory.addMemory(stateBuffer.pop(0), actionBuffer.pop(0), rewardBuffer, statePrime[0], done)
                rewardBuffer.pop(0)

                lr = c.LR[np.clip(trainState['totalSteps'], 0, len(c.LR) - 1)]
                updateTargetNet = trainState['totalSteps'] >= nextTargetNetUpdate
                if updateTargetNet:
                    nextTargetNetUpdate += c.C
                memSample = memory.sampleMemory(c.BATCH_SIZE)
                epochLoss += agent.fit(memSample, lr=lr)

            if done:
                break

            sessionSteps += 1
            printCounter += 1
            trainState['totalSteps'] += 1
            state = statePrime

        trainState['lrCurve'].append(len(env.xLabeled));        trainState['greedCurve'].append(float(greed))
        trainState['rewardCurve'].append(float(epochRewards));  trainState['lossCurve'].append(float(epochLoss))
        trainState['f1Curve'].append(float(env.currentTestF1))
        trainState['qCurve'].append(float(vStart))

        if printCounter >= c.PRINT_FREQ:
            printCounter = 0
            timePerStep = (time()-startTime) / float(sessionSteps)
            etaSec = timePerStep * (c.MIN_INTERACTIONS - trainState['totalSteps'] - 1)
            etaH = etaSec / 60 / 60
            print('ETA %1.2f h | total steps %d \t '
                  '# images %d \t epoch reward %1.3f \t F1 %1.3f \t greed %1.3f \t lr %1.5f' % (
                      etaH, trainState['totalSteps'], len(env.xLabeled), epochRewards, env.currentTestF1, greed, lr))
            trainState['eta'] = etaH
            reload(Plotting)
            Plotting.plot(trainState, c, c.OUTPUT_FOLDER)
            memory.writeToDisk(c.memDir)
            Misc.saveTrainState(c, trainState)

        if c.USE_STOPSWITCH and not Misc.checkStopSwitch():
            break

except KeyboardInterrupt:
    print('training stopped by user')

memory.writeToDisk(c.memDir)
Misc.saveTrainState(c, trainState)