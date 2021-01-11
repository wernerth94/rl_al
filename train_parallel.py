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
from multiprocessing import Process, Pipe

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


def runAL(args):
    outConnection, STATE_SPACE, dataset = args

    import config.batchConfig as c
    classifier = Classifier.EmbeddingClassifier(1280)

    def resetALPool(dataset):
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
        return xLabeled, yLabeled, xUnlabeled, yUnlabeled, perClassIntances


    def addDatapointToPool(xLabeled, yLabeled, xUnlabeled, yUnlabeled, perClassIntances, dpId:int):
        # add images
        perClassIntances[int(np.argmax(yUnlabeled[dpId]))] += 1 # keep track of the added images
        xLabeled = np.append(xLabeled, xUnlabeled[dpId:dpId + 1], axis=0)
        yLabeled = np.append(yLabeled, yUnlabeled[dpId:dpId + 1], axis=0)
        xUnlabeled = np.delete(xUnlabeled, dpId, axis=0)
        yUnlabeled = np.delete(yUnlabeled, dpId, axis=0)
        return xLabeled, yLabeled, xUnlabeled, yUnlabeled, perClassIntances


    def addPoolInformation(xUnlabeled, xLabeled, stateIds, alFeatures):
        presentedImg = xUnlabeled[stateIds]
        labeledPool = np.mean(xLabeled, axis=0)
        poolFeat = np.tile(labeledPool, (len(alFeatures),1))
        return np.concatenate([alFeatures, presentedImg, poolFeat], axis=1)


    def createState(env, xUnlabeled, xLabeled, stateIds):
        alFeatures = env.createState(xUnlabeled[stateIds])
        state = addPoolInformation(xUnlabeled, xLabeled, stateIds, alFeatures)
        return state


    def sampleNewBatch(xUnlabeled):
        return np.random.choice(xUnlabeled.shape[0], c.SAMPLE_SIZE)


    def checkDone(xLabeled, yUnlabeled):
        return len(xLabeled) - (c.INIT_POINTS_PER_CLASS * yUnlabeled.shape[1]) >= c.BUDGET


    env = Environment.ALGame(dataset=dataset, modelFunction=classifier, config=c, verbose=0)
    memory = Memory.NStepMemory(STATE_SPACE, nSteps, maxLength=c.MEMORY_CAP)
    agent = Agent.DDVN(STATE_SPACE, nSteps)

    stop = False
    while not stop:
        stateBuffer = []; rewardBuffer = []

        # receive seeds
        npSeed, tfSeed = outConnection.recv()
        np.random.seed(npSeed)
        tf.random.set_seed(tfSeed)

        xLabeled, yLabeled, xUnlabeled, yUnlabeled, perClassIntances = resetALPool(dataset)
        env.reset(xLabeled, yLabeled)
        stateIds = sampleNewBatch(xUnlabeled)
        state = createState(env, xUnlabeled, xLabeled, stateIds)

        #receive current total steps, greed
        totalSteps, greed = outConnection.recv()

        epochLoss, epochRewards = 0, 0
        steps, done, vStart = 0, False, 0
        while not done:
            for n in range(nSteps):
                V, a = agent.predict(state, greedParameter=greed)
                a = a[0]
                if steps == 0:
                    vStart = V[a]

                xLabeled, yLabeled, xUnlabeled, yUnlabeled, perClassIntances = addDatapointToPool(xLabeled, yLabeled, xUnlabeled, yUnlabeled, perClassIntances, a)
                reward = env.fitClassifier(xLabeled, yLabeled)
                stateIds = sampleNewBatch(xUnlabeled)
                statePrime = createState(env, xUnlabeled, xLabeled, stateIds)
                done = checkDone(xLabeled, yUnlabeled)

                epochRewards += reward
                stateBuffer.append(state[a])
                rewardBuffer.append(reward)

                if len(stateBuffer) >= nSteps:
                    memory.addMemory(stateBuffer.pop(0), rewardBuffer, np.mean(statePrime, axis=0), done)
                    rewardBuffer.pop(0)

                if done:
                    break
                state = statePrime

        # send back memories
        outConnection.send(memory.memory)
        # send reward and V(s0)
        outConnection.send([epochRewards, vStart, env.currentTestF1])
        # receive new agent weights
        weights = outConnection.recv()
        agent.setAgentWeights(weights)

        stop = outConnection.recv()
        memory.clear()


def trainAgent(args):
    conn, STATE_SPACE, BATCH_STATE = args

    cp_callback = keras.callbacks.ModelCheckpoint(c.stateValueDir, verbose=0, save_freq=c.C, save_weights_only=False)
    mainAgent = Agent.DDVN(STATE_SPACE, nSteps, fromCheckpoints=c.stateValueDir, callbacks=[cp_callback])

    stop = False
    while not stop:
        # receive memory
        mem, lr, updateTargetNet = conn.recv()
        loss = mainAgent.fit(mem, lr=lr)
        if updateTargetNet:
            mainAgent.copyWeights()
        # send loss
        conn.send(loss)
        #send new weights
        conn.send(mainAgent.getAgentWeights())
        # receive stop signal
        stop = conn.recv()



##############################################################################
###### Main ##################################################################
STATE_SPACE = 3 + 2*dataset[0].shape[1]
NUM_PROCESSES = 4

mainMemory = Memory.NStepMemory(STATE_SPACE, nSteps, maxLength=c.MEMORY_CAP)
mainMemory.loadFromDisk(c.memDir)

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
seed = startTime
nextTargetNetUpdate = c.WARMUP + c.C

# start Environment processes
childProcesses = list()
connections = list()
# start processes
for i in range(NUM_PROCESSES):
    parent_conn, child_conn = Pipe()
    args = [child_conn, STATE_SPACE, dataset]
    p = Process(target=runAL, args=(args,))
    childProcesses.append(p)
    connections.append(parent_conn)
    p.start()

# start agent training process
trainingConn, child_conn = Pipe()
trainingProcess = Process(target=trainAgent, args=([child_conn, STATE_SPACE, c.BATCH_SIZE],))
trainingProcess.start()

try:
    while trainState['totalSteps'] < c.MIN_INTERACTIONS:
        # send seeds
        for i, conn in enumerate(connections):
            npSeed = int(seed / (i+1)) + len(trainState['rewardCurve'])
            tfSeed = int(seed / 2 / (i+1)) + len(trainState['rewardCurve'])
            conn.send([npSeed, tfSeed])

        # send total steps and greed
        greed = c.GREED[np.clip(trainState['totalSteps'], 0, len(c.GREED) - 1)]
        # gl = c.GL[np.clip(trainState['totalSteps'], 0, len(c.GL) - 1)]
        for i, conn in enumerate(connections):
            conn.send([trainState['totalSteps'], greed])

        # receive memories
        for i, conn in enumerate(connections):
            mem = conn.recv()
            mainMemory._append(mem)

        # receive rewards and vStart
        meanReward, meanV, meanF1 = 0, 0, 0
        for i, conn in enumerate(connections):
            r, v, f1 = conn.recv()
            meanReward += r
            meanV += v
            meanF1 += f1
        meanReward /= NUM_PROCESSES
        meanV /= NUM_PROCESSES
        meanF1 /= NUM_PROCESSES
        trainState['rewardCurve'].append(meanReward)
        trainState['qCurve'].append(meanV)

        # agent training
        epochLoss = 0
        lr = c.LR[np.clip(trainState['totalSteps'], 0, len(c.LR) - 1)]

        updateTargetNet = trainState['totalSteps'] >= nextTargetNetUpdate
        memSample = mainMemory.sampleMemory(c.BATCH_SIZE * c.BUDGET)
        # send memory data to training process
        trainingConn.send([memSample, lr, updateTargetNet])
        # receive loss from agent training
        epochLoss += trainingConn.recv()

        if updateTargetNet:
            nextTargetNetUpdate += c.C

        newWeights = trainingConn.recv()
        for i, conn in enumerate(connections):
            conn.send(newWeights)

        trainState['totalSteps'] += c.BUDGET
        sessionSteps += c.BUDGET
        printCounter += c.BUDGET

        # logging ##################
        lossPerStep = epochLoss / c.BUDGET
        trainState['lossCurve'].append(lossPerStep)
        trainState['imgCurve'].append(c.BUDGET + (10 * c.INIT_POINTS_PER_CLASS))
        trainState['stepCurve'].append(c.BUDGET)
        trainState['lrCurve'].append(lr)
        trainState['greedCurve'].append(greed)
        #trainState['glCurve'].append(gl)

        if printCounter >= c.PRINT_FREQ:
            printCounter = 0
            timePerStep = (time()-startTime) / float(sessionSteps)
            etaSec = timePerStep * (c.MIN_INTERACTIONS - trainState['totalSteps'] - 1)
            etaH = etaSec / 60 / 60
            print('ETA %1.2f h | total steps %d \t '
                  'epoch reward %1.3f \t meanF1 %1.3f \t greed %1.3f \t lr %1.5f' % (
                      etaH, trainState['totalSteps'], meanReward, meanF1, greed, lr))
            trainState['eta'] = etaH
            plot(trainState, c, c.OUTPUT_FOLDER)
            mainMemory.writeToDisk(c.memDir)
            Misc.saveTrainState(c, trainState)

        # check for early stopping
        stop = c.USE_STOPSWITCH and not os.path.exists('stopSwitch')
        trainingConn.send(stop)
        for i, conn in enumerate(connections):
            conn.send(stop)
        if stop:
            break
except KeyboardInterrupt:
    print('training stopped by user')

#######################################
# Finalization
mainMemory.writeToDisk(c.memDir)
Misc.saveTrainState(c, trainState)

print('took', (time()-startTime)/60.0/60.0, 'hours')

for p in childProcesses:
    p.terminate()
    p.close()
trainingProcess.close()
print('all processes closed')