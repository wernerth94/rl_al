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
import os, sys
from importlib import reload
from time import time
import tensorflow.keras as keras
import tensorflow as tf
from multiprocessing import Process, Pipe

import Classifier
import Environment
import Agent
import Memory
import Plotting
import Misc

import config.mnistConfig as c
print('#########################################################')
print('loaded config', c.MODEL_NAME)

print('planned interactions', c.MIN_INTERACTIONS)
print(c.N_EXPLORE, 'exploration', c.N_CONVERSION, 'conversion')
print('#########################################################')
print('#########################################################')

if c.DATASET == 'mnist':
    from Data import loadMNIST
    dataset = loadMNIST()
else:
    from Data import load_mnist_embedded
    dataset = load_mnist_embedded(c.DATASET)


def runAL(args):
    outConnection, dataset = args
    import tensorflow as tf
    tf.config.optimizer.set_jit(True)

    import config.mnistConfig as c
    if c.DATASET == 'mnist':
        classifier = Classifier.DenseClassifierMNIST
    else:
        classifier = Classifier.EmbeddingClassifier(c.EMBEDDING_SIZE)

    env = Environment.ALGame(dataset=dataset, modelFunction=classifier, config=c, verbose=0)
    memory = Memory.NStepMemory(env.stateSpace, c.N_STEPS, maxLength=c.MEMORY_CAP)
    agent = Agent.DDVN(env.stateSpace, gamma=c.AGENT_GAMMA)

    stop = False
    while not stop:
        _ = env.reset()
        # receive seeds
        npSeed, tfSeed = outConnection.recv()
        np.random.seed(npSeed)
        tf.random.set_seed(tfSeed)
        #receive current total steps, greed
        totalSteps, greed = outConnection.recv()

        epochLoss, epochRewards = 0, 0
        steps, vStart = 0, 0
        while not env.hardReset:
            state = env.reset()
            done = False
            stateBuffer = []; rewardBuffer = []
            while not done:
                for n in range(c.N_STEPS):
                    V, a = agent.predict(state, greedParameter=greed)
                    a = a[0]
                    if steps == 0:
                        vStart = V[a]

                    statePrime, reward, done = env.step(a)

                    epochRewards += reward
                    stateBuffer.append(state[a])
                    rewardBuffer.append(reward)

                    if len(stateBuffer) >= c.N_STEPS:
                        memory.addMemory(stateBuffer.pop(0), rewardBuffer, np.mean(statePrime, axis=0), done)
                        rewardBuffer.pop(0)

                    if done:
                        break
                    state = statePrime
        # send back memories
        outConnection.send(memory.memory)
        memory.clear()
        # send reward and V(s0)
        outConnection.send([epochRewards, vStart, env.currentTestF1])
        # receive new agent weights
        weights = outConnection.recv()
        agent.setAgentWeights(weights)
        # receive stop signal
        stop = outConnection.recv()


def trainAgent(args):
    conn, STATE_SPACE = args
    import tensorflow as tf
    tf.config.optimizer.set_jit(True)

    cp_callback = keras.callbacks.ModelCheckpoint(c.stateValueDir, verbose=0, save_freq=c.C, save_weights_only=False)
    mainAgent = Agent.DDVN(STATE_SPACE, gamma=c.AGENT_GAMMA, fromCheckpoints=c.stateValueDir, callbacks=[cp_callback])

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

if len(sys.argv) > 1:
    NUM_PROCESSES = int(sys.argv[1])
else:
    NUM_PROCESSES = 4


# STATE_SPACE = 7
STATE_SPACE = 1
mainMemory = Memory.NStepMemory(STATE_SPACE, c.N_STEPS, maxLength=c.MEMORY_CAP)
mainMemory.loadFromDisk(c.memDir)

if c.USE_STOPSWITCH:
    Misc.createStopSwitch()

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

# start Environment processes
childProcesses = list()
connections = list()
# start processes
for i in range(NUM_PROCESSES):
    parent_conn, child_conn = Pipe()
    args = [child_conn, dataset]
    p = Process(target=runAL, args=(args,))
    childProcesses.append(p)
    connections.append(parent_conn)
    p.start()

# start agent training process
trainingConn, child_conn = Pipe()
trainingProcess = Process(target=trainAgent, args=([child_conn, STATE_SPACE],))
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
        trainState['f1Curve'].append(meanF1)
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
        trainState['lrCurve'].append(lr)
        trainState['greedCurve'].append(greed)

        if printCounter >= c.PRINT_FREQ:
            printCounter = 0
            timePerStep = (time()-startTime) / float(sessionSteps)
            etaSec = timePerStep * (c.MIN_INTERACTIONS - trainState['totalSteps'] - 1)
            etaH = etaSec / 60 / 60
            print('ETA %1.2f h | total steps %d \t '
                  'epoch reward %1.3f \t meanF1 %1.3f \t greed %1.3f \t lr %1.5f' % (
                      etaH, trainState['totalSteps'], meanReward, meanF1, greed, lr))
            trainState['eta'] = etaH
            reload(Plotting)
            Plotting.plot(trainState, c, c.OUTPUT_FOLDER)
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