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

import config.modelConfig as c


if c.DATASET == 'mnist':
    from Data import loadMNIST
    dataset = loadMNIST()
else:
    from Data import load_mnist_embedded
    dataset = load_mnist_embedded(c.DATASET)

# Worker
def collectData(args):
    outConnection, dataset = args
    import tensorflow as tf
    tf.config.optimizer.set_jit(True)

    import config.modelConfig as c
    if c.DATASET == 'mnist':
        classifier = Classifier.DenseClassifierMNIST
    else:
        classifier = Classifier.EmbeddingClassifier(c.EMBEDDING_SIZE)

    env = Environment.ALGame(dataset=dataset, modelFunction=classifier, config=c, verbose=0)
    memory = Memory.NStepVMemory(env.stateSpace, c.N_STEPS, maxLength=c.MEMORY_CAP)
    agent = Agent.DDVN(env.stateSpace, gamma=c.AGENT_GAMMA, nHidden=c.AGENT_NHIDDEN)

    stop = False
    while not stop:
        _ = env.reset()
        # receive seeds
        npSeed, tfSeed = outConnection.recv()
        if c.DEBUG: print('start data collection')
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

# Worker
def trainAgentInModel(agent, dynamicsModel, realMemorySample, greed=0.1):
    memory = Memory.NStepVMemory(dynamicsModel.sample().shape[1], c.N_STEPS, maxLength=c.MEMORY_CAP)
    for epoch in range(c.DATA_COLLECTION_EPOCHS):
        state = dynamicsModel.sample()
        stateBuffer = []; rewardBuffer = []
        for i in range(c.TRAJECTORY_LENGTH):
            V, a = agent.predict(state, greedParameter=greed)
            a = a[0]
            nextState = dynamicsModel(np.expand_dims(state[a], axis=0), additionalSamples=10)
            proxyReward = np.mean(nextState[:, 0]) - state[a, 0]
            stateBuffer.append(state[a])
            rewardBuffer.append(proxyReward)

            if len(stateBuffer) >= c.N_STEPS:
                memory.addMemory(stateBuffer.pop(0), rewardBuffer, np.mean(nextState, axis=0), False)
                rewardBuffer.pop(0)
            state = nextState
    memory._append(realMemorySample)

    trainSteps = 0
    sampleSize = int(len(memory)*0.6)
    for epoch in range(c.AGENT_TRAINING_EPOCHS):
        trainSteps += sampleSize
        sample = memory.sampleMemory(sampleSize)
        loss = agent.fit(sample)
        if trainSteps > c.C:
            trainSteps -= c.C
            agent.copyWeights()

    return loss


# Worker
def trainAgent(args):
    conn, STATE_SPACE = args
    import tensorflow as tf
    from AutoEncoder import VAE
    from Misc import trainTestIDSplit
    import config.modelConfig as c
    tf.config.optimizer.set_jit(True)

    cp_callback = keras.callbacks.ModelCheckpoint(c.stateValueDir, verbose=0, save_freq=c.C, save_weights_only=False)
    mainAgent = Agent.DDVN(STATE_SPACE, gamma=c.AGENT_GAMMA, nHidden=c.AGENT_NHIDDEN, fromCheckpoints=c.stateValueDir, callbacks=[cp_callback])
    dynamicsModel = VAE(STATE_SPACE, alpha=0.01)
    dynamicsModel.compile(optimizer=tf.keras.optimizers.Adam(0.00002))
    es_callback = keras.callbacks.EarlyStopping(patience=2)

    stop = False
    while not stop:
        # receive memory
        mem, lr, updateTargetNet = conn.recv()
        if c.DEBUG: print('start dynamic model training')
        # train world model
        state, _, newState, __ = mem.sampleMemory(c.BATCH_SIZE * c.BUDGET)
        train, test = trainTestIDSplit(len(state))
        if len(train) > 0:
            train_hist = dynamicsModel.fit(state[train], newState[train], validation_data=(state[test], newState[test]),
                                            epochs=100, batch_size=32, callbacks=[es_callback], verbose=0)
            print('world model recon loss', train_hist.history['reconstruction_loss'][0], 'mae', train_hist.history['mae'][0])

        if c.DEBUG: print('start agent training')
        loss = trainAgentInModel(mainAgent, dynamicsModel, mem.sampleMemory(c.DATA_COLLECTION_EPOCHS * c.TRAJECTORY_LENGTH, returnTable=True))
        # send loss and new weights
        conn.send(loss)
        conn.send(mainAgent.getAgentWeights())
        # receive stop signal
        stop = conn.recv()



##############################################################################
###### Main ##################################################################

if len(sys.argv) > 1:
    NUM_PROCESSES = int(sys.argv[1])
else:
    NUM_PROCESSES = 4


mainMemory = Memory.NStepVMemory(Environment.ALGame.stateSpace, c.N_STEPS, maxLength=c.MEMORY_CAP)
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
seed = 1622619045 # 2 June 2021 9:31
nextTargetNetUpdate = c.WARMUP + c.C

# start Environment processes
childProcesses = list()
connections = list()
# start processes
for i in range(NUM_PROCESSES):
    parent_conn, child_conn = Pipe()
    args = [child_conn, dataset]
    p = Process(target=collectData, args=(args,))
    childProcesses.append(p)
    connections.append(parent_conn)
    p.start()

# start agent training process
trainingConn, child_conn = Pipe()
trainingProcess = Process(target=trainAgent, args=([child_conn, Environment.ALGame.stateSpace],))
trainingProcess.start()

try:
    while trainState['totalSteps'] < c.MIN_INTERACTIONS:
        # send seeds
        for i, conn in enumerate(connections):
            npSeed = int(seed / (i+1)) + len(trainState['rewardCurve'])
            tfSeed = int(seed / 2 / (i+1)) + len(trainState['rewardCurve'])
            conn.send([npSeed, tfSeed])

        # send total steps and greed
        # start data collection
        greed = c.GREED[np.clip(trainState['totalSteps'], 0, len(c.GREED) - 1)]
        # gl = c.GL[np.clip(trainState['totalSteps'], 0, len(c.GL) - 1)]
        for i, conn in enumerate(connections):
            conn.send([trainState['totalSteps'], greed])

        # agent training
        lr = c.LR[np.clip(trainState['totalSteps'], 0, len(c.LR) - 1)] # some dummies to keep the
        # target network update schedule
        updateTargetNet = trainState['totalSteps'] >= nextTargetNetUpdate
        if updateTargetNet:
            nextTargetNetUpdate += c.C
        #memSample = mainMemory.sampleMemory(c.BATCH_SIZE * c.BUDGET, returnTable=True)
        # send memory data to training process
        trainingConn.send([mainMemory, lr, updateTargetNet])
        # receive loss from agent training
        epochLoss = trainingConn.recv()

        # receive memories
        for i, conn in enumerate(connections):
            mem = conn.recv()
            mainMemory._append(mem)

        # receive rewards and vStart from collection workers
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

        # receive new agent weights and update the childs
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
        stop = c.USE_STOPSWITCH and not Misc.checkStopSwitch()
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
    p.join(timeout=100)
    p.terminate()
    p.close()
trainingProcess.join(timeout=100)
trainingProcess.terminate()
trainingProcess.close()
print('all processes closed')