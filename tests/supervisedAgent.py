import os
import numpy as np
import Memory, Agent
import config.mnistConfig as c
import tensorflow.keras as K
import tensorflow as tf
import json

STATE_SPACE = 3

backlogDir = '../memoryBacklog'
memory = Memory.NStepVMemory(STATE_SPACE, c.N_STEPS, maxLength=np.inf)
for dir in os.listdir(backlogDir):
    memDir = os.path.join(backlogDir, dir)
    m = Memory.NStepVMemory(STATE_SPACE, c.N_STEPS, maxLength=np.inf)
    m.loadFromDisk(memDir)
    memory._append(m.memory)
print('memory size', len(memory))

def trainAgent(dataset, lr, bs, nHidden, activation, callbacks=[], verbose=1, agent=None):
    if not agent:
        agent = Agent.DDVN(STATE_SPACE, n_hidden=nHidden, activation=activation, callbacks=callbacks, gamma=0.0)
    for epoch in range(3):
        lastLoss = agent.fit(dataset, lr=lr, batchSize=bs, verbose=verbose)
    return lastLoss, agent

def gridSearch(callbacks=[]):
    paramList = list()
    lossList = list()
    allIds = np.arange(len(memory))
    np.random.shuffle(allIds)
    cutoff = int(len(memory) * 0.8)
    train = memory.rowsToArgs(memory.memory[allIds[:cutoff]])
    test = memory.rowsToArgs(memory.memory[allIds[cutoff:]])
    # dataset = memory.rowsToArgs(memory.memory)
    i, total = 0, 2*2
    for lr in [0.001]:
        for bs in [16]:
            for nHidden in [10, 50]:
                for activation in ['tanh', 'relu']:
                    loss = 0
                    valLoss = 0
                    for run in range(3):
                        print('train')
                        l, agent = trainAgent(train, lr, bs, nHidden, activation, callbacks=callbacks)
                        loss += l
                        print('test')
                        vl, _ = trainAgent(test, lr, bs, nHidden, activation, callbacks=callbacks)
                        valLoss += vl
                    loss /= 3.0
                    valLoss /= 3.0
                    lossList.append(valLoss)
                    paramList.append( (lr, bs, nHidden, activation) )
                    i += 1
                    print(i, '/', total, '|', lr, bs, nHidden, activation, ': train', loss, 'validation', valLoss)

    sort = sorted(zip(lossList, paramList), reverse=True)
    print(sort)

    json.dump(sort, open('gridsearch', 'w'))

# (2.021148247877136e-05, (0.001, 16, 50, 'tanh')),
# (1.8936551593166467e-05, (0.001, 16, 10, 'tanh')),
# (1.836288841635299e-05, (0.001, 16, 50, 'relu')),
# (1.8352149102914456e-05, (0.001, 16, 10, 'relu'))

# best setting:
# ReLU
# BS [16, 32]
# nHidden [50-80]
# LR 0.001
tf.config.optimizer.set_jit(True)
cp_callback = K.callbacks.ModelCheckpoint('supervisedAgent', verbose=0, save_freq=2000, save_weights_only=False)
gridSearch()
#trainAgent(train, test, lr=0.001, bs=16, nHidden=80, activation='relu', callbacks=[], verbose=1)


# gridSearch()
# (0.0008957878356644263, (0.001, 16, 20, 'tanh')),
# (0.0006555954993624861, (0.001, 32, 20, 'relu')),
# (0.00045088532109123963, (0.001, 32, 20, 'tanh')),
# (0.00034861832197445136, (0.01, 32, 80, 'tanh')),
# (0.0003481125944138815, (0.005, 32, 20, 'tanh')),
# (0.0003467305117131521, (0.001, 32, 50, 'tanh')),
# (0.0003300141785681869, (0.005, 32, 80, 'tanh')),
# (0.0003165133045210193, (0.01, 16, 20, 'tanh')),
# (0.00030318520051271963, (0.005, 16, 20, 'tanh')),
# (0.00029127173669015366, (0.001, 16, 50, 'tanh')),
# (0.00028319726698100567, (0.01, 32, 50, 'relu')),
# (0.00026574807028130937, (0.001, 32, 80, 'tanh')),
# (0.00026305150822736323, (0.01, 32, 20, 'tanh')),
# (0.00026233736813689273, (0.01, 16, 50, 'tanh')),
# (0.00026209788241734106, (0.005, 16, 50, 'tanh')),
# (0.0002461722712420548, (0.01, 16, 20, 'relu')),
# (0.00024178855043525496, (0.005, 16, 80, 'tanh')),
# (0.00023577372000242272, (0.001, 16, 80, 'tanh')),
# (0.00023412246082443744, (0.005, 16, 50, 'relu')),
# (0.00023136816162150353, (0.005, 32, 50, 'tanh')),
# (0.0002301592806664606, (0.01, 16, 80, 'relu')),
# (0.00021845679536151388, (0.005, 16, 20, 'relu')),
# (0.00021774309182850024, (0.01, 16, 50, 'relu')),
# (0.0002105860961213087, (0.01, 32, 50, 'tanh')),
# (0.00020363858493510634, (0.005, 32, 50, 'relu')),
# (0.000203373609110713, (0.01, 16, 80, 'tanh')),
# (0.0002002567343879491, (0.01, 32, 80, 'relu')),
# (0.00019837991567328572, (0.01, 32, 20, 'relu')),
# (0.00019700651561530927, (0.005, 32, 20, 'relu')),
# (0.00017379399408431104, (0.005, 16, 80, 'relu')),
# (0.00017316483717877418, (0.005, 32, 80, 'relu')),
# (0.0001690921174789158, (0.001, 16, 50, 'relu')),
# (0.0001555896936527764, (0.001, 32, 50, 'relu')),
# (0.00012340672158946595, (0.001, 32, 80, 'relu')),
# (0.00011153102726287518, (0.001, 16, 80, 'relu')),
# (0.0001115064587793313, (0.001, 16, 20, 'relu'))
