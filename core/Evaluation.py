import numpy as np
from PoolManagement import *
import Memory

def scoreAgent(agent, env, budget, dataset, printInterval=10, greed=0, imgsPerStep=1):
    STATE_SPACE = 3 # + 2 * dataset[0].shape[1]
    memory = Memory.NStepMemory(STATE_SPACE, nSteps=1)

    xLabeled, yLabeled, xUnlabeled, yUnlabeled, perClassInstances = resetALPool(dataset)
    env.reset(xLabeled, yLabeled)
    stateIds = sampleNewBatch(xUnlabeled)
    state = createState(env, xUnlabeled, xLabeled, stateIds)
    f1Prog = []
    i = 0
    done = False
    while not done:
        #for nImg in range(imgsPerStep):
        V, a = agent.predict(state, greedParameter=greed)
        a = a[0]
        xLabeled, yLabeled, xUnlabeled, yUnlabeled, perClassInstances = addDatapointToPool(xLabeled, yLabeled,
                                                                                          xUnlabeled, yUnlabeled,
                                                                                          perClassInstances, a)
        reward = env.fitClassifier(xLabeled, yLabeled)
        stateIds = sampleNewBatch(xUnlabeled)
        statePrime = createState(env, xUnlabeled, xLabeled, stateIds)
        done = checkDone(xLabeled, yUnlabeled, budget)

        #for _ in range(imgsPerStep):
        f1Prog.append(env.currentTestF1)
        memory.addMemory(state[a], [reward], np.mean(statePrime, axis=0), done)

        state = statePrime
        if i % printInterval == 0 and len(f1Prog) > 0:
            print('%d | %1.3f'%(i, f1Prog[-1]))
        i += 1

    print('stopping with', len(yLabeled) - c.INIT_POINTS_PER_CLASS * yLabeled.shape[1], 'images \t f1', f1Prog[-1])
    return memory, f1Prog