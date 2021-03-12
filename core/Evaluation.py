import numpy as np
from PoolManagement import *
import Memory

def scoreAgent(agent, env, printInterval=10, greed=0, imgsPerStep=1):

    memory = Memory.NStepVMemory(env.stateSpace, nSteps=1)

    state = env.reset()
    f1Prog = []
    i = 0
    #done = False
    while not env.hardReset:
        #for nImg in range(imgsPerStep):
        V, a = agent.predict(state, greedParameter=greed)
        a = a[0]
        statePrime, reward, done = env.step(a)

        #for _ in range(imgsPerStep):
        f1Prog.append(env.currentTestF1)
        memory.addMemory(state[a], [reward], np.mean(statePrime, axis=0), done)

        state = statePrime
        if i % printInterval == 0 and len(f1Prog) > 0:
            print('%d | %1.3f'%(i, f1Prog[-1]))
        i += 1

    print('stopping with', len(env.yLabeled) - c.INIT_POINTS_PER_CLASS * env.yLabeled.shape[1], 'images \t f1', f1Prog[-1])
    return memory, f1Prog