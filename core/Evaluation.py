import numpy as np

def scoreAgent(agent, env, numImgs, printInterval=10):
    state = env.reset()
    lossProg = []
    f1Prog = []
    i = 0
    done = False
    while not done:
        Q, a = agent.predict(state, greedParameter=-1)
        a = a[0]
        state, reward, done, _ = env.step(a)

        if env.sampleSize == env.actionSpace or a < Q.shape[1] - 1:
            for _ in range(env.imgsToAvrg):
                f1Prog.append(env.currentTestF1)
                lossProg.append(env.currentTestLoss)

        i += 1
        if i % printInterval == 0 and len(f1Prog) > 0:
            print('%d | %d : %1.3f'%(i, env.addedImages, f1Prog[-1]))

    print('stopping with', len(f1Prog))
    if env.addedImages >= numImgs:
        return f1Prog, lossProg
    else:
        raise AssertionError('not converged')