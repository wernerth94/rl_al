
def scoreAgent(agent, env, printInterval=100, greed=0):

    #memory = Memory.NStepVMemory(env.stateSpace, nSteps=1)

    state = env.reset()
    f1Prog = []
    i = 0
    done = False
    while not done:
        V, a = agent.predict(state, greed=greed)
        a = a[0].item()
        statePrime, reward, done, _ = env.step(a)
        f1Prog.append(env.currentTestF1)
        #memory.addMemory(state[a].cpu().numpy(), [reward], np.mean(statePrime.cpu().numpy(), axis=0), done)
        state = statePrime
        if i % printInterval == 0 and len(f1Prog) > 0:
            print('%d | %1.3f'%(i, f1Prog[-1]))
        i += 1

    improvement = f1Prog[-1] - f1Prog[0]
    print('start %1.3f end %1.3f improvement %1.3f'%(f1Prog[0], f1Prog[-1], improvement))
    print("-------------------------------------------------------")
    return f1Prog, improvement
    # return memory, f1Prog