import numpy as np
import torch
from tianshou.data import Batch

def score_tianshou_agent(agent, env, print_interval=100):
    data = Batch({
        "obs": torch.unsqueeze(env.reset(), dim=0),
        "info": Batch(),
    })
    f1_prog = []
    i = 0
    done = False
    while not done:
        result = agent(data)
        a = int(result.act[0])
        state_prime, reward, done, _ = env.step(a)
        f1_prog.append(env.current_test_f1)
        data.obs = torch.unsqueeze(state_prime, dim=0)
        if i % print_interval == 0 and len(f1_prog) > 0:
            print('%d | %1.3f'%(i, f1_prog[-1]))
        i += 1

    improvement = f1_prog[-1] - f1_prog[0]
    print('start %1.3f end %1.3f improvement %1.3f'%(f1_prog[0], f1_prog[-1], improvement))
    print("-------------------------------------------------------")
    return f1_prog, improvement


def score_agent(agent, env, print_interval=20, greed=0, f1_threshold=np.inf):
    state = env.reset()
    f1_prog = []
    i = 0
    done = False
    while not done:
        V, a = agent(state, greed=greed)
        a = a[0].item()
        state_prime, reward, done, _ = env.step(a)
        f1_prog.append(env.current_test_f1)
        #memory.addMemory(state[a].cpu().numpy(), [reward], np.mean(statePrime.cpu().numpy(), axis=0), done)
        state = state_prime
        if i % print_interval == 0 and len(f1_prog) > 0:
            print('%d | %1.3f'%(i, f1_prog[-1]))
        if env.current_test_f1 >= 0.95 * f1_threshold:
            break
        i += 1

    improvement = f1_prog[-1] - f1_prog[0]
    print('start %1.3f end %1.3f improvement %1.3f'%(f1_prog[0], f1_prog[-1], improvement), flush=True)
    print("-------------------------------------------------------")
    return f1_prog, improvement
    # return memory, f1Prog