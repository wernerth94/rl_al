import sys
import getpass
print(F"The user is: {getpass.getuser()}")
print(F"The virtualenv is: {sys.prefix}")

# path additions for the cluster
sys.path.append("core")
sys.path.append("evaluation")
sys.path.append("config")
print(F"updated path is {sys.path}")

from core.Evaluation import scoreAgent
from core.Misc import saveNumpyFile
import numpy as np
import os, time
import argparse
import Classifier, Agent, Environment
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--name', '-n', default='bvssb', type=str)
parser.add_argument('--chkpt', '-c', default='', type=str)
parser.add_argument('--iterations', '-i', type=int, default=10)
parser.add_argument('--samplesize', '-s', type=int)
parser.add_argument('--budget', '-b', type=int)
args = parser.parse_args()

##################################
### MAIN
baselineName = str(args.name)

from config import mockConfig as c
# from config import cifarConfig as c
from Data import load_cifar10_custom as load_data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# overwrite usual config
if args.samplesize:
    print(f"overwrite sample size to {args.samplesize}")
    c.SAMPLE_SIZE = args.samplesize
if args.budget:
    print(f"overwrite budget to {args.budget}")
    c.BUDGET = args.budget

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

envFunc = Environment.MockALGame
dataset = [None]*4
classifier = None

# envFunc = Environment.ALGame
# dataset = load_data(return_tensors=True)
# dataset = [d.to(device) for d in dataset]
# classifier = Classifier.Cifar10ClassifierFactory()
# classifier = Classifier.EmbeddingClassifierFactory(dataset[0].size(1))


print('#########################################################')
print('testing', baselineName, 'with budget', c.BUDGET)
print('#########################################################')

startTime = time.time()

avrgImprov = 0
result = list()
for run in range(args.iterations):
    seed = int(startTime / 100) + run
    print('run %d/%d seed %d' % (run, args.iterations, seed))
    np.random.seed(int(seed))

    env = envFunc(dataset=dataset, modelFunction=classifier, config=c, verbose=0)
    if baselineName == 'bvssb':
        agent = Agent.Baseline_BvsSB()
    elif baselineName == 'entropy':
        agent = Agent.Baseline_Entropy()
    elif baselineName == 'random':
        agent = Agent.Baseline_Random()
    elif baselineName == 'truth':
        agent = Agent.Baseline_Heuristic(m=3)
    elif baselineName == 'agent':
        assert args.chkpt != ''
        # agent = Agent.DDVN(env.stateSpace, gamma=c.AGENT_GAMMA, n_hidden=c.AGENT_NHIDDEN,
        #                    weight_copy_interval=c.AGENT_C)
        path = os.path.join("runs", args.chkpt, "best_agent.pt")
        agent = torch.load(path, map_location=device)
        agent.device = device
        agent.to(device)
    else:
        raise ValueError('baseline not in all_baselines;  given: ' + baselineName)

    f1, improvement = scoreAgent(agent, env)
    avrgImprov += improvement
    result.append(f1)

avrgImprov /= args.iterations
result = np.array(result)
f1 = np.array([np.mean(result, axis=0),
               np.std(result, axis=0)])

folder = 'baselines'
os.makedirs(folder, exist_ok=True)
file = os.path.join(folder, f"{baselineName}_b{c.BUDGET}_s{c.SAMPLE_SIZE}")
saveNumpyFile(file, f1)

print('time needed', int(time.time() - startTime), 'seconds')
print(f"average improvement {avrgImprov}")