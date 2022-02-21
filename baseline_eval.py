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
import Classifier, Agent, Environment

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--name', '-n', default='bvssb', type=str)
parser.add_argument('--iterations', '-i', default=5, type=int)
args = parser.parse_args()

##################################
### MAIN
all_baselines = ['random', 'bvssb', 'entropy']
baselineName = str(args.name)

from config import cifarConfig as c
from Data import load_cifar10_custom

envFunc = Environment.ALGame
dataset = load_cifar10_custom(return_tensors=True)
classifier = Classifier.EmbeddingClassifierFactory(dataset[0].size(1))


print('#########################################################')
print('testing', baselineName, 'with budget', c.BUDGET)
print('#########################################################')

startTime = time.time()

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
    elif baselineName == 'agent':
        agent = Agent.DDVN(env)
    else:
        raise ValueError('baseline not in all_baselines;  given: ' + baselineName)

    f1 = scoreAgent(agent, env)
    result.append(f1)

result = np.array(result)
f1 = np.array([np.mean(result, axis=0),
               np.std(result, axis=0)])

folder = 'baselines'
os.makedirs(folder, exist_ok=True)
file = os.path.join(folder, baselineName + '2_' + str(c.BUDGET))
saveNumpyFile(file, f1)

print('time needed', int(time.time() - startTime), 'seconds')