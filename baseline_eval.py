import sys
import getpass
print(F"The user is: {getpass.getuser()}")
print(F"The virtualenv is: {sys.prefix}")

# path additions for the cluster
sys.path.append("core")
sys.path.append("evaluation")
print(F"updated path is {sys.path}")

from core.Evaluation import scoreAgent
from core.Misc import saveNumpyFile
import numpy as np
import os, time
import tensorflow
import Classifier, Agent, Environment


##################################
### MAIN
all_baselines = ['random', 'bvssb', 'entropy']
baselineName = 'bvssb'
if baselineName not in all_baselines: raise ValueError('baseline not in all_baselines;  given: ' + baselineName)

from config import cifarConfig as c
from Data import load_cifar10_custom

envFunc = Environment.ALGame
dataset = load_cifar10_custom(return_tensors=True)
classifier = Classifier.EmbeddingClassifierFactory(dataset[0].size(1))

if baselineName == 'bvssb':
    agent = Agent.Baseline_BvsSB()
elif baselineName == 'entropy':
    agent = Agent.Baseline_Entropy()
elif baselineName == 'random':
    agent = Agent.Baseline_Random()

print('#########################################################')
print('testing', baselineName, 'with samplesize', c.SAMPLE_SIZE)
print('#########################################################')

c.EVAL_ITERATIONS = 5
startTime = time.time()

result = list()
for run in range(c.EVAL_ITERATIONS):
    seed = int(startTime / 100) + run
    print('%d/%d seed %d \t start' % (run, c.EVAL_ITERATIONS, seed))
    tensorflow.random.set_seed(int(seed))
    np.random.seed(int(seed))

    env = envFunc(dataset=dataset, modelFunction=classifier, config=c, verbose=0)
    #agent = agentFunc(env, fromCheckpoints=c.ckptDir)

    f1 = scoreAgent(agent, env)
    result.append(f1)

result = np.array(result)
f1 = np.array([np.mean(result, axis=0),
               np.std(result, axis=0)])

folder = 'baselines'
os.makedirs(folder, exist_ok=True)
file = os.path.join(folder, baselineName + '_' + str(c.BUDGET))
saveNumpyFile(file, f1)

print('time needed', int(time.time() - startTime), 'seconds')