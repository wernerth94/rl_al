import sys
import getpass
print(F"The user is: {getpass.getuser()}")
print(F"The virtualenv is: {sys.prefix}")

# path additions for the cluster
sys.path.append("core")
sys.path.append("evaluation")
print(F"updated path is {sys.path}")

from core.Evaluation import scoreAgent
from core.Misc import saveFile
import numpy as np
import os, time
import tensorflow
import Classifier, Agent, Environment


##################################
### MAIN
all_baselines = ['random', 'bvssb', 'entropy']
baselineName = str(sys.argv[1])
sampleSize = int(sys.argv[2])
budget = int(sys.argv[3])
if baselineName not in all_baselines: raise ValueError('baseline not in all_baselines;  given: ' + baselineName)

from config import mnistConfig as c

envFunc = Environment.BatchALGame
from Data import loadMNIST
dataset = loadMNIST()
classifier = Classifier.DenseClassifierMNIST
#classifier = Classifier.EmbeddingClassifier(embeddingSize=1280)


if baselineName == 'bvssb':
    agent = Agent.Baseline_BvsSB()
elif baselineName == 'entropy':
    agent = Agent.Baseline_Entropy()
elif baselineName == 'random':
    agent = Agent.Baseline_Random()

print('#########################################################')
print('testing', baselineName, 'with samplesize', sampleSize)
print('#########################################################')

c.BUDGET = budget
c.GAME_LENGTH = c.BUDGET
c.EVAL_ITERATIONS = 15
c.SAMPLE_SIZE = sampleSize
startTime = time.time()

result = list()
for run in range(c.EVAL_ITERATIONS):
    seed = int(startTime / 100) + run
    print('%d/%d seed %d \t start' % (run, c.EVAL_ITERATIONS, seed))
    tensorflow.random.set_seed(int(seed))
    np.random.seed(int(seed))

    env = envFunc(dataset=dataset, modelFunction=classifier, config=c, verbose=0)
    #agent = agentFunc(env, fromCheckpoints=c.ckptDir)

    f1, loss = scoreAgent(agent, env, c.BUDGET)
    result.append(f1)

result = np.array(result)
f1 = np.array([np.mean(result, axis=0),
               np.std(result, axis=0)])

folder = 'baselines'
os.makedirs(folder, exist_ok=True)
file = os.path.join(folder, baselineName + '_' + str(sampleSize))
saveFile(file, f1)

print('time needed', int(time.time() - startTime), 'seconds')