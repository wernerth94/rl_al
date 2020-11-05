import sys
import getpass
print(F"The user is: {getpass.getuser()}")
print(F"The virtualenv is: {sys.prefix}")

# path additions for the cluster
sys.path.append("core")
sys.path.append("plotting")
print(F"updated path is {sys.path}")

from core.Evaluation import scoreAgent
import numpy as np
import os

import Data, Classifier, Agent, Environment
from core.Misc import saveFile

all_datasets = ['mnist', 'iris']
all_setups = ['conv', 'dense']
datasetName = str(sys.argv[-1])
setup = str(sys.argv[-2])
if datasetName not in all_datasets: raise ValueError('dataset not in all_datasets;  given: ' + datasetName)
if setup not in all_setups: raise ValueError('setup not in all_setups;  given: ' + setup)

if setup == 'dense':
    import config as c
    envFunc = Environment.ImageClassificationGame
    agentFunc = Agent.DenseAgent
elif setup == 'conv':
    import convConfig as c
    envFunc = Environment.ConvALGame
    agentFunc = Agent.ConvAgent

print('#########################################################')
print('loaded config', c.MODEL_NAME, 'loaded dataset', datasetName)
print('#########################################################')

if datasetName == 'iris':
    dataset = Data.loadIRIS()
    classifier = Classifier.SimpleClassifier
elif datasetName == 'mnist':
    dataset = Data.loadMNIST()
    classifier = Classifier.DenseClassifierMNIST

env = envFunc(dataset=dataset, modelFunction=classifier, config=c, verbose=0)

#agent = agentFunc(env, fromCheckpoints=c.ckptDir)
agent = Agent.Baseline_Random(env)
c.MODEL_NAME = 'random_'+datasetName
c.EVAL_ITERATIONS = 10

lossCurves = []
f1Curves = []

for i in range(c.EVAL_ITERATIONS):
    print('%d ########################'%(i))
    f1, loss = scoreAgent(agent, env, c.BUDGET, printInterval=100)
    if len(f1) == c.BUDGET:
        lossCurves.append(loss)
        f1Curves.append(f1)

#lossCurves = np.array(lossCurves)
f1Curves = np.array(f1Curves)
f1Mean = np.mean(f1Curves, axis=0)
f1Std = np.std(f1Curves, axis=0)
f1 = np.stack([f1Mean, f1Std])

file = os.path.join(c.OUTPUT_FOLDER, c.MODEL_NAME)
saveFile(file + '_f1', f1)
#saveFile(file + '_loss', lossCurves)