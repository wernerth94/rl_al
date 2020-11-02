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
dataset = str(sys.argv[-1])
setup = str(sys.argv[-2])
if dataset not in all_datasets: raise ValueError('dataset not in all_datasets;  given: ' + dataset)
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
print('loaded config', c.MODEL_NAME, 'loaded dataset', dataset)
print('#########################################################')

if dataset == 'iris':
    dataset = Data.loadIRIS()
    classifier = Classifier.SimpleClassifier
elif dataset == 'mnist':
    dataset = Data.loadMNIST()
    classifier = Classifier.DenseClassifierMNIST

env = envFunc(dataset=dataset, modelFunction=classifier, config=c, verbose=0)

agent = agentFunc(env, fromCheckpoints=c.ckptDir)
#agent = Agent.Baseline_Random(env)
#c.MODEL_NAME = 'random'

lossCurves = []
f1Curves = []

for i in range(c.EVAL_ITERATIONS):
    print('%d ########################' % (i))
    f1, loss = scoreAgent(agent, env, c.BUDGET, printInterval=20)
    if len(f1) == c.BUDGET:
        lossCurves.append(loss)
        f1Curves.append(f1)

lossCurves = np.array(lossCurves)
lossCurves = np.mean(lossCurves, axis=0)
f1Curves = np.array(f1Curves)
f1Curves = np.mean(f1Curves, axis=0)

file = os.path.join(c.OUTPUT_FOLDER, c.MODEL_NAME)
saveFile(file + '_f1', f1Curves)
saveFile(file + '_loss', lossCurves)