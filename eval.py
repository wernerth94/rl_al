import sys
import getpass
print(F"The user is: {getpass.getuser()}")
print(F"The virtualenv is: {sys.prefix}")

# path additions for the cluster
sys.path.append("core")
sys.path.append("evaluation")
print(F"updated path is {sys.path}")

from core.Evaluation import scoreAgent
import numpy as np
import matplotlib.pyplot as plt
import os, time, gc
import tensorflow

import Data, Classifier, Agent, Environment
from core.Misc import saveFile

all_datasets = ['mnist', 'iris']
all_setups = ['conv', 'dense', 'batch']
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
elif setup == 'batch':
    import batchConfig as c
    envFunc = Environment.BatchALGame
    agentFunc = Agent.BatchAgent

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

agent = agentFunc(env, fromCheckpoints=c.ckptDir)
# agent.model1._make_predict_function()
# agent.model2._make_predict_function()
# agent = Agent.Baseline_Random(env)
# c.MODEL_NAME = 'random_'+datasetName
# c.EVAL_ITERATIONS = 10

def  doEval(agent, envFunc, dataset, classifier, config, seed):
    try:
        print('################### \t seed %d'%(seed))
        tensorflow.random.set_seed(seed)
        np.random.seed(seed)
        env = envFunc(dataset=dataset, modelFunction=classifier, config=config, verbose=0)
        f1, loss = scoreAgent(agent, env, c.BUDGET, seed, printInterval=100)
        del env
        gc.collect()
        return f1
    except KeyboardInterrupt:
        print('stopped by user')

    exit(0)

startTime = time.time()
seed = int(time.time())
f1Curves = []
for run in range(c.EVAL_ITERATIONS):
    f1Curves.append(doEval(agent, envFunc, dataset, classifier, c, seed+run))
f1Curves = np.array(f1Curves)

folder = os.path.join(c.OUTPUT_FOLDER, 'curves')
os.makedirs(folder, exist_ok=True)
file = os.path.join(folder, str(c.BUDGET) + '_' + str(seed))
saveFile(file, f1Curves)

print('time needed', time.time() - startTime, 'seconds')