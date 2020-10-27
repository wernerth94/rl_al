import sys
import getpass
print(F"The user is: {getpass.getuser()}")
print(F"The virtualenv is: {sys.prefix}")

# path additions for the cluster
sys.path.append("core")
sys.path.append("plotting")
sys.path.append("prioritized-experience-replay")
print(F"updated path is {sys.path}")

from core.Evaluation import scoreAgent
import numpy as np
import os

import Data, Classifier, Agent
from core.Environment import ImageClassificationGame
from core.Misc import saveFile
import config as c

dataset = Data.loadMNIST()

env = ImageClassificationGame(dataset=dataset, modelFunction=Classifier.DenseClassifierMNIST, config=c, verbose=0)

#agent = Agent.DDQN(env, fromCheckpoints=c.ckptDir)
agent = Agent.Baseline_Random(env)
c.MODEL_NAME = 'random'

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