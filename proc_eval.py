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
import os, time, gc
from multiprocessing import Pool
import tensorflow as tf

tf.config.optimizer.set_jit(True)
budget = int(sys.argv[1])
numProcesses = 5

# CONFIG ##################################
import config.mnistConfig as c
# import config.trafficConfig as c
SAVE_TRAJ = False
#c.OUTPUT_FOLDER = 'out_MNIST_BVSSB'
###########################################

def doEval(args):
    dataset, seed, iterations, budget = args
    import tensorflow
    import Classifier, Agent, Environment
    # CONFIG #############################
    import config.mnistConfig as c
    ######################################

    c.BUDGET = budget
    c.GAME_LENGTH = c.BUDGET
    c.SAMPLE_SIZE = 1000

    envFunc = Environment.ALGame
    #agentFunc = Agent.Baseline_Random
    #agentFunc = Agent.Baseline_BvsSB
    agentFunc = Agent.DDVN

    if c.DATASET == 'iris':
        classifier = Classifier.SimpleClassifier
    elif c.DATASET == 'mnist':
        classifier = Classifier.DenseClassifierMNIST
    elif c.DATASET == 'traffic_signs':
        classifier = Classifier.trafficClassifier
    else:
        classifier = Classifier.EmbeddingClassifier(embeddingSize=c.EMBEDDING_SIZE)


    trajectories = list()
    scores = list()
    for run in range(iterations):
        print('seed %d \t start \t %d/%d'%(seed, run, iterations))
        tensorflow.random.set_seed(int(seed+run))
        np.random.seed(int(seed+run))

        env = envFunc(dataset=dataset, modelFunction=classifier, config=c, verbose=0)
        agent = agentFunc(env.stateSpace, gamma=c.AGENT_GAMMA, nHidden=c.AGENT_NHIDDEN, fromCheckpoints=c.stateValueDir)

        memory, f1 = scoreAgent(agent, env, imgsPerStep=1, greed=0.0, printInterval=200)
        trajectories.append(memory)
        scores.append(f1)
        del env
        gc.collect()
    return (scores, trajectories)

##################################
### MAIN

from Data import *
if c.DATASET == 'mnist':
    dataset = loadMNIST()
if c.DATASET == 'traffic_signs':
    dataset = loadTrafficSigns()
else:
    dataset = load_mnist_embedded(c.DATASET)


startTime = time.time()
seeds = int(startTime)
seeds = [seeds/i for i in range(1, numProcesses+1)]
with Pool(numProcesses) as pool:
    args = zip([dataset]*numProcesses, seeds, [int(c.EVAL_ITERATIONS/numProcesses)]*numProcesses, [budget]*numProcesses)
    result = pool.map(doEval, args)

trajectories = []
f1Curves = []
for workerResult in result:
    f1Curves.append(workerResult[0][0])
    trajectories.append(workerResult[1][0])
f1Curves = np.array(f1Curves)

#folder = os.path.join('baselines', 'small')
folder = os.path.join(c.OUTPUT_FOLDER, 'curves')
os.makedirs(folder, exist_ok=True)
file = os.path.join(folder, str(c.BUDGET) + 'x' + str(c.SAMPLE_SIZE) + '_' + str(int(startTime))[-4:])
saveFile(file, f1Curves)

if SAVE_TRAJ:
    trajFolder = os.path.join(folder, 'trajectories')
    os.makedirs(trajFolder, exist_ok=True)
    for i, t in enumerate(trajectories):
        trajPath = os.path.join(trajFolder, str(i))
        t.writeToDisk(trajPath)

print('time needed', int(time.time() - startTime), 'seconds')