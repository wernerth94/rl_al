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
import Memory


def doEval(args):
    setup, datasetName, dataset, seed, iterations = args
    import tensorflow
    import Classifier, Agent, Environment
    import config.batchConfig as c

    envFunc = Environment.ALGame
    agentFunc = Agent.DDVN

    if datasetName == 'iris':
        classifier = Classifier.SimpleClassifier
    elif datasetName == 'mnist':
        classifier = Classifier.DenseClassifierMNIST
    elif datasetName == 'mnist_mobilenet':
        classifier = Classifier.EmbeddingClassifier(embeddingSize=1280)

    STATE_SPACE = 3 + 2 * dataset[0].shape[1]

    trajectories = list()
    scores = list()
    for run in range(iterations):
        print('seed %d \t start \t %d/%d'%(seed, run, iterations))
        tensorflow.random.set_seed(int(seed+run))
        np.random.seed(int(seed+run))

        env = envFunc(dataset=dataset, modelFunction=classifier, config=c, verbose=0)
        agent = agentFunc(STATE_SPACE, nSteps=5, fromCheckpoints=c.stateValueDir)

        memory, f1 = scoreAgent(agent, env, dataset, greed=0.0, printInterval=200)
        trajectories.append(memory)
        scores.append(f1)
        del env
        gc.collect()
    return (scores, trajectories)

##################################
### MAIN
all_datasets = ['mnist', 'iris', 'mnist_mobilenet']
all_setups = ['conv', 'dense', 'batch']
setup = str(sys.argv[1])
datasetName = str(sys.argv[2])
budget = int(sys.argv[3])
if datasetName not in all_datasets: raise ValueError('dataset not in all_datasets;  given: ' + datasetName)
if setup not in all_setups: raise ValueError('setup not in all_setups;  given: ' + setup)

if setup == 'dense':
    import config.config as c
elif setup == 'conv':
    import config.convConfig as c
elif setup == 'batch':
    import config.batchConfig as c

if datasetName == 'iris':
    from Data import loadIRIS
    dataset = loadIRIS()
elif datasetName == 'mnist':
    from Data import loadMNIST
    dataset = loadMNIST()
elif datasetName == 'mnist_mobilenet':
    from Data import load_mnist_mobilenet
    dataset = load_mnist_mobilenet()

print('#########################################################')
print('loaded config', c.MODEL_NAME, 'loaded dataset', datasetName)
print('#########################################################')

# adjust budget
c.BUDGET = budget
c.GAME_LENGTH = c.BUDGET

# load old model
# c.MODEL_NAME = 'DDQN_MNIST_BATCH'
# c.OUTPUT_FOLDER = 'out'+c.MODEL_NAME
# c.cacheDir = os.path.join(c.OUTPUT_FOLDER, 'cache')
# c.ckptDir = os.path.join(c.cacheDir, 'ckpt')
# assert os.path.exists(c.ckptDir + '.index')

numProcesses = 5
startTime = time.time()
seeds = int(startTime)
seeds = [seeds/i for i in range(1, numProcesses+1)]
with Pool(numProcesses) as pool:
    args = zip([setup]*numProcesses, [datasetName]*numProcesses, [dataset]*numProcesses,
               seeds, [int(c.EVAL_ITERATIONS/numProcesses)]*numProcesses)
    result = pool.map(doEval, args)

trajectories = []
f1Curves = []
for workerResult in result:
    f1Curves.append(workerResult[0][0])
    trajectories.append(workerResult[1][0])
f1Curves = np.array(f1Curves)

folder = os.path.join(c.OUTPUT_FOLDER, 'curves')
os.makedirs(folder, exist_ok=True)
file = os.path.join(folder, str(c.BUDGET) + 'x' + str(c.SAMPLE_SIZE) + '_' + str(int(startTime))[-4:])
saveFile(file, f1Curves)

trajFolder = os.path.join(folder, 'trajectories')
os.makedirs(trajFolder, exist_ok=True)
for i, t in enumerate(trajectories):
    trajPath = os.path.join(trajFolder, str(i))
    t.writeToDisk(trajPath)

print('time needed', int(time.time() - startTime), 'seconds')