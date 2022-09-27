import sys
import getpass

import tqdm

import Evaluation

print(F"The user is: {getpass.getuser()}")
print(F"The virtualenv is: {sys.prefix}")

# path additions for the cluster
sys.path.append("core")
sys.path.append("evaluation")
sys.path.append("config")
sys.path.append("rl_core")
sys.path.append("reimplementations")
sys.path.append("dataset_study")
print(F"updated path is {sys.path}")

import os
from time import time, sleep
import argparse
from tqdm import tqdm
import pandas as pd
from helper_functions import *
from data_loader import load_dataset
from classifier_factory import create_classifier

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--heuristic', default='agent', type=str)
parser.add_argument('--iterations', type=int, default=10)
parser.add_argument('--samplesize', type=int)
parser.add_argument('--budget', type=int)
args = parser.parse_args()

##################################
### MAIN
heuristic = str(args.heuristic)

from config import mockConfig as c
# from config import cifarConfig as c
from Data import load_cifar10_custom as load_data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CLUSTER = False
if sys.prefix.startswith('/home/werner/miniconda3'):
    CLUSTER = True

# config overwrites from cmd arguments
if args.samplesize:
    print(f"overwrite sample size to {args.samplesize}")
    c.SAMPLE_SIZE = args.samplesize
if args.budget:
    print(f"overwrite budget to {args.budget}")
    c.BUDGET = args.budget

# Environment class - gets created withing the eval loop
env_function = Environment.MockALGame
# env_function = Environment.ALGame

def test_dataset(dataset, classifier, upper_bound_performance):
    dataset = [d.to(device) for d in dataset]
    classifier = classifier.to(device)

    startTime = time()

    avrg_improv = 0
    result = list()
    for run in tqdm(range(args.iterations), disable=CLUSTER):
        seed = int(startTime / 100) + run
        print('run %d/%d seed %d' % (run, args.iterations, seed))
        np.random.seed(int(seed))

        env = env_function(dataset=dataset, modelFunction=classifier, config=c)
        eval_function = Evaluation.score_agent

        if heuristic == 'bvssb':
            agent = Agent.Baseline_BvsSB()
        elif heuristic == 'entropy':
            agent = Agent.Baseline_Entropy()
        elif heuristic == 'random':
            agent = Agent.Baseline_Random()
        else:
            raise ValueError('baseline not in all_baselines;  given: ' + heuristic)

        f1_curve, improvement = eval_function(agent, env)
        avrg_improv += improvement
        result.append(f1_curve)
        sleep(0.1)  # prevent tqdm printing uglyness

    avrg_improv /= args.iterations
    result = np.array(result)
    # convert into mean and std vectors
    # f1_curve = np.array([np.mean(result, axis=0),
    #                      np.std(result, axis=0)])
    f1_curve = np.mean(result, axis=0)
    print('time needed', int(time() - startTime), 'seconds')
    print(f"average improvement {avrg_improv}")

    threshold = 0
    for i in range(len(f1_curve)):
        # TODO find the first occurrence that is within 95% upper bound performance
        pass
    return threshold


def get_upper_bound_performance(dataset, classifier)->float:
    # Trains the model based on all available data
    # returns upper bound F1 Score
    pass


if __name__ == '__main__':
    dataframe = pd.DataFrame(columns=["dataset", "upper_bound", "threshold", "fraction"])
    list_of_datasets = ["some", "dataset"]
    for id, dataset_name in enumerate(list_of_datasets):
        dataset = load_dataset(dataset_name)
        classifier = create_classifier(dataset[0])
        upper_bound = get_upper_bound_performance(dataset, classifier)
        threshold = test_dataset(dataset, classifier, upper_bound)
        # compute percentage of used data
        fraction = float(threshold) / len(dataset[0])
        print(f"{dataset_name} \t upper %1.3f threshold %1.3f fraction %1.3f"%(upper_bound, threshold, fraction))
        dataframe.loc[id] = [dataset_name, upper_bound, threshold, fraction]
    dataframe.to_csv("result.csv")