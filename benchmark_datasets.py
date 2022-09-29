import sys
import getpass

import torch

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
from torch.utils.data import TensorDataset, DataLoader
from helper_functions import *
from sklearn.metrics import f1_score
from data_loader import load_dataset
from classifier_factory import create_classifier
import Evaluation
import core.Agent

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--heuristic', default='entropy', type=str)
parser.add_argument('--iterations', type=int, default=1)
parser.add_argument('--samplesize', type=int)
args = parser.parse_args()

##################################
### MAIN
heuristic = str(args.heuristic)

# loaded config
from config import mockConfig as c

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CLUSTER = False
if sys.prefix.startswith('/home/werner/miniconda3'):
    CLUSTER = True

# config overwrites from cmd arguments
if args.samplesize:
    print(f"overwrite sample size to {args.samplesize}")
    c.SAMPLE_SIZE = args.samplesize

# Environment class - gets created withing the eval loop
# env_function = Environment.MockALGame
env_function = Environment.ALGame

def test_dataset(dataset, classifier, upper_bound_performance):
    # classifier = classifier.to(device)

    startTime = time()

    avrg_improv = 0
    result = list()
    for run in tqdm(range(args.iterations), disable=CLUSTER):
        seed = int(startTime / 100) + run
        print('run %d/%d seed %d' % (run, args.iterations, seed))
        np.random.seed(int(seed))

        env = env_function(dataset=dataset, classifier_function=classifier, config=c)
        eval_function = Evaluation.score_agent

        if heuristic == 'bvssb':
            agent = Agent.Baseline_BvsSB()
        elif heuristic == 'entropy':
            agent = Agent.Baseline_Entropy()
        elif heuristic == 'random':
            agent = Agent.Baseline_Random()
        else:
            raise ValueError('baseline not in all_baselines;  given: ' + heuristic)

        f1_curve, improvement = eval_function(agent, env, f1_threshold=upper_bound_performance)
        avrg_improv += improvement
        result.append(f1_curve)
        sleep(0.1)  # prevent tqdm printing uglyness

    avrg_improv /= args.iterations
    result = [len(i) for i in result]
    result = np.mean(result)[0]
    print('time needed', int(time() - startTime), 'seconds')

    return result


def get_upper_bound_performance(dataset, classifier)->float:
    # Trains the model based on all available data
    # returns upper bound F1 Score
    class EarlyStop:
        def __init__(self, threshold=0.0001, patience=5):
            self.threshold = threshold
            self.loss_multiplier = 1.0 + threshold
            self.patience = patience
            self.best_loss = np.inf
            self.counter = 0

        def check_stop(self, loss_value):
            if loss_value * self.loss_multiplier < self.best_loss:
                self.best_loss = loss_value
                self.counter = 0
            else:
                if self.counter >= self.patience:
                    return True
                self.counter += 1
            return False

    classifier = classifier.to(device)
    x_train, y_train, x_test, y_test = dataset
    BATCH_SIZE = 128
    train_dataloader = DataLoader(TensorDataset(x_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(TensorDataset(x_test, y_test), batch_size=100, shuffle=False)

    optim = torch.optim.Adam(classifier.parameters(), lr=0.001)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[10, 25], gamma=0.1)
    loss_ce = nn.CrossEntropyLoss()

    early_stop = EarlyStop(patience=3) # TODO
    for epoch in range(50):
        for batch_x, batch_y in tqdm(train_dataloader, disable=CLUSTER):
            yHat = classifier(batch_x)
            loss_val = loss_ce(yHat, torch.argmax(batch_y.long(), dim=1))
            optim.zero_grad()
            loss_val.backward()
            optim.step()
        lr_scheduler.step()

        sum_class = 0.0
        counter = 0
        total = 0.0
        correct = 0.0
        for batch_x, batch_y in test_dataloader:
            yHat = classifier(batch_x)
            # one_hot_y_hat = np.eye(10, dtype='uint8')[torch.argmax(yHat, dim=1).cpu().detach()]
            #one_hot_y_batch = np.eye(10, dtype='uint8')[batch_y.int().cpu().detach()]
            predicted = torch.argmax(yHat, dim=1)
            # _, predicted = torch.max(yHat.data, 1)
            total += batch_y.size(0)
            correct += (predicted == torch.argmax(batch_y, dim=1)).sum().item()
            class_loss = loss_ce(yHat, torch.argmax(batch_y.long(), dim=1))
            sum_class += class_loss.detach().cpu().numpy()
            counter += 1
        sum_class /= counter
        print("%d: classification loss %1.3f - Test F1 %1.3f" % (epoch, sum_class, 100 * correct / total))
        sleep(0.1)  # fix some printing ugliness with tqdm
        if early_stop.check_stop(sum_class):
            print("early stop")
            break
    print("####################################")
    print("final upper bound F1 score %1.3f"%(correct / total))
    print("####################################")
    return correct / total


if __name__ == '__main__':
    dataframe = pd.DataFrame(columns=["dataset", "upper_bound", "threshold", "fraction"])
    list_of_datasets = ["dna", "usps"]
    for id, dataset_name in enumerate(list_of_datasets):
        dataset = load_dataset(dataset_name)
        dataset = [torch.Tensor(d).float() for d in dataset]
        dataset = [d.to(device) for d in dataset]
        x_train = dataset[0]
        y_train = dataset[1]
        c.BUDGET = len(x_train) # override AL budget to be the entire dataset
        classifier = create_classifier(x_train, y_train)
        upper_bound = get_upper_bound_performance(dataset, classifier())
        threshold = test_dataset(dataset, classifier, upper_bound)
        # compute percentage of used data
        fraction = float(threshold) / len(x_train)
        print(f"{dataset_name} \t upper %1.3f threshold %1.3f fraction %1.3f"%(upper_bound, threshold, fraction), flush=True)
        dataframe.loc[id] = [dataset_name, upper_bound, threshold, fraction]
    dataframe.to_csv("result.csv")