import sys
import getpass

import Evaluation

print(F"The user is: {getpass.getuser()}")
print(F"The virtualenv is: {sys.prefix}")

# path additions for the cluster
sys.path.append("core")
sys.path.append("evaluation")
sys.path.append("config")
sys.path.append("rl_core")
sys.path.append("reimplementations")
print(F"updated path is {sys.path}")

from core.Evaluation import score_agent
import os, time
import argparse
from helper_functions import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--name', default='agent', type=str)
parser.add_argument('--chkpt', default='mock/dqn/0923-100146_seed_0/', type=str)
parser.add_argument('--n-hidden', type=int, default=32)
parser.add_argument('--iterations', type=int, default=10)
parser.add_argument('--samplesize', type=int)
parser.add_argument('--budget', type=int)
args = parser.parse_args()

##################################
### MAIN
baseline_name = str(args.name)

from config import mockConfig as c
# from config import cifarConfig as c
from Data import load_cifar10_custom as load_data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# config overwrites from cmd arguments
if args.samplesize:
    print(f"overwrite sample size to {args.samplesize}")
    c.SAMPLE_SIZE = args.samplesize
if args.budget:
    print(f"overwrite budget to {args.budget}")
    c.BUDGET = args.budget

env_function = Environment.MockALGame
# envFunc = Environment.ALGame
dataset = load_data(return_tensors=True)
dataset = [d.to(device) for d in dataset]
classifier = Classifier.EmbeddingClassifierFactory(dataset[0].size(1))

print('#########################################################')
print('testing', baseline_name, 'with budget', c.BUDGET)
print('#########################################################')

startTime = time.time()

avrgImprov = 0
result = list()
for run in range(args.iterations):
    seed = int(startTime / 100) + run
    print('run %d/%d seed %d' % (run, args.iterations, seed))
    np.random.seed(int(seed))

    env = env_function(dataset=dataset, modelFunction=classifier, config=c)
    eval_function = Evaluation.score_agent

    if baseline_name == 'bvssb':
        agent = Agent.Baseline_BvsSB()
    elif baseline_name == 'entropy':
        agent = Agent.Baseline_Entropy()
    elif baseline_name == 'random':
        agent = Agent.Baseline_Random()
    elif baseline_name == 'truth':
        agent = Agent.Baseline_Heuristic(m=3)
    elif baseline_name == 'linear':
        agent = get_linear_agent(c)
    elif baseline_name == 'agent':
        assert args.chkpt != ''
        path = os.path.join("runs", args.chkpt)
        if os.path.exists(os.path.join(path, "best_agent.pt")):
            # own implementation
            path = os.path.join(path, "best_agent.pt")
            agent = torch.load(path, map_location=device)
            agent.device = device
        elif os.path.exists(os.path.join(path, "best_policy.pth")):
            # Tianshou
            path = os.path.join(path, "best_policy.pth")
            agent = load_tianshou_agent_for_eval(path, env, n_hidden=args.n_hidden)
            eval_function = Evaluation.score_tianshou_agent
        else:
            raise ValueError("Agent checkpoint not found")
        agent.to(device)
    else:
        raise ValueError('baseline not in all_baselines;  given: ' + baseline_name)

    f1_curve, improvement = eval_function(agent, env)
    avrgImprov += improvement
    result.append(f1_curve)

avrgImprov /= args.iterations
result = np.array(result)
# convert into mean and std vectors
f1_curve = np.array([np.mean(result, axis=0),
                     np.std(result, axis=0)])

folder = 'baselines'
os.makedirs(folder, exist_ok=True)
if baseline_name == 'agent':
    file = os.path.join(folder, f"{baseline_name}_{args.chkpt.replace('/', '_')}.npy")
else:
    file = os.path.join(folder, f"{baseline_name}.npy")
if os.path.exists(file):
    os.remove(file)
np.save(file, f1_curve)

print('time needed', int(time.time() - startTime), 'seconds')
print(f"average improvement {avrgImprov}")