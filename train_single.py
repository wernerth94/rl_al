import sys
import getpass
print(F"The user is: {getpass.getuser()}")
print(F"The virtualenv is: {sys.prefix}")

# path additions for the cluster
sys.path.append("core")
sys.path.append("evaluation")
sys.path.append("config")
print(F"updated path is {sys.path}")

import argparse
import Classifier
import Environment
import Agent
import Memory
from Misc import *
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

import config.cifarConfig as c
from Data import load_cifar10_mobilenet,  load_cifar10_custom

parser = argparse.ArgumentParser()
parser.add_argument('--max_epochs', '-m', default=2000, type=int)
parser.add_argument('--warmup_epochs', '-w', default=10, type=int)
parser.add_argument('--batch_size', '-b', default=16, type=int)
args = parser.parse_args()

# dataset = load_cifar10_mobilenet()
dataset = load_cifar10_custom(return_tensors=True)
classifier = Classifier.EmbeddingClassifierFactory(dataset[0].size(1))

env = Environment.ALGame(dataset=dataset, modelFunction=classifier, config=c, verbose=0)
memory = Memory.NStepVMemory(env.stateSpace, 1, maxLength=c.MEMORY_CAP)
agent = Agent.DDVN(env.stateSpace, gamma=c.AGENT_GAMMA, nHidden=c.AGENT_NHIDDEN)

current_time = datetime.now().strftime('%b%d_%H-%M-%S')
log_dir = os.path.join('runs', current_time)
summary_writer = SummaryWriter(log_dir=log_dir)
with open(os.path.join(log_dir, "config.txt"), "w") as f:
    f.write(c.get_description())

total_epochs = 0
with RLEnvLogger(summary_writer, env, print_interval=1) as env:
    with RLAgentLogger(summary_writer, agent) as agent:
        while total_epochs < args.max_epochs:
            done = False
            state = env.reset()
            while not done:
                greed = c.GREED[min(total_epochs, len(c.GREED)-1)]
                q, action = agent.predict(state, greed=greed)
                action = action[0].item()

                new_state, reward, done, _ = env.step(action)
                memory.addMemory(state[action].numpy(), [reward], np.mean(new_state.numpy(), axis=0), done)

                if total_epochs > args.warmup_epochs:
                    sample = memory.sampleMemory(args.batch_size)
                    if len(sample) > 0:
                        agent.fit(sample)
                state = new_state
            total_epochs += 1