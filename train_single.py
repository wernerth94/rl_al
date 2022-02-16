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
from Misc import *
from env_logger import RLEnvLogger
from agent_logger import RLAgentLogger
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from ReplayBuffer import PrioritizedReplayMemory

import config.cifarConfig as c
from Data import load_cifar10_custom

# Used for Cluster vs Local settings
parser = argparse.ArgumentParser()
parser.add_argument('--record_al_perf', '-a', default=1, type=int)
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# dataset = load_cifar10_mobilenet()
dataset = load_cifar10_custom(return_tensors=True)
classifier = Classifier.EmbeddingClassifierFactory(dataset[0].size(1))
dataset = [d.to(device) for d in dataset]

env = Environment.ALGame(dataset=dataset, modelFunction=classifier, config=c, verbose=0)
#memory = Memory.NStepVMemory(env.stateSpace, 1, maxLength=c.MEMORY_CAP)
replay_buffer = PrioritizedReplayMemory(10000) # 2000
agent = Agent.DDVN(env.stateSpace, lr=0.1, gamma=c.AGENT_GAMMA, nHidden=c.AGENT_NHIDDEN)

current_time = datetime.now().strftime('%b%d_%H-%M-%S')
log_dir = os.path.join('runs', current_time)
summary_writer = SummaryWriter(log_dir=log_dir)
with open(os.path.join(log_dir, "config.txt"), "w") as f:
    f.write(c.get_description())

total_epochs = 0
with RLEnvLogger(summary_writer, env,
                 print_interval=1, record_al_perf=bool(args.record_al_perf)) as env:
    with RLAgentLogger(summary_writer, agent) as agent:
        while total_epochs < c.MAX_EPOCHS:
            done = False
            state = env.reset()
            while not done:
                greed = c.GREED[min(total_epochs, len(c.GREED)-1)]
                q, action = agent.predict(state, greed=greed)
                action = action[0].item()

                new_state, reward, done, _ = env.step(action)
                # memory.addMemory(state[action], [reward], torch.mean(new_state, dim=0), done)

                replay_buffer.push( (state[action], [reward], torch.mean(new_state, dim=0), done) )
                # if len(replay_buffer) > c.BATCH_SIZE:
                #     sample, idxs, weights = replay_buffer.sample(c.BATCH_SIZE)
                #     loss, prios = agent.fit(sample, weights, return_priorities=True)
                #     replay_buffer.update_priorities(idxs, prios)

                if total_epochs > c.WARMUP_EPOCHS:
                    sample, idxs, weights = replay_buffer.sample(c.BATCH_SIZE)
                    loss, prios = agent.fit(sample, weights, return_priorities=True)
                    replay_buffer.update_priorities(idxs, prios)
                state = new_state
            total_epochs += 1