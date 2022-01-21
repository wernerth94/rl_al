import argparse

import torch

import Classifier
import Environment
import Agent
import Memory
from Misc import *
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

import config.cifarConfig as c
from Data import load_cifar10_custom

parser = argparse.ArgumentParser()
parser.add_argument('--max_epochs', '-m', default=5000, type=int)
parser.add_argument('--warmup_epochs', '-w', default=100, type=int)
parser.add_argument('--batch_size', '-b', default=16, type=int)
args = parser.parse_args()

dataset = load_cifar10_custom()
classifier = Classifier.EmbeddingClassifierFactory(c.EMBEDDING_SIZE)

env = Environment.ALGame(dataset=dataset, modelFunction=classifier, config=c, verbose=0)
memory = Memory.NStepVMemory(env.stateSpace, 1, maxLength=c.MEMORY_CAP)
agent = Agent.DDVN(env.stateSpace, gamma=c.AGENT_GAMMA, nHidden=c.AGENT_NHIDDEN)

current_time = datetime.now().strftime('%b%d_%H-%M-%S')
summary_writer = SummaryWriter(log_dir=os.path.join('runs', current_time))

total_epochs = 0
with RLEnvLogger(summary_writer, env, print_interval=100) as env:
    with RLAgentLogger(summary_writer, agent) as agent:
        while total_epochs < args.max_epochs:
            done = False
            state = env.reset()
            while not done:
                greed = 0.1 if total_epochs < 1000 else 0.01
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