import sys
import getpass
print(F"The user is: {getpass.getuser()}")
print(F"The virtualenv is: {sys.prefix}")

# path additions for the cluster
sys.path.append("core")
sys.path.append("rl_core")
sys.path.append("evaluation")
sys.path.append("config")
print(F"updated path is {sys.path}")

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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = load_cifar10_custom(return_tensors=True)
classifier = Classifier.EmbeddingClassifierFactory(dataset[0].size(1))
dataset = [d.to(device) for d in dataset]

env = Environment.PALGame(dataset=dataset, classifier_function=classifier, config=c, verbose=0)
replay_buffer = PrioritizedReplayMemory(c.MEMORY_CAP)
agent = Agent.DDQN(env.state_space, env.actionSpace,
                   gamma=c.AGENT_GAMMA, n_hidden=c.AGENT_NHIDDEN)

current_time = datetime.now().strftime('%b%d_%H-%M-%S')
log_dir = os.path.join('runs', f"pal_{current_time}")
summary_writer = SummaryWriter(log_dir=log_dir)
with open(os.path.join(log_dir, "config.txt"), "w") as f:
    f.write(c.get_description())

total_epochs = 0
with RLEnvLogger(summary_writer, env,
                 print_interval=1, record_al_perf=c.RECORD_AL_PERFORMANCE) as env:
    with RLAgentLogger(summary_writer, agent, checkpoint_interval=1) as agent:
        while total_epochs < c.MAX_EPOCHS:
            done = False
            state = env.reset()
            while not done:
                greed = c.GREED[min(total_epochs, len(c.GREED)-1)]
                q, action = agent.predict(state, greed=greed)
                action = action[0].item()

                new_state, reward, done, _ = env.step(action)
                replay_buffer.push( (state.squeeze(), [reward], new_state.squeeze(), done) )

                if total_epochs > c.WARMUP_EPOCHS:
                    lr = c.LR[min(total_epochs, len(c.GREED) - 1)]
                    sample, idxs, weights = replay_buffer.sample(c.BATCH_SIZE)
                    loss, prios = agent.fit(sample, weights, lr=lr, return_priorities=True)
                    replay_buffer.update_priorities(idxs, prios)
                state = new_state
            total_epochs += 1