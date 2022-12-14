import os
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

import numpy as np
from datetime import datetime
import torch
from torch.utils.tensorboard import SummaryWriter

import Classifier
import Environment
import Agent
from env_logger import RLEnvLogger
from agent_logger import RLAgentLogger
from ReplayBuffer import PrioritizedReplayMemory

import config.cifarConfig as c
from Data import load_cifar10_custom

def run(log_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # dataset = load_cifar10_mobilenet()
    dataset = load_cifar10_custom(return_tensors=True)
    classifier = Classifier.EmbeddingClassifierFactory(dataset[0].size(1))
    dataset = [d.to(device) for d in dataset]

    env = Environment.ALGame(dataset=dataset, classifier_function=classifier, config=c)
    agent = Agent.DDVN(env.state_space, gamma=c.AGENT_GAMMA, n_hidden=c.AGENT_NHIDDEN,
                       weight_copy_interval=c.AGENT_C, weight_decay=c.AGENT_REG)

    replay_buffer = PrioritizedReplayMemory(c.MEMORY_CAP, env.state_space, c.N_STEPS)

    summary_writer = SummaryWriter(log_dir=log_dir)
    with open(os.path.join(log_dir, "config.txt"), "w") as f:
        f.write(c.get_description())
    best_model_file = os.path.join(log_dir, 'best_agent.pt')

    moving_reward = 0
    best_reward = 0
    epoch_treshold = 30
    weight = 1.0 / epoch_treshold
    total_epochs = 0
    with RLEnvLogger(summary_writer, env, c, print_interval=1, record_al_perf=c.RECORD_AL_PERFORMANCE) as env:
        with RLAgentLogger(summary_writer, agent, checkpoint_interval=1) as agent:
            while total_epochs < c.MAX_EPOCHS:
                epoch_reward = 0
                done = False
                state = env.reset()
                state_buffer = [state]
                reward_buffer = []
                greed = c.GREED[min(total_epochs, len(c.GREED)-1)]
                while not done:
                    q, action = agent.predict(state, greed=greed)
                    action = action[0].item()

                    new_state, reward, done, _ = env.step(action)
                    state_buffer.append(new_state)
                    reward_buffer.append(reward)
                    epoch_reward += reward

                    if len(reward_buffer) >= c.N_STEPS:
                        replay_buffer.push( (state_buffer.pop(0)[action], reward_buffer,
                                             torch.mean(state_buffer[-1], dim=0), done) )
                        reward_buffer.pop(0)

                    if total_epochs > c.WARMUP_EPOCHS:
                        lr = c.LR[min(total_epochs, len(c.LR) - 1)]
                        sample, idxs, weights = replay_buffer.sample(c.BATCH_SIZE)
                        loss, prios = agent.fit(sample, weights, lr=lr, return_priorities=True)
                        replay_buffer.update_priorities(idxs, prios)

                    state = new_state
                total_epochs += 1
                summary_writer.add_scalar('memory/length', len(replay_buffer), total_epochs)

                moving_reward = weight * epoch_reward + (1-weight) * moving_reward
                if total_epochs > epoch_treshold:
                    if moving_reward > best_reward:
                        best_reward = moving_reward
                        if os.path.exists(best_model_file):
                            os.remove(best_model_file)
                        torch.save(agent.agent, best_model_file)

    if c.RECORD_AL_PERFORMANCE:
        baseline_perf = np.load(c.BASELINE_FILE)[0, c.BUDGET-1]
        regret = baseline_perf - moving_reward
        return regret

if __name__ == '__main__':
    current_time = datetime.now().strftime('%m-%d_%H:%M:%S')
    log_dir = os.path.join('runs', f"v_{current_time}")
    values = list()
    for r in range(3):
        current_log_dir = os.path.join(log_dir, str(r))
        values.append(run(current_log_dir))
    with open(os.path.join(log_dir, "regret.txt"), "w") as f:
        f.write(f"Values: {values}\n")
        f.write(f"Regret: {np.mean(values)}+-{np.std(values)}\n=========================\n")
        f.write(f"{c.get_description()}")
