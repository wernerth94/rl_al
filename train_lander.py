import os
import sys
import getpass
print(F"The user is: {getpass.getuser()}")
print(F"The virtualenv is: {sys.prefix}")

# path additions for the cluster
sys.path.append("core")
sys.path.append("evaluation")
sys.path.append("config")
print(F"updated path is {sys.path}")

import gym
import Agent
from Misc import *
from env_logger import RLEnvLogger
from agent_logger import RLAgentLogger
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from ReplayBuffer import *

import config.landerConfig as c

def run():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    env = gym.make('LunarLander-v2')
    agent = Agent.DDQN(env.observation_space.shape[0], env.action_space.n,
                       gamma=c.AGENT_GAMMA, n_hidden=c.AGENT_NHIDDEN, weight_copy_interval=c.AGENT_C)

    replay_buffer = PrioritizedQReplay(c.MEMORY_CAP, env.observation_space.shape[0], c.N_STEPS)

    current_time = datetime.now().strftime('%m-%d_%H:%M:%S')
    log_dir = os.path.join('runs', f"lander_{current_time}")
    summary_writer = SummaryWriter(log_dir=log_dir)
    with open(os.path.join(log_dir, "config.txt"), "w") as f:
        f.write(c.get_description())
    best_model_file = os.path.join(log_dir, 'best_agent.pt')

    moving_reward = 0
    best_reward = 0
    epoch_treshold = 30
    weight = 1.0 / epoch_treshold
    total_epochs = 0
    with RLEnvLogger(summary_writer, env, c, print_interval=1, record_al_perf=False) as env:
        with RLAgentLogger(summary_writer, agent, checkpoint_interval=1000) as agent:
            while total_epochs < c.MAX_EPOCHS:
                epoch_reward = 0
                done = False
                state = env.reset()
                state = torch.from_numpy(state).to(device)
                state_buffer = [state]
                action_buffer = []
                reward_buffer = []
                while not done:
                    greed = c.GREED[min(total_epochs, len(c.GREED)-1)]
                    q, action = agent.predict(state, greed=greed)
                    action = action[0].item()
                    action_buffer.append(action)

                    new_state, reward, done, _ = env.step(action)
                    new_state = torch.from_numpy(new_state).to(device)
                    state_buffer.append(new_state)
                    reward_buffer.append(reward)
                    epoch_reward += reward

                    if len(reward_buffer) >= c.N_STEPS:
                        replay_buffer.push( (state_buffer.pop(0), action_buffer.pop(0), reward_buffer,
                                             state_buffer[-1], done) )
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

def test(log_dir):
    env = gym.make('LunarLander-v2')
    best_model_file = os.path.join(log_dir, 'agent.pt')
    agent = torch.load(best_model_file)
    total_epochs = 0
    while total_epochs < 100:
        epoch_reward = 0
        done = False
        state = env.reset()
        while not done:
            env.render()
            q, action = agent.predict(state, greed=0.0)
            action = action[0].item()
            new_state, reward, done, _ = env.step(action)
            epoch_reward += reward
            state = new_state

        total_epochs += 1
        print(f"{total_epochs}/100: %1.4f"%epoch_reward)

if __name__ == '__main__':
    # run()
    test("runs/lander_05-16_10:00:22")
    pass
