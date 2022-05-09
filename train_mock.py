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
import Environment
import Agent
from Misc import *
from env_logger import RLEnvLogger
from agent_logger import RLAgentLogger
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from ReplayBuffer import PrioritizedReplayMemory

import config.mockConfig as c

arg_parse = argparse.ArgumentParser()
arg_parse.add_argument("--budget", "-b", type=int, default=1000)
arg_parse.add_argument("--noise", "-n", type=float, default=1)

def run():
    args = arg_parse.parse_args()
    c.BUDGET = args.budget
    c.MAX_EPOCHS = int(c.MIN_INTERACTIONS / args.budget)

    env = Environment.MockALGame(config=c, noise_level=args.noise)
    # agent = Agent.DDVN(env.stateSpace, gamma=c.AGENT_GAMMA, n_hidden=c.AGENT_NHIDDEN,
    #                    weight_copy_interval=c.AGENT_C)
    agent = Agent.LinearVN(env.stateSpace, gamma=c.AGENT_GAMMA, n_hidden=24,
                           weight_copy_interval=c.AGENT_C)
    replay_buffer = PrioritizedReplayMemory(c.MEMORY_CAP, env.stateSpace, c.N_STEPS)

    current_time = datetime.now().strftime('%m-%d_%H:%M:%S.%f')
    log_dir = f"{c.MODEL_NAME}_{current_time}"
    log_dir = os.path.join('runs', log_dir)
    summary_writer = SummaryWriter(log_dir=log_dir)
    with open(os.path.join(log_dir, "config.txt"), "w") as f:
        f.write(c.get_description())
    best_model_file = os.path.join(log_dir, 'best_agent.pt')

    moving_reward = 0
    best_reward = 0
    epoch_treshold = 30
    weight = 1.0 / epoch_treshold
    total_epochs = 0
    with RLEnvLogger(summary_writer, env, print_interval=1, record_al_perf=c.RECORD_AL_PERFORMANCE) as env:
        with RLAgentLogger(summary_writer, agent, checkpoint_interval=1) as agent:
            while total_epochs < c.MAX_EPOCHS:
                epoch_reward = 0
                done = False
                state = env.reset()
                state_buffer = [state]
                reward_buffer = []
                while not done:
                    greed = c.GREED[min(total_epochs, len(c.GREED)-1)]
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
        with open(os.path.join(log_dir, "regret.txt"), "w") as f:
            f.write(f"Budget: {c.BUDGET}\n")
            f.write(f"Noise: {args.noise}\n")
            f.write(f"Regret: {regret}")

if __name__ == '__main__':
    run()
