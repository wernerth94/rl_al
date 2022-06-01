import os
import sys
import getpass

import numpy as np

print(F"The user is: {getpass.getuser()}")
print(F"The virtualenv is: {sys.prefix}")

# path additions for the cluster
sys.path.append("core")
sys.path.append("rl_core")
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
arg_parse.add_argument("--budget", "-b", type=int, default=200)
arg_parse.add_argument("--noise",  "-n", type=float, default=0.0)
arg_parse.add_argument("--alpha",  "-a", type=float, default=0.6)
arg_parse.add_argument("--c",      "-c", type=int, default=500)
arg_parse.add_argument("--nsteps", "-s", type=int, default=1)
arg_parse.add_argument("--interactions", "-i", type=int, default=500000)
args = arg_parse.parse_args()

def run(log_dir):
    c.BUDGET = args.budget
    c.MIN_INTERACTIONS = args.interactions
    c.MAX_EPOCHS = int(args.interactions / args.budget)
    c.AGENT_C = args.c
    c.N_STEPS = args.nsteps

    if c.RECORD_AL_PERFORMANCE:
        baseline_perf = np.load(c.BASELINE_FILE)[0]

    env = Environment.MockALGame(config=c, noise_level=args.noise)
    # agent = Agent.DDVN(env.stateSpace, gamma=c.AGENT_GAMMA, n_hidden=c.AGENT_NHIDDEN,
    #                    weight_copy_interval=c.AGENT_C)
    agent = Agent.LinearVN(env.stateSpace, gamma=c.AGENT_GAMMA, n_hidden=24,
                           weight_copy_interval=c.AGENT_C)
    replay_buffer = PrioritizedReplayMemory(c.MEMORY_CAP, env.stateSpace, c.N_STEPS,
                                            alpha=args.alpha)


    summary_writer = SummaryWriter(log_dir=log_dir)
    with open(os.path.join(log_dir, "config.txt"), "w") as f:
        f.write(c.get_description())
    best_model_file = os.path.join(log_dir, 'best_agent.pt')

    moving_performance = -1
    best_performance = 0
    weight = 0.99 # Tensorboard style averaging
    total_epochs = 0
    with RLEnvLogger(summary_writer, env, c, print_interval=1, record_al_perf=c.RECORD_AL_PERFORMANCE) as env:
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

                if total_epochs == 0:
                    moving_performance = env.env.currentTestF1
                moving_performance = weight * moving_performance + (1 - weight) * env.env.currentTestF1
                if total_epochs > c.CONVERSION_GREED:
                    # only save best agents after the greed is reduced
                    if moving_performance > best_performance:
                        best_performance = moving_performance
                        if os.path.exists(best_model_file):
                            os.remove(best_model_file)
                        torch.save(agent.agent, best_model_file)

                total_epochs += 1
                summary_writer.add_scalar('memory/length', len(replay_buffer), total_epochs)

    if c.RECORD_AL_PERFORMANCE:
        regret = baseline_perf[c.BUDGET -1] - moving_performance
        return regret

if __name__ == '__main__':
    current_time = datetime.now().strftime('%m-%d_%H:%M:%S.%f')
    log_dir = f"{c.MODEL_NAME}_{current_time}"
    log_dir = os.path.join('runs', log_dir)
    regrets = list()
    for r in range(3):
        current_log_dir = os.path.join(log_dir, str(r))
        regrets.append(run(current_log_dir))
    with open(os.path.join(log_dir, "result.txt"), "w") as f:
        f.write(f"Budget: {c.BUDGET}\n")
        f.write(f"Noise: {args.noise}\n")
        f.write(f"Alpha: {args.alpha}\n")
        f.write(f"C: {args.c}\n")
        f.write(f"N-Steps: {args.nsteps}\n")
        f.write(f"LR: {c.LR[0]}\n")
        f.write(f"Interactions: {args.interactions}\n")
        f.write("Regret: %1.4f +- %1.4f\n\n"%(np.mean(regrets), np.std(regrets)))
        f.write(f"Values: \n{regrets}")
