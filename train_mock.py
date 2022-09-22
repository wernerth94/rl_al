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
import util
import torch
from env_logger import RLEnvLogger
from agent_logger import RLAgentLogger
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from ReplayBuffer import PrioritizedReplayMemory

import config.mockConfig as c

arg_parse = argparse.ArgumentParser()
arg_parse.add_argument("--noise", type=float, default=0.0)
arg_parse.add_argument("--reward-noise", type=int, default=1)
arg_parse.add_argument("--runs", type=int, default=3)
arg_parse.add_argument("--alpha", type=float)
arg_parse.add_argument("--budget", type=int)
arg_parse.add_argument("--c", type=int)
arg_parse.add_argument("--nsteps", type=int)
arg_parse.add_argument("--interactions", type=int)
arg_parse.add_argument("--nhidden", type=int)
arg_parse.add_argument("--linear", type=int, default=0)
arg_parse.add_argument("--regularization", type=float)
args = arg_parse.parse_args()

def run(log_dir):
    if args.budget: c.BUDGET = args.budget
    if args.interactions:
        c.MIN_INTERACTIONS = args.interactions
        c.MAX_EPOCHS = int(args.interactions / args.budget)
        c.CONVERSION_GREED = int(c.MIN_INTERACTIONS * 0.2 / c.BUDGET)
        c.CONVERSION_LR = int(c.MIN_INTERACTIONS * 0.5 / c.BUDGET)
        c.GREED = util.parameterPlan(0.9, 0.05, warmup=c.WARMUP_EPOCHS, conversion=c.CONVERSION_GREED)
        c.LR = util.parameterPlan(0.01, 0.01, warmup=c.WARMUP_EPOCHS, conversion=c.CONVERSION_LR)
    if args.c: c.AGENT_C = args.c
    if args.nsteps: c.N_STEPS = args.nsteps
    if args.alpha: c.MEMORY_ALPHA = args.alpha
    if args.nhidden: c.AGENT_NHIDDEN = args.nhidden
    if args.regularization: c.AGENT_REG = args.regularization

    print("\n\nUPDATED CONFIG \n===============")
    print(c.get_description())

    if c.RECORD_AL_PERFORMANCE:
        baseline_perf = np.load(c.BASELINE_FILE)[0]

    early_stop_patience = int(0.1 * c.MAX_EPOCHS)
    early_stop_counter = 0

    env = Environment.MockALGame(config=c, noise_level=args.noise, reward_noise=args.reward_noise)
    if args.linear:
        agent = Agent.LinearVN(env.stateSpace, gamma=c.AGENT_GAMMA, n_hidden=c.AGENT_NHIDDEN,
                               weight_copy_interval=c.AGENT_C, weight_decay=c.AGENT_REG)
    else:
        agent = Agent.DDVN(env.stateSpace, gamma=c.AGENT_GAMMA, n_hidden=c.AGENT_NHIDDEN,
                           weight_copy_interval=c.AGENT_C, weight_decay=c.AGENT_REG)
    replay_buffer = PrioritizedReplayMemory(c.MEMORY_CAP, env.stateSpace, c.N_STEPS,
                                            alpha=c.MEMORY_ALPHA)

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

                if total_epochs == 0:
                    moving_performance = env.env.currentTestF1
                moving_performance = weight * moving_performance + (1 - weight) * env.env.currentTestF1
                if total_epochs > c.CONVERSION_GREED:
                    # only save best agents / early stop after the greed is reduced
                    early_stop_counter += 1
                    if early_stop_counter > early_stop_patience:
                        print(f"Early stop after {early_stop_patience} epochs of no improvement")
                    if moving_performance > best_performance:
                        early_stop_counter = 0
                        best_performance = moving_performance
                        if os.path.exists(best_model_file):
                            os.remove(best_model_file)
                        torch.save(agent.agent, best_model_file)

                total_epochs += 1
                summary_writer.add_scalar('memory/length', len(replay_buffer), total_epochs)

    if c.RECORD_AL_PERFORMANCE:
        regret = baseline_perf[c.BUDGET-1] - best_performance
        return regret

if __name__ == '__main__':
    current_time = datetime.now().strftime('%m-%d_%H:%M:%S.%f')
    log_dir = f"{c.MODEL_NAME}_{current_time}"
    log_dir = os.path.join('runs', log_dir)
    regrets = list()
    for r in range(args.runs):
        current_log_dir = os.path.join(log_dir, str(r))
        regrets.append(run(current_log_dir))
    with open(os.path.join(log_dir, "result.txt"), "w") as f:
        f.write(f"Reward-Noise: {args.reward_noise}\n")
        f.write(f"Budget: {c.BUDGET}\n")
        f.write(f"Noise: {args.noise}\n")
        f.write(f"Alpha: {c.MEMORY_ALPHA}\n")
        f.write(f"C: {c.AGENT_C}\n")
        f.write(f"N-Steps: {c.N_STEPS}\n")
        f.write(f"LR: {c.LR[0]}\n")
        f.write(f"Interactions: {c.MIN_INTERACTIONS}\n")
        f.write(f"Gamma: {c.AGENT_GAMMA}\n")
        f.write(f"N-Hidden: {c.AGENT_NHIDDEN}\n")
        f.write(f"Linear: {args.linear}\n")
        f.write("Regularization: %1.6f\n"%(c.AGENT_REG))
        f.write("Regret: %1.4f +- %1.4f\n\n"%( float(np.mean(regrets)), float(np.std(regrets))) )
        f.write(f"Values: \n{regrets}")
