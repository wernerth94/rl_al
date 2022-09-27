import sys
sys.path.append("core")
sys.path.append("rl_core")
sys.path.append("evaluation")
sys.path.append("config")
sys.path.append("reimplementations")
print(F"updated path is {sys.path}")


import argparse
import os
import pprint
import datetime

import gym
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import Classifier
import Environment
from Data import load_cifar10_custom

from tianshou.data import Collector, VectorReplayBuffer, PrioritizedVectorReplayBuffer
from tianshou.env import DummyVectorEnv, SubprocVectorEnv
from tianshou.policy import DQNPolicy
from tianshou.trainer import offpolicy_trainer
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import Net

from reimplementations.tian_extends import TianTimeDistributedNet

def get_args():
    parser = argparse.ArgumentParser()
    # the parameters are found by Optuna
    parser.add_argument('--task', type=str, default="mock") # mock,rl_al
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--epoch', type=int, default=1000)
    parser.add_argument('--gamma', type=float, default=0.0)
    parser.add_argument('--n-step', type=int, default=1)
    parser.add_argument('--eps-test', type=float, default=0.01)
    parser.add_argument('--eps-train', type=float, default=1.0)
    parser.add_argument('--eps-train-final', type=float, default=0.1)
    parser.add_argument('--eps-anneal', type=int, default=1e6) # Lander: 1M
    parser.add_argument('--buffer-size', type=int, default=int(7e+4)) # 100000
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--beta', type=float, default=0.4)
    parser.add_argument('--beta-final', type=float, default=1.)
    parser.add_argument('--beta-anneal-step', type=int, default=5000000) # 5M
    parser.add_argument('--lr', type=float, default=0.001) # Lander: 0.013
    parser.add_argument('--target-update-freq', type=int, default=500)
    parser.add_argument('--step-per-epoch', type=int, default=10000) # should correspond to 'training-num'
    parser.add_argument('--step-per-collect', type=int, default=10)
    parser.add_argument('--update-per-step', type=float, default=0.1)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--hidden-sizes', type=int, nargs='*', default=[32,]) # [128,]
    parser.add_argument('--training-num', type=int, default=1) # should correspond to 'step-per-epoch'
    parser.add_argument('--test-num', type=int, default=1)
    # Env Config
    parser.add_argument('--BUDGET', type=int, default=2000)
    parser.add_argument('--REWARD_SCALE', type=float, default=1.0)
    parser.add_argument('--REWARD_SHAPING', type=bool, default=True)
    parser.add_argument('--CLASS_FROM_SCRATCH', type=bool, default=True)
    parser.add_argument('--INIT_POINTS_PER_CLASS', type=int, default=1)
    parser.add_argument('--SAMPLE_SIZE', type=int, default=20)
    # Mock Config
    parser.add_argument('--MAX_REWARD', type=float, default=0.0005)
    # Other
    parser.add_argument('--logdir', type=str, default='runs')
    parser.add_argument('--render', type=float, default=0.)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()


def make_env(args):
    if args.task == "rl_al":
        dataset = load_cifar10_custom(return_tensors=True)
        classifier = Classifier.EmbeddingClassifierFactory(dataset[0].size(1))
        dataset = [d.to(args.device) for d in dataset]
        env = Environment.ALGame(dataset=dataset, classifier_function=classifier, config=args)
        return env
    elif args.task == "mock":
        return Environment.MockALGame(args)
    else:
        raise ValueError(f"Unknown environment: {args.task}")

def test_dqn(args=get_args()):
    env = make_env(args)
    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n
    train_envs = DummyVectorEnv( [lambda: make_env(args) for _ in range(args.training_num)] )
    test_envs = DummyVectorEnv( [lambda: make_env(args) for _ in range(args.test_num)] )
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)
    # model
    # net = Net(args.state_shape[0], args.action_shape,
    #           hidden_sizes=args.hidden_sizes, device=args.device ).to(args.device)
    net = TianTimeDistributedNet(args.state_shape[0], args.hidden_sizes[0]).to(args.device)
    optim = torch.optim.Adam(net.parameters(), lr=args.lr)
    policy = DQNPolicy(net, optim, args.gamma, args.n_step, target_update_freq=args.target_update_freq)
    # collector
    buffer = PrioritizedVectorReplayBuffer(args.buffer_size, buffer_num=len(train_envs),
                                           alpha=args.alpha, beta=args.beta)
    train_collector = Collector(policy, train_envs, buffer )
    test_collector = Collector(policy, test_envs )
    train_collector.collect(n_step=args.batch_size * args.training_num)
    # log
    log_path = os.path.join(
        args.logdir, args.task, 'dqn',
        f'{datetime.datetime.now().strftime("%m%d-%H%M%S")}_seed_{args.seed}'
    )
    writer = SummaryWriter(log_path)
    logger = TensorboardLogger(writer)
    with open(os.path.join(log_path, "args.txt"), "w") as f:
        f.write(str(args.__dict__).replace(", '", "\n'"))

    def save_fn(policy, file_name='best_policy.pth'):
        torch.save(policy.state_dict(), os.path.join(log_path, file_name))

    def stop_fn(mean_rewards):
        return False
        # return mean_rewards >= env.spec.reward_threshold

    def train_fn(epoch, env_step):
        # set epsilon
        # eps = max(args.eps_train * (1 - 5e-6)**env_step, args.eps_test)
        if env_step <= args.eps_anneal:
            eps = args.eps_train - env_step / args.eps_anneal * \
                (args.eps_train - args.eps_train_final)
        else:
            eps = args.eps_train_final
        policy.set_eps(eps)
        if env_step % 5000 == 0:
            logger.write("train/env_step", env_step, {"train/eps": eps})

        # set beta
        if isinstance(buffer, PrioritizedVectorReplayBuffer):
            if env_step <= args.beta_anneal_step:
                beta = args.beta - env_step / args.beta_anneal_step * \
                       (args.beta - args.beta_final)
            else:
                beta = args.beta_final
            buffer.set_beta(beta)
            if env_step % 5000 == 0:
                logger.write("train/env_step", env_step, {"train/beta": beta})

    def test_fn(epoch, env_step):
        policy.set_eps(args.eps_test)

    # trainer
    try:
        result = offpolicy_trainer(
            policy,
            train_collector,
            test_collector,
            args.epoch,
            args.step_per_epoch,
            args.step_per_collect,
            args.test_num,
            args.batch_size,
            update_per_step=args.update_per_step,
            stop_fn=stop_fn,
            train_fn=train_fn,
            test_fn=test_fn,
            save_best_fn=save_fn,
            logger=logger
        )
        pprint.pprint(result)
    except KeyboardInterrupt:
        print("Training stopped via KeyboardInterrupt")

    save_fn(policy, file_name="final_policy.pth")
    print("Testing Agent")
    policy.eval()
    policy.set_eps(args.eps_test)
    test_collector.reset()
    result = test_collector.collect(n_episode=args.test_num, render=False)
    rew = result["rews"].mean()
    print(f'Mean reward (over {result["n/ep"]} episodes): {rew}')
    pprint.pprint(result)


if __name__ == '__main__':
    test_dqn(get_args())
