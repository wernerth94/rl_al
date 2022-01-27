import os, gc
import numpy as np
import json
from collections import OrderedDict
import torch
from torch import Tensor

def multi_norm(
    tensors, p = 2, q = 2, normalize = True
) -> Tensor:
    r"""Return the (scaled) p-q norm of the gradients.

    Parameters
    ----------
    tensors: list[Tensor]
    p: float, default: 2
    q: float, default: 2
    normalize: bool, default: True
        If true, accumulate with mean instead of sum

    Returns
    -------
    Tensor
    """
    if len(tensors) == 0:
        return torch.tensor(0.0)

    # TODO: implement special cases p,q = ±∞
    if normalize:
        # Initializing s this way automatically gets the dtype and device correct
        s = torch.mean(tensors.pop() ** p) ** (q / p)
        for x in tensors:
            s += torch.mean(x ** p) ** (q / p)
        return (s / (1 + len(tensors))) ** (1 / q)
    # else
    s = torch.sum(tensors.pop() ** p) ** (q / p)
    for x in tensors:
        s += torch.sum(x ** p) ** (q / p)
    return s ** (1 / q)


def accuracy(yHat, labels):
    yHat = torch.argmax(yHat, dim=1)
    labels = torch.argmax(labels, dim=1)
    correct = yHat == labels
    acc = torch.sum(correct) / len(yHat)
    return acc.numpy()


def trainTestIDSplit(length, cutoff=0.8):
    ids = np.arange(length)
    split = int(length * cutoff)
    np.random.shuffle(ids)
    return ids[:split], ids[split:]

def avrg(curve, window):
    if len(curve) <= 0:
        return [0]
    if len(curve) < 2:
        return [curve[0]]
    avrgCurve = []

    for i in range(len(curve)):
        avrgCurve.append(np.mean( curve[max(0, i - int(window/2)) : min(len(curve), i + int(window/2))] ))
    return avrgCurve


def saveFile(name, file):
    if os.path.exists(name + '.npy'):
        os.remove(name + '.npy')
    np.save(name + '.npy', file)


def parameterPlan(val1, val2, warmup, conversion):
    plan1 = np.full(warmup, val1)
    plan2 = np.linspace(val1, val2, conversion)
    return np.concatenate([plan1, plan2])

def asympParameterPlan(val1, val2, warmup, conversion):
    plan1 = np.full(warmup, val1)
    def f(x):
        return -np.square(x-1) + 1
    space = np.linspace(0, 1, conversion)
    plan2 = (1-f(space)) * val1  +  f(space) * val2
    return np.concatenate([plan1, plan2])


def checkStopSwitch():
    return os.path.exists("config/stopSwitch")

def createStopSwitch():
    if not os.path.exists("config/stopSwitch"):
        open("config/stopSwitch", 'a').close()


def saveTrainState(config, state:dict):
    file = os.path.join(config.OUTPUT_FOLDER, config.MODEL_NAME+'.json')
    if os.path.exists(file):
        os.remove(file)
    json.dump(state, open(file, 'w'))


def loadTrainState(config, path_prefix=None):
    file = os.path.join(config.OUTPUT_FOLDER, config.MODEL_NAME+'.json')
    if path_prefix is not None:
        file = os.path.join(path_prefix, file)
    if os.path.exists(file):
        return json.load(open(file))
    return None


class RLAgentLogger:

    def __init__(self, writer, agent):
        self.agent = agent
        self.writer = writer


    def predict(self, state, greed=0.1):
        q, action = self.agent.predict(state, greed=greed)
        return q, action


    def fit(self, sample):
        loss = self.agent.fit(sample)
        self.writer.add_scalar('agent/loss', loss, self.step)

        variables = list(self.agent.model.parameters())
        gradients = [w.grad for w in variables]
        self.writer.add_scalar(f"agent/variables", multi_norm(variables), self.step)
        self.writer.add_scalar(f"agent/gradients", multi_norm(gradients), self.step)
        self.writer.flush()

        self.step += 1
        return loss


    def __enter__(self):
        self.step = 1
        agent_conf = self._get_agent_config()
        with open(self.writer.log_dir + '/agent_config.txt', 'w') as f:
            f.write(agent_conf)
        return self


    def __exit__(self, type, value, traceback):
        try:
            self.writer.flush()
            self.writer.close()
        except:
            pass


    def _get_agent_config(self):
        res = ''
        for attr in dir(self.agent):
            if not attr.startswith('__'):
                value = getattr(self.agent, attr)
                if not callable(value):
                    if type(value) == np.ndarray:
                        value = F'ndarray {value.shape}'
                    res += F'{attr}: {value} \n'
        return res



class RLEnvLogger:

    def __init__(self, writer, env, print_interval, smoothing_window=100):
        self.env = env
        self.print_interval = print_interval
        self.smoothing_window = smoothing_window
        self.writer = writer

    def reset(self):
        self.writer.add_scalar('env/reward', self.current_reward, self.current_epoch)
        self.writer.add_scalar('env/steps per epoch', self.steps_in_epoch, self.current_epoch)
        self.writer.flush()

        self.current_epoch += 1
        self.epoch_reward_list.append(self.current_reward)
        if self.current_epoch % self.print_interval == 0:
            meanReward = float(np.mean(self.epoch_reward_list[-self.smoothing_window:]))
            print('%d - reward %1.4f steps %d'%(self.current_epoch, meanReward, self.steps_in_epoch))
        self.current_reward = 0
        self.steps_in_epoch = 0
        return self.env.reset()


    def step(self, action):
        new_state, reward, done, _ = self.env.step(action)
        self.total_steps += 1
        self.steps_in_epoch += 1
        self.current_reward += reward
        return new_state, reward, done, _


    def __enter__(self):
        self.steps_in_epoch = 0
        self.total_steps = 0
        self.current_reward = 0
        self.current_epoch = 1
        self.epoch_reward_list = []
        print("this environment will be logged")
        print('== ENVIRONMENT CONFIG ==')
        env_conf = self._get_env_config()
        with open(self.writer.log_dir + '/env_config.txt', 'w') as f:
            f.write(env_conf)
        print(env_conf)
        return self


    def __exit__(self, type, value, traceback):
        try:
            self.writer.flush()
            self.writer.close()
        except:
            pass

    def _get_env_config(self)->str:
        res = ''
        for attr in dir(self.env):
            if not attr.startswith('__'):
                value = getattr(self.env, attr)
                if not callable(value):
                    if type(value) == np.ndarray:
                        value = F'ndarray {value.shape}'
                    elif type(value) == torch.Tensor:
                        value = F'Tensor {value.size()}'
                    elif type(value) == tuple:
                        value = F'tuple {len(value)}'
                    elif type(value) == OrderedDict:
                        value = F'OrderedDict {value.keys()}'

                    value = str(value)
                    if len(value) > 150:
                        value = value[:150] + ' ...'
                    res += F'{attr}: {value} \n'
        return res
