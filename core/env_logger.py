import time
import numpy as np
import torch
from collections import OrderedDict

class RLEnvLogger:

    def __init__(self, writer, env, print_interval, smoothing_window=100, record_al_perf=True):
        self.env = env
        self.print_interval = print_interval
        self.smoothing_window = smoothing_window
        self.writer = writer
        self.record_al_perf = record_al_perf
        if record_al_perf:
            self.al_baseline = np.load(env.config.BASELINE_FILE)[0,:env.budget] # select the mean performance
            self.al_lower_bound = np.load(env.config.LOWER_BOUND_FILE)[0,:env.budget] # select the mean performance

    def reset(self):
        end = time.time()
        if self.epoch_time > 0:
            self.writer.add_scalar('env/time per epoch', end - self.epoch_time, self.current_epoch)
        self.epoch_time = end

        if self.steps_in_epoch > 0:
            self.writer.add_scalar('env/reward', self.current_reward, self.current_epoch)
            self.writer.add_scalar('env/steps per epoch', self.steps_in_epoch, self.current_epoch)
            auc = self.auc / self.env.budget
            self.writer.add_scalar('env/auc', auc, self.current_epoch)

        if self.record_al_perf:
            self._log_al_performance()

        self.writer.flush()

        if self.current_epoch > 0:
            self.epoch_reward_list.append(self.current_reward)
            if self.current_epoch % self.print_interval == 0:
                meanReward = float(np.mean(self.epoch_reward_list[-self.smoothing_window:]))
                print('%d - reward %1.4f steps %d'%(self.current_epoch, meanReward, self.steps_in_epoch))
        self.current_epoch += 1
        self.current_reward = 0
        self.auc = 0
        self.steps_in_epoch = 0
        return self.env.reset()


    def _log_al_performance(self):
        for step in range(len(self.al_baseline)):
            values = {"agent": self.al_performance[step], }
            if self.record_al_perf:
                values["baseline"] = self.al_baseline[step]
                values["lower bound"] = self.al_lower_bound[step]
            self.writer.add_scalars('env/al_performance', values, step)


    def step(self, action):
        new_state, reward, done, _ = self.env.step(action)
        if self.record_al_perf:
            self.al_performance[self.steps_in_epoch] = 0.95 * self.al_performance[self.steps_in_epoch] + \
                                                       0.05 * self.env.currentTestF1
        self.total_steps += 1
        self.steps_in_epoch += 1
        self.current_reward += reward
        self.auc += self.env.currentTestF1
        return new_state, reward, done, _


    def __enter__(self):
        self.steps_in_epoch = 0
        self.total_steps = 0
        self.current_reward = 0
        self.auc = 0
        self.current_epoch = 0
        self.epoch_reward_list = []
        self.epoch_time = -1
        if self.record_al_perf:
            self.al_performance = self.al_baseline.copy() # initialize the moving average with sensible values
        print("this environment will be logged")
        print('== ENVIRONMENT CONFIG ==')
        env_conf = self._get_env_config()
        with open(self.writer.log_dir + '/env_config.txt', 'w') as f:
            f.write(env_conf)
        self.writer.add_text("env_conf", env_conf)
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
