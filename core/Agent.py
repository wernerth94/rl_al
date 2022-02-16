import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim

class DDVN:

    def __init__(self, stateSpace, clipped=False, gamma=0.99, lr=0.001, nHidden=10):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(F'placing model on device {self.device}')

        self.gamma = gamma
        self.stateSpace = stateSpace
        self.clipped = clipped
        self.weight_copy_interval = 3
        self.training_steps = 0

        self.model = self.v_network(stateSpace, n_hidden=nHidden)
        self.model = self.model.to(self.device)
        self.target_model = self.v_network(stateSpace, n_hidden=nHidden)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model = self.target_model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss = nn.SmoothL1Loss()


    def v_network(self, state_space, n_hidden):
        return nn.Sequential(nn.Linear(state_space, n_hidden),
                             nn.LeakyReLU(),
                             nn.Linear(n_hidden, int(n_hidden/2)),
                             nn.LeakyReLU(),
                             nn.Linear(int(n_hidden/2), 1)
                             )


    def predict(self, inputs, greed=1, model='main'):
        if type(inputs) is not torch.Tensor:
            inputs = torch.tensor(inputs)
        if len(inputs.shape) < 2:
            inputs = torch.unsqueeze(inputs, 0)
        inputs = inputs.to(self.device).float()

        if model == 'target':
            v = self.target_model(inputs)
        else:
            v = self.model(inputs)

        eps = np.random.rand()
        if greed <= 0 or eps > greed:
            action = torch.argmax(v, dim=0)
        else:
            action = torch.tensor([np.random.randint(2)])
        return v, action


    def fit(self, memory_batch, weights=None, return_priorities=False):
        state = memory_batch[0]
        rewards = memory_batch[1]
        next_states = memory_batch[2]
        dones = memory_batch[3]

        v_hat, _ = self.predict(state)
        if weights is None:
            weights = torch.zeros(len(v_hat))

        with torch.no_grad():
            v = v_hat.clone()
            v_target, _ = self.predict(next_states, model='target')
            next_action = torch.argmax(v_target, dim=1)  # .squeeze()

            expected_rewards_c = v_target[:, next_action][0]

            r_c = torch.zeros(len(state)).to(self.device)
            for i, rew in enumerate(rewards):
                r_c += (self.gamma ** i) * rew # [:, 0]

            for b, rew in enumerate(r_c):
                v[b, 0] = rew + (1 - dones[b]) * (self.gamma ** len(rewards)) * expected_rewards_c[b]

        total_loss = torch.sum(weights * torch.squeeze(v_hat - v)**2)
        # total_loss = self.loss(v_hat, v)

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
        self.optimizer.step()

        if self.training_steps >= self.weight_copy_interval:
            self.target_model.load_state_dict(self.model.state_dict())
            self.training_steps = 0
        self.training_steps += 1

        if return_priorities:
            prios = torch.abs(v_hat - v)
            return total_loss, prios
        return total_loss



class Baseline_Entropy:
    def __init__(self, *args, **kwargs):
        pass

    def predict(self, state, greed=0):
        scores = state[:, 2]
        if greed <= 0 or np.random.rand() > greed:
            a = np.expand_dims(np.argmax(scores), axis=-1)
            return scores, a
        else:
            i = np.random.randint(len(scores))
            return scores, np.array(i).reshape(-1)


class Baseline_BvsSB:
    def __init__(self, *args, **kwargs):
        pass

    def predict(self, state, greed=0):
        scores = state[:, 1]
        if greed <= 0 or np.random.rand() > greed:
            a = np.expand_dims(np.argmax(scores), axis=-1)
            return scores, a
        else:
            a = np.random.randint(len(scores))
            return scores, np.array(a).reshape(-1)


class Baseline_Random:
    def __init__(self, *args, **kwargs):
        pass

    def predict(self, state, greed=0):
        return None, np.expand_dims(np.random.randint(len(state)), axis=-1)