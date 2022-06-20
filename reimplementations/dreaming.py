import torch.nn as nn
import torch
import numpy as np
from Agent import DDQN


class TimeDistributedAgent(DDQN):

    def network(self, state_space, action_space, n_hidden):
        # Weights are shared between timesteps (see Keras TimeDistributed)
        # Pytorch broadcasting takes care of the time distribution
        # https://stackoverflow.com/questions/61372645/how-to-implement-time-distributed-dense-tdd-layer-in-pytorch
        return nn.Sequential(nn.Linear(state_space, n_hidden),
                             nn.LeakyReLU(),
                             nn.Linear(n_hidden, 1),
                             )

    def predict(self, inputs, greed=1, model='main'):
        q = self._forward(inputs, model)
        q = q.squeeze(dim=-1)
        eps = np.random.rand()
        if greed <= 0 or eps > greed:
            action = torch.argmax(q, dim=1)
        else:
            action = torch.tensor([np.random.randint(self.action_space)])
        return q, action