import torch.nn as nn
import torch
import numpy as np
from Agent import DDQN

class TimeDistributedNet(torch.nn.Module):
    def __init__(self, state_space, n_hidden):
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.net = nn.Sequential(nn.Linear(state_space, n_hidden),
                             nn.LeakyReLU(),
                             nn.Linear(n_hidden, 1),)
    def forward(self, input, *args, **kwargs):
        input = torch.as_tensor(input, device=self.device, dtype=torch.float32)
        return self.net(input)


class TimeDistributedAgent(DDQN):

    def network(self, state_space, action_space, n_hidden):
        # Weights are shared between timesteps (see Keras TimeDistributed)
        # Pytorch broadcasting takes care of the time distribution as long as the input is 3D [batch, sample, features]
        # https://stackoverflow.com/questions/61372645/how-to-implement-time-distributed-dense-tdd-layer-in-pytorch
        return TimeDistributedNet(state_space, n_hidden)

    def predict(self, inputs, greed=1, model='main'):
        q = self._forward(inputs, model)
        q = q.squeeze(dim=-1)
        eps = np.random.rand()
        if greed <= 0 or eps > greed:
            action = torch.argmax(q, dim=1)
        else:
            action = torch.tensor([np.random.randint(self.action_space)])
        return q, action