import torch
import numpy as np
from reimplementations.tian_extends import *
from tianshou.policy import DQNPolicy
import Classifier, Agent, Environment


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_linear_agent(conf):
    env = Environment.MockALGame(conf)
    X = np.zeros(shape=(0, 2))
    Y = np.zeros(shape=(0, 1))
    for _ in range(100):
        state = env.reset()
        X = np.concatenate([X, state[:, 1:3]])
        Y = np.concatenate([Y, state[:, 3:4]])
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(X, Y)

    class LinearAgent():
        def __init__(self, model:LinearRegression):
            self.model = model
        def predict(self, inputs, *args, **kwargs):
            y_hat = self.model.predict(inputs[:, 1:3])
            action = np.argmax(y_hat)
            return y_hat[action], torch.Tensor([int(action)]).int()

    return LinearAgent(model)

def load_tianshou_agent_for_eval(path, env, n_hidden=64, test_eps=0.0):
    # TODO: dont hardcode the hidden sizes and testing epsilon
    net = TianTimeDistributedNet(env.observation_space.shape[0], n_hidden=n_hidden).to(device)
    _ = torch.optim.Adam(net.parameters(), lr=0.0)
    agent = DQNPolicy(net, _, target_update_freq=1)
    agent.load_state_dict(torch.load(path, map_location=device))
    agent.set_eps(test_eps)
    return agent
