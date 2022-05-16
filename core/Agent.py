import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim


class BaseAgent:
    def __init__(self, stateSpace, action_space, clipped, gamma,
                 weight_copy_interval, n_hidden, lr):
        self.clipped = clipped
        self.gamma = gamma
        self.weight_copy_interval = weight_copy_interval

        self.stateSpace = stateSpace
        self.training_steps = 0

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(F'placing model on device {self.device}')
        self.model = self.network(stateSpace, action_space, n_hidden=n_hidden)
        self.model = self.model.to(self.device)
        self.target_model = self.network(stateSpace, action_space, n_hidden=n_hidden)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model = self.target_model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss = nn.SmoothL1Loss()

    def network(self, state_space, action_space, n_hidden):
        raise NotImplementedError()


    def _forward(self, inputs, model='main'):
        if type(inputs) is not torch.Tensor:
            inputs = torch.tensor(inputs)
        if len(inputs.shape) < 2:
            inputs = torch.unsqueeze(inputs, 0)
        inputs = inputs.to(self.device).float()

        if model == 'target':
            return self.target_model(inputs)
        else:
            return self.model(inputs)


    def _apply_update(self, total_loss, lr):
        if lr:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        self.optimizer.zero_grad()
        total_loss.backward()
        if self.clipped:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
        self.optimizer.step()

        if self.training_steps >= self.weight_copy_interval:
            self.target_model.load_state_dict(self.model.state_dict())
            self.training_steps = 0
        self.training_steps += 1


    def to(self, device):
        self.model = self.model.to(device)
        self.target_model = self.target_model.to(device)


class DDVN(BaseAgent):

    def __init__(self, stateSpace, clipped=True, gamma=0.99,
                 lr=0.001, n_hidden=10, weight_copy_interval=3):
        super().__init__(stateSpace, None, clipped, gamma, weight_copy_interval, n_hidden, lr)


    def network(self, state_space, action_space, n_hidden):
        return nn.Sequential(nn.Linear(state_space, 64),
                             nn.LeakyReLU(),
                             nn.Linear(64, n_hidden),
                             nn.LeakyReLU(),
                             nn.Linear(n_hidden, n_hidden),
                             nn.LeakyReLU(),
                             nn.Linear(n_hidden, 1)
                             )


    def predict(self, inputs, greed=1, model='main'):
        v = self._forward(inputs, model)
        eps = np.random.rand()
        if greed <= 0 or eps > greed:
            action = torch.argmax(v, dim=0)
        else:
            action = torch.tensor([np.random.randint(2)])
        return v, action


    def fit(self, memory_batch, weights=None, lr=None, return_priorities=False):
        state = memory_batch[0]
        rewards = memory_batch[1]
        next_states = memory_batch[2]
        dones = memory_batch[3]

        v_hat, _ = self.predict(state)

        with torch.no_grad():
            v_target, _ = self.predict(next_states, model='target')
            # next_action = torch.argmax(v_target, dim=1)
            expected_rewards = v_target[:, 0]

            r_c = torch.zeros(len(state)).to(self.device)
            for i, rew in enumerate(rewards):
                r_c += (self.gamma ** i) * rew

            v = torch.zeros_like(v_hat)
            for b, rew in enumerate(r_c):
                v[b, 0] = rew + (1 - dones[b]) * (self.gamma ** len(rewards)) * expected_rewards[b]

        if weights is None:
            weights = torch.ones(len(v_hat))
        total_loss = torch.sum(weights * torch.squeeze(v_hat - v)**2)
        self._apply_update(total_loss, lr)

        if return_priorities:
            prios = torch.abs(v_hat - v)
            return total_loss, prios
        return total_loss


class LinearVN(DDVN):

    def network(self, state_space, action_space, n_hidden):
        return nn.Sequential(nn.Linear(state_space, n_hidden),
                             nn.LeakyReLU(),
                             nn.Linear(n_hidden, 1)
                             )


class DuelingAgent:
    class ActionHead(nn.Module):
        def __init__(self, latent_space, n_hidden=10):
            super().__init__()
            self.model = nn.Sequential(
                nn.Linear(latent_space, 64),
                nn.LeakyReLU(),
                nn.Linear(64, n_hidden),
                nn.LeakyReLU(),
                nn.Linear(n_hidden, 1) )


        def forward(self, inputs, *args, **kwargs):
            context_embedding, action_embedding = inputs
            if context_embedding.shape[0] != action_embedding.shape[0]:
                context_embedding = context_embedding.repeat_interleave(action_embedding.shape[0], dim=0)
            combined = torch.cat([context_embedding, action_embedding], dim=1)
            return self.model(combined)


    class DuelingNetwork(nn.Module):
        def __init__(self, context_space, state_space, n_hidden=10):
            super().__init__()
            context_lat_size = 64
            action_lat_size = 16
            self.context_enc = nn.Sequential(nn.Linear(context_space, n_hidden),
                                             nn.LeakyReLU(),
                                             nn.Linear(n_hidden, context_lat_size))

            self.action_enc = nn.Sequential(nn.Linear(state_space, n_hidden),
                                            nn.LeakyReLU(),
                                            nn.Linear(n_hidden, action_lat_size))

            self.action_head = DuelingAgent.ActionHead(latent_space=context_lat_size + action_lat_size,
                                               n_hidden=n_hidden)
            self.v_head = nn.Linear(context_lat_size, 1)


        def forward(self, inputs):
            context, state = inputs
            cnxt_emb  = self.context_enc(context)
            state_emb = self.action_enc(state)
            v = self.v_head(cnxt_emb)
            a = self.action_head((cnxt_emb, state_emb))
            a_mean = torch.mean(a, dim=0)
            q = v + (a - a_mean)
            return q



    def __init__(self, state_space, context_space, clipped=False, gamma=0.99,
                 lr=0.001, n_hidden=10, weight_copy_interval=3):
        self.clipped = clipped
        self.gamma = gamma
        self.weight_copy_interval = weight_copy_interval

        self.stateSpace = state_space
        self.training_steps = 0

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(F'placing model on device {self.device}')
        self.build_networks(state_space, context_space, n_hidden)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss = nn.SmoothL1Loss()

    def build_networks(self, state_space, context_space, n_hidden):
        self.model = DuelingAgent.DuelingNetwork(context_space, state_space, n_hidden)
        self.model = self.model.to(self.device)
        self.target_model = DuelingAgent.DuelingNetwork(context_space, state_space, n_hidden)
        self.target_model = self.target_model.to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())


    def _forward(self, inputs, model='main'):
        if model == 'target':
            return self.target_model(inputs)
        else:
            return self.model(inputs)


    def predict(self, inputs, greed=1, model='main'):
        q = self._forward(inputs, model) # TODO
        eps = np.random.rand()
        if greed <= 0 or eps > greed:
            action = torch.argmax(q, dim=1)
        else:
            action = torch.tensor([np.random.randint(2)])
        return q, action


    def _apply_update(self, total_loss, lr):
        if lr:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        self.optimizer.zero_grad()
        total_loss.backward()
        if self.clipped:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
        self.optimizer.step()

        if self.training_steps >= self.weight_copy_interval:
            self.target_model.load_state_dict(self.model.state_dict())
            self.training_steps = 0
        self.training_steps += 1


    def fit(self, memory_batch, weights=None, lr=None, return_priorities=False):
        # TODO
        state = (memory_batch[0], memory_batch[1])
        rewards = memory_batch[2]
        next_states = (memory_batch[3], memory_batch[4])
        dones = memory_batch[5]

        v_hat, _ = self.predict(state)

        with torch.no_grad():
            v_target, _ = self.predict(next_states, model='target')
            expected_rewards_c = v_target[:, 0]

            r_c = torch.zeros(len(state[0])).to(self.device)
            for i, rew in enumerate(rewards):
                r_c += (self.gamma ** i) * rew

            v = torch.zeros_like(v_hat)
            for b, rew in enumerate(r_c):
                v[b, 0] = rew + (1 - dones[b]) * (self.gamma ** len(rewards)) * expected_rewards_c[b]

        if weights is None:
            weights = torch.ones(len(v_hat))
        total_loss = torch.sum(weights * torch.squeeze(v_hat - v)**2)
        self._apply_update(total_loss, lr)

        if return_priorities:
            prios = torch.abs(v_hat - v)
            return total_loss, prios
        return total_loss


class DDQN(BaseAgent):

    def __init__(self, state_space, action_space, clipped=False, gamma=0.99,
                 lr=0.001, n_hidden=10, weight_copy_interval=3):
        super().__init__(state_space, action_space, clipped, gamma, weight_copy_interval, n_hidden, lr)


    def network(self, state_space, action_space, n_hidden):
        return nn.Sequential(nn.Linear(state_space, 64),
                             nn.LeakyReLU(),
                             nn.Linear(64, n_hidden),
                             nn.LeakyReLU(),
                             nn.Linear(n_hidden, action_space)
                             )


    def predict(self, inputs, greed=1, model='main'):
        q = self._forward(inputs, model)
        eps = np.random.rand()
        if greed <= 0 or eps > greed:
            action = torch.argmax(q, dim=1)
        else:
            action = torch.tensor([np.random.randint(2)])
        return q, action


    def fit(self, memory_batch, weights=None, lr=None, return_priorities=False):
        state = memory_batch[0]
        actions = memory_batch[1]
        rewards = memory_batch[2]
        next_states = memory_batch[3]
        dones = memory_batch[4]

        q_hat, _ = self.predict(state)
        if weights is None:
            weights = torch.ones(len(q_hat))
        q_hat = q_hat.gather(1, actions.unsqueeze(1))

        with torch.no_grad():
            q = q_hat.clone()
            q_target, _ = self.predict(next_states, model='target')
            next_action = torch.argmax(q_target, dim=1)

            expected_rewards_c = q_target.gather(1, next_action.unsqueeze(1))

            r_c = torch.zeros(len(state)).to(self.device)
            for i, rew in enumerate(rewards):
                r_c += (self.gamma ** i) * rew # [:, 0]

            for b, rew in enumerate(r_c):
                q[b, 0] = rew + (1 - dones[b]) * (self.gamma ** len(rewards)) * expected_rewards_c[b, 0]

        total_loss = torch.sum(weights * torch.sum(q_hat - q, dim=1)**2)
        self._apply_update(total_loss, lr)

        if return_priorities:
            prios = torch.sum(torch.abs(q_hat - q), dim=1)
            return total_loss, prios
        return total_loss



class Baseline_Entropy:
    def __init__(self, *args, **kwargs):
        pass

    def predict(self, state, greed=0):
        if isinstance(state, torch.Tensor):
            state = state.cpu().numpy()
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
        if isinstance(state, torch.Tensor):
            state = state.cpu().numpy()
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