#######################################################
# Source: Quintin Fettes
# Github: https://github.com/qfettes/DeepRL-Tutorials
#######################################################

import random
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from core.DataStructures import MinSegmentTree, SumSegmentTree


class ExperienceReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, transition):
        self.memory.append(transition)
        if len(self.memory) > self.capacity:
            del self.memory[0]

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size), None, None

    def __len__(self):
        return len(self.memory)


class PrioritizedReplayMemory(object):
    def __init__(self, size, state_space, alpha=0.6, beta_start=0.4, beta_frames=100000):
        """Create Prioritized Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        alpha: float
            how much prioritization is used
            (0 - no prioritization, 1 - full prioritization)
        See Also
        --------
        ReplayBuffer.__init__
        """
        super().__init__()
        self._storage = []
        self.state_space = state_space
        self._maxsize = size
        self._next_idx = 0

        assert alpha >= 0
        self._alpha = alpha

        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1

        it_capacity = 1
        while it_capacity < size:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0

    def beta_by_frame(self, frame_idx):
        return min(1.0, self.beta_start + frame_idx * (1.0 - self.beta_start) / self.beta_frames)

    def push(self, data):
        if isinstance(data, tuple):
            combined = torch.cat([
                data[0],
                torch.Tensor([data[1][0]]).to(device),
                data[2],
                torch.Tensor([data[3]]).to(device),
            ], dim=0)
            data = combined

        idx = self._next_idx

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha

    def _encode_sample(self, idxes):
        return [self._storage[i] for i in idxes]

    def _sample_proportional(self, batch_size):
        res = []
        for _ in range(batch_size):
            # TODO(szymon): should we ensure no repeats?
            mass = random.random() * self._it_sum.sum(0, len(self._storage) - 1)
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res

    def _reorganize(self, encoded_sample):
        s, r, s_p, d = [], [], [], []
        for i in encoded_sample:
            j = self.state_space
            s.append(i[:j])
            r.append(i[j])
            j += 1
            s_p.append(i[j:j+self.state_space])
            d.append(i[-1])
        encoded_sample = [
            torch.stack(s),
            torch.Tensor(r).T.to(device),
            torch.stack(s_p),
            torch.Tensor(d).to(device),
        ]
        return encoded_sample

    def sample(self, batch_size):
        """Sample a batch of experiences.
        compared to ReplayBuffer.sample
        it also returns importance weights and idxes
        of sampled experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        beta: float
            To what degree to use importance weights
            (0 - no corrections, 1 - full correction)
        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        weights: np.array
            Array of shape (batch_size,) and dtype np.float32
            denoting importance weight of each sampled transition
        idxes: np.array
            Array of shape (batch_size,) and dtype np.int32
            idexes in buffer of sampled experiences
        """

        idxes = self._sample_proportional(batch_size)

        weights = []

        # find smallest sampling prob: p_min = smallest priority^alpha / sum of priorities^alpha
        p_min = self._it_min.min() / self._it_sum.sum()

        beta = self.beta_by_frame(self.frame)
        self.frame += 1

        # max_weight given to smallest prob
        max_weight = (p_min * len(self._storage)) ** (-beta)

        for idx in idxes:
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * len(self._storage)) ** (-beta)
            weights.append(weight / max_weight)
        weights = torch.tensor(weights, device=device, dtype=torch.float)
        encoded_sample = self._encode_sample(idxes)
        encoded_sample = self._reorganize(encoded_sample)
        return encoded_sample, idxes, weights

    def update_priorities(self, idxes, priorities):
        """Update priorities of sampled transitions.
        sets priority of transition at index idxes[i] in buffer
        to priorities[i].
        Parameters
        ----------
        idxes: [int]
            List of idxes of sampled transitions
        priorities: [float]
            List of updated priorities corresponding to
            transitions at the sampled idxes denoted by
            variable `idxes`.
        """
        assert len(idxes) == len(priorities)
        for idx, priority in zip(idxes, priorities):
            assert 0 <= idx < len(self._storage)
            self._it_sum[idx] = (priority + 1e-5) ** self._alpha
            self._it_min[idx] = (priority + 1e-5) ** self._alpha

            self._max_priority = max(self._max_priority, (priority + 1e-5))

    def __len__(self):
        return len(self._storage)


class DuelingPrioritizedReplay(PrioritizedReplayMemory):

    def __init__(self, size, context_space, state_space, alpha=0.6, beta_start=0.4, beta_frames=100000):
        super().__init__(size, state_space, alpha, beta_start, beta_frames)
        self.context_space = context_space

    def push(self, data):
        if isinstance(data, tuple):
            combined = torch.cat([
                data[0],
                data[1],
                torch.Tensor([data[2][0]]).to(device),
                data[3],
                data[4],
                torch.Tensor([data[5]]).to(device),
            ], dim=0)
            data = combined
        super().push(data)

    def _reorganize(self, encoded_sample):
        cntx, s, r, cntx_p, s_p, d = [], [], [], [], [], []
        for i in encoded_sample:
            j = self.context_space
            cntx.append(i[:j])
            s.append(i[j:j+self.state_space])
            j += self.state_space
            r.append(i[j])
            j += 1
            cntx_p.append(i[j:j+self.context_space])
            j += self.context_space
            s_p.append(i[j:j+self.state_space])
            d.append(i[-1])
        encoded_sample = [
            torch.stack(cntx),
            torch.stack(s),
            torch.Tensor(r).T.to(device),
            torch.stack(cntx_p),
            torch.stack(s_p),
            torch.Tensor(d).to(device),
        ]
        return encoded_sample