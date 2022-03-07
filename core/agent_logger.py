import os
import numpy as np
import torch
from torch import Tensor


def multi_norm(tensors, p = 2, q = 2, normalize = True) -> Tensor:
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



class RLAgentLogger:

    def __init__(self, writer, agent, log_interval=300, checkpoint_interval=-1):
        self.agent = agent
        self.writer = writer
        self.log_interval = log_interval
        self.checkpoint_interval = checkpoint_interval
        self.checkpoint_file = os.path.join(self.writer.log_dir, "agent.pt")


    def predict(self, state, greed=0.1):
        self.log_counter += 1
        self.step += 1
        if self.log_counter % self.log_interval == 0:
            self.writer.add_scalar('agent/greed', greed, self.step)
        q, action = self.agent.predict(state, greed=greed)
        return q, action


    def fit(self, *args, **kwargs):
        self.checkpoint_counter += 1

        ret_val = self.agent.fit(*args, **kwargs)
        if isinstance(ret_val, tuple):
            loss = ret_val[0] # extract the loss
        else:
            loss = ret_val

        if self.log_counter % self.log_interval == 0:
            self.log_counter = 1
            if 'lr' in kwargs:
                self.writer.add_scalar('agent/lr', kwargs['lr'], self.step)
            self.writer.add_scalar('agent/loss', loss, self.step)
            variables = list(self.agent.model.parameters())
            gradients = [w.grad for w in variables]
            self.writer.add_scalar(f"agent/variables", multi_norm(variables), self.step)
            self.writer.add_scalar(f"agent/gradients", multi_norm(gradients), self.step)
            self.writer.flush()

        if self.checkpoint_counter % self.checkpoint_interval == 0:
            self.checkpoint_counter = 0
            if os.path.exists(self.checkpoint_file):
                os.remove(self.checkpoint_file)
            torch.save(self.agent, self.checkpoint_file)

        return ret_val


    def __enter__(self):
        self.step = 0
        self.log_counter = 0
        self.checkpoint_counter = 0
        agent_conf = self._get_agent_config()
        with open(self.writer.log_dir + '/agent_config.txt', 'w') as f:
            f.write(agent_conf)
        self.writer.add_text("agent_conf", agent_conf)
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
