import torch.nn as nn
from dreaming import TimeDistributedNet

class TianTimeDistributedNet(TimeDistributedNet):

    def forward(self, input, *args, **kwargs):
        logits = super().forward(input, *args, **kwargs )
        logits = logits.squeeze(dim=-1) # get rid of time distributed dimensions
        return logits, None # no hidden state