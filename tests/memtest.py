
from ReplayBuffer import PrioritizedReplayMemory, DuelingPrioritizedReplay
import numpy as np
from numpy.random import rand
from torch import ones
from memory_profiler import profile

@profile
def t1():
    replay = PrioritizedReplayMemory(cap)
    for _ in range(100):
        replay.push( ( ones(int(cap/4)), ones(int(cap/4)), ones(int(cap/4)), ones(int(cap/4)) ) )
    del replay

@profile
def t2():
    replay = PrioritizedReplayMemory(cap)
    for _ in range(100):
        replay.push((ones(cap)))
    del replay

cap = 1000000
t1()
t2()