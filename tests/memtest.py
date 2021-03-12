import Memory, Environment, Data, Classifier
import matplotlib.pyplot as plt
from config import mnistConfig as c
import numpy as np

env = Environment.BatchALGame(Data.loadMNIST(prefix='..'), Classifier.DenseClassifierMNIST, c)
mem = Memory.NStepVMemory(env, 5)
assert mem.loadFromDisk('../'+c.memDir)

state, rewardList, newState, done = mem.sampleMemory(len(mem))

allStates = np.concatenate([state, newState], axis=0)

mean = np.mean(allStates, axis=0)
std = np.std(allStates, axis=0)

allRewards = np.concatenate(rewardList)

rMean = np.mean(allRewards)
rStd = np.std(allRewards)

print()