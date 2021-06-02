import os
import Memory
from Misc import avrg
import tensorflow.keras as K
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from config import mnistConfig as c

memory = Memory.NStepVMemory(3, c.N_STEPS)
assert memory.loadFromDisk(os.path.join('..', c.memDir))
state, rewardList, newState, done = memory.rowsToArgs(memory.memory)

ids = np.arange(len(memory))
split = int(len(memory)*0.8)
np.random.shuffle(ids)
train, test = ids[:split], ids[split:]

x_train = state[train]
y_train = newState[train,0]
x_test = state[test]
y_test = newState[test,0]

model = K.Sequential([
    K.layers.Input(shape=(3)),
    K.layers.Dense(10, activation='tanh'),
    K.layers.Dense(1)
])
model.compile(optimizer=K.optimizers.Adam(), loss=K.losses.MSE)
model.fit(x_train, y_train, epochs=12, batch_size=32)
res = model.evaluate(x_test, y_test, batch_size=128)
print('Val MSE', res)

yHat = np.squeeze(model.predict(x_test, batch_size=128))
residual = yHat - y_test

sns.histplot(residual)
plt.show()

# good accuracy