import tensorflow as tf
import tensorflow.keras as K
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from AutoEncoder import VAE
import os

import Memory

from config import mnistConfig as c


memory = Memory.NStepVMemory(3 + 1280, 5)
assert memory.loadFromDisk(os.path.join('..', c.memDir))

state, rewardList, newState, done = memory.rowsToArgs(memory.memory)


"""
## Train the VAE
"""

ids = np.arange(len(memory))
split = int(len(memory)*0.8)
np.random.shuffle(ids)
train, test = ids[:split], ids[split:]

vae = VAE(3 + 1280)
vae.compile(optimizer=K.optimizers.Adam(0.00001))
lr_schedule = K.callbacks.ReduceLROnPlateau(monitor='loss', patience=3)
train_hist = vae.fit(state[train], newState[train], validation_data=(state[test], newState[test]), epochs=40, batch_size=16)
plt.plot(train_hist.history['reconstruction_loss'])
sns.set()
plt.ylabel("loss")
plt.show()

# recon = vae.predict(state[test])
# test_y = newState[test]
# for i in range(5):
#     print();print()
#    print(test_y[i], recon[i])