import shutil

import tensorflow as tf
import tensorflow.keras as K
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from AutoEncoder import VAE
import os

import Memory
from Misc import avrg

from config import mnistConfig as c


memory = Memory.NStepVMemory(3, c.N_STEPS)
# assert memory.loadFromDisk(os.path.join('../experiment_backlog', c.memDir))
assert memory.loadFromDisk(os.path.join('../', c.memDir))

state, rewardList, newState, done = memory.rowsToArgs(memory.memory)
# final validation MAE: 0.0397 -> reconstructions sux

"""
## Train the VAE
"""

ids = np.arange(len(memory))
split = int(len(memory)*0.8)
np.random.shuffle(ids)
train, test = ids[:split], ids[split:]

vae = VAE(state.shape[1], alpha=0.01)
vae.compile(optimizer=K.optimizers.Adam(0.00002))
lr_schedule = K.callbacks.ReduceLROnPlateau(monitor='loss', patience=3)
train_hist = vae.fit(state[train], newState[train], validation_data=(state[test], newState[test]), epochs=100, batch_size=32)
if os.path.exists('vae'):
    shutil.rmtree('vae')
vae.save('vae')
plt.plot(avrg(train_hist.history['reconstruction_loss'], window=7), label='recon loss')
plt.plot(avrg(train_hist.history['mae'], window=7), label='MAE')
sns.set()
plt.ylabel("loss")
plt.legend()
plt.savefig('vae.jpg')
plt.show()

recon = vae.predict(state[test])
test_y = newState[test]
for i in range(5):
    print(test_y[i], '\n', recon[i], '\n')