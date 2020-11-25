import tensorflow as tf
import tensorflow.keras as K
import matplotlib.pyplot as plt
import numpy as np

from Memory import Memory
from Environment import BatchALGame
import Data, Classifier

import batchConfig as c

dataset = Data.loadMNIST()
classifier = Classifier.DenseClassifierMNIST
env = BatchALGame(dataset, classifier, c)
memory = Memory(env)
assert memory.loadFromDisk(c.memDir)

ae = K.models.Sequential([
    K.layers.Input(shape=env.stateSpace),
    K.layers.Dense(10, activation='tanh'),
    K.layers.Dropout(0.1),
    K.layers.Dense(10, activation='tanh'),
    K.layers.Dropout(0.1),
    K.layers.Dense(env.stateSpace),
])
ae.compile(optimizer=K.optimizers.Adam(learning_rate=0.0005),
           loss=K.losses.mse)

trainHist = ae.fit(memory.state, memory.newState, epochs=50)

plt.plot(trainHist.history['loss'])
plt.show()