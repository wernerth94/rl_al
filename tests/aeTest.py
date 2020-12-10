import tensorflow as tf
import tensorflow.keras as K
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from Memory import Memory
from Environment import BatchALGame
import Data, Classifier

from config import batchConfig as c

dataset = Data.loadMNIST()
classifier = Classifier.DenseClassifierMNIST
env = BatchALGame(dataset, classifier, c)
memory = Memory(env)
assert memory.loadFromDisk(c.memDir)

# x = memory.state
# y = memory.newState
"""
## Create a sampling layer
"""


class Sampling(K.layers.Layer):
    """Uses (z_mean, z_log_var) to sample z"""
    def call(self, inputs, **kwargs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


"""
## Build the encoder
"""

latent_dim = 2

encoder_inputs = K.Input(shape=env.stateSpace)
x = K.layers.Dense(12, activation="relu")(encoder_inputs)
x = K.layers.Dense(24, activation="relu")(x)
x = K.layers.Dense(8, activation="relu")(x)
z_mean = K.layers.Dense(latent_dim, name="z_mean")(x)
z_log_var = K.layers.Dense(latent_dim, name="z_log_var")(x)
z = Sampling()([z_mean, z_log_var])
encoder = K.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
encoder.summary()

"""
## Build the decoder
"""

latent_inputs = K.Input(shape=(latent_dim,))
x = K.layers.Dense(24, activation="relu")(latent_inputs)
x = K.layers.Dense(12, activation="relu")(x)
decoder_outputs = K.layers.Dense(env.stateSpace)(x)
decoder = K.Model(latent_inputs, decoder_outputs, name="decoder")
decoder.summary()

"""
## Define the VAE as a `Model` with a custom `train_step`
"""


class VAE(K.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def call(self, inputs, training=None, mask=None):
        z_mean, z_log_var, z = self.encoder(inputs)
        return self.decoder(z)

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data[0])
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(tf.sqrt(K.losses.mse(data[1], reconstruction)))
            val_mae = tf.reduce_mean(K.losses.mae(data[1], reconstruction))
            #reconstruction_loss *= 28 * 28
            kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
            kl_loss = tf.reduce_mean(kl_loss)
            kl_loss *= -0.5
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
            'mae': val_mae
        }


"""
## Train the VAE
"""

ids = np.arange(len(memory))
split = int(len(memory)*0.8)
np.random.shuffle(ids)
train, test = ids[:split], ids[split:]

vae = VAE(encoder, decoder)
vae.compile(optimizer=K.optimizers.Adam(0.00001))
lr_schedule = K.callbacks.ReduceLROnPlateau(monitor='loss', patience=3)
train_hist = vae.fit(memory.state[train], memory.newState[train], validation_data=(memory.state[test], memory.newState[test]), epochs=40, batch_size=16)
plt.plot(train_hist.history['reconstruction_loss'])
plt.plot(train_hist.history['kl_loss'])
sns.set()
plt.ylabel("loss")
plt.show()

recon = vae.predict(memory.state[test])
test_y = memory.newState[test]
for i in range(5):
    print();print()
    print(test_y[i], recon[i])