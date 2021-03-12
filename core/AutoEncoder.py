import tensorflow as tf
import tensorflow.keras as K


class Sampling(K.layers.Layer):
    """Uses (z_mean, z_log_var) to sample z"""
    def call(self, inputs, **kwargs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon




class VAE(K.Model):
    def __init__(self, stateSpace, latent_dim=12, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.latent_dim = latent_dim
        self.stateSpace = stateSpace
        self.encoder = self._buildEncoder(stateSpace, latent_dim)
        self.decoder = self._buildDecoder(stateSpace, latent_dim)


    def _buildEncoder(self, stateSpace, latent_dim):
        encoder_inputs = K.Input(shape=stateSpace)
        x = K.layers.Dense(12, activation="relu")(encoder_inputs)
        x = K.layers.Dense(24, activation="relu")(x)
        z_mean = K.layers.Dense(latent_dim, name="z_mean")(x)
        z_log_var = K.layers.Dense(latent_dim, name="z_log_var")(x)
        z = Sampling()([z_mean, z_log_var])
        encoder = K.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
        # encoder.summary()
        return encoder


    def _buildDecoder(self, stateSpace, latent_dim):
        latent_inputs = K.Input(shape=(latent_dim,))
        x = K.layers.Dense(24, activation="relu")(latent_inputs)
        x = K.layers.Dense(12, activation="relu")(x)
        decoder_outputs = K.layers.Dense(stateSpace)(x)
        decoder = K.Model(latent_inputs, decoder_outputs, name="decoder")
        #decoder.summary()
        return decoder


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