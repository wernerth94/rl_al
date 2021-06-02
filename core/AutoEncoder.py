import tensorflow as tf
import tensorflow.keras as K
import tensorflow_probability as tfp

class Sampling(K.layers.Layer):
    """Uses (z_mean, z_log_var) to sample z"""
    def call(self, inputs, **kwargs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon




class VAE(K.Model):
    def __init__(self, stateSpace, latent_dim=24, alpha=1, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.latent_dim = latent_dim
        self.stateSpace = stateSpace
        self.alpha = alpha
        self.encoder = self._buildEncoder(stateSpace, latent_dim)
        self.decoder = self._buildDecoder(stateSpace, latent_dim)
        self.dist = tfp.distributions.Normal(loc=0., scale=1.)
        self.sampler = Sampling()


    def _buildEncoder(self, stateSpace, latent_dim):
        encoder_inputs = K.Input(shape=stateSpace)
        x = K.layers.Dense(12, activation="tanh")(encoder_inputs)
        x = K.layers.Dropout(0.5)(x)
        x = K.layers.Dense(24, activation="tanh")(x)
        x = K.layers.Dropout(0.5)(x)
        z_mean = K.layers.Dense(latent_dim, name="z_mean")(x)
        z_log_var = K.layers.Dense(latent_dim, name="z_log_var")(x)
        z = Sampling()([z_mean, z_log_var])
        encoder = K.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
        # encoder.summary()
        return encoder


    def _buildDecoder(self, stateSpace, latent_dim):
        latent_inputs = K.Input(shape=(latent_dim,))
        x = K.layers.Dense(24, activation="tanh")(latent_inputs)
        x = K.layers.Dropout(0.5)(x)
        x = K.layers.Dense(12, activation="tanh")(x)
        x = K.layers.Dropout(0.5)(x)
        decoder_outputs = K.layers.Dense(stateSpace)(x)
        decoder = K.Model(latent_inputs, decoder_outputs, name="decoder")
        #decoder.summary()
        return decoder


    def sample(self):
        z_mean = tf.zeros((1, self.latent_dim))
        z_log_var = tf.ones((1, self.latent_dim))
        # batch = 1
        # dim = self.latent_dim
        # epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        # return self.decoder( z_mean + tf.exp(0.5 * z_log_var) * epsilon )
        samples = self.sampler((z_mean, z_log_var))
        return self.decoder(samples)


    def call(self, inputs, additionalSamples=0, training=None, mask=None):
        z_mean, z_log_var, z = self.encoder(inputs)
        if additionalSamples > 0:
            additionalZ = self.sampler((tf.repeat(z_mean, additionalSamples, axis=0),
                                        tf.repeat(z_log_var, additionalSamples, axis=0)))
            z = tf.concat([z, additionalZ], axis=0)
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
            total_loss = reconstruction_loss + self.alpha * kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
            'mae': val_mae
        }