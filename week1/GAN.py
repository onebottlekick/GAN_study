import tensorflow as tf
import numpy as np


img_rows = 28
img_cols = 28
img_channels = 1
img_shape = (img_rows, img_cols, img_channels)

latent_dim = 100

generator = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(latent_dim, )),
    tf.keras.layers.Dense(128, activation='leaky_relu'),
    tf.keras.layers.Dense(int(np.prod(img_shape)), activation='tanh'),
    tf.keras.layers.Reshape(img_shape)
])

discriminator = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=img_shape),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='leaky_relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

GAN = tf.keras.models.Sequential([
    generator,
    discriminator
])

# discriminator -> non trainable
GAN.layers[1].trainable = False
