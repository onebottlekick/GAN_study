import tensorflow as tf
import numpy as np


img_rows = 28
img_cols = 28
img_channels = 1
img_shape = (img_rows, img_cols, img_channels)

latent_dim = 100

generator = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(latent_dim, )),
    tf.keras.layers.Dense(256*7*7),
    tf.keras.layers.Reshape((7, 7, 256)),
    tf.keras.layers.Conv2DTranspose(128, kernel_size=3, strides=2, padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.LeakyReLU(alpha=0.01),
    tf.keras.layers.Conv2DTranspose(64, kernel_size=3, strides=1, padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.LeakyReLU(alpha=0.01),
    tf.keras.layers.Conv2DTranspose(1, kernel_size=3, strides=2, padding='same'),
    tf.keras.layers.Activation('tanh')
])

discriminator = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=img_shape),
    tf.keras.layers.Conv2D(32, kernel_size=3, strides=2, padding='same'),
    tf.keras.layers.LeakyReLU(alpha=0.01),
    tf.keras.layers.Conv2D(64, kernel_size=3, strides=2, padding='same'),
    tf.keras.layers.LeakyReLU(alpha=0.01),
    tf.keras.layers.Conv2D(128, kernel_size=3, strides=2, padding='same'),
    tf.keras.layers.LeakyReLU(alpha=0.01),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

DCGAN = tf.keras.models.Sequential([
    generator,
    discriminator
])