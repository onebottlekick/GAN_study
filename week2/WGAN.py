from pickletools import optimize
import tensorflow as tf


img_shape = (28, 28, 1)
latent_dim = 100

generator = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(latent_dim, )),
    tf.keras.layers.Dense(128*7*7, activation='relu'),
    tf.keras.layers.Reshape((7, 7, 128)),
    tf.keras.layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding='same'),
    tf.keras.layers.BatchNormalization(momentum=0.8),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.Conv2DTranspose(64, kernel_size=4, strides=2, padding='same'),
    tf.keras.layers.BatchNormalization(momentum=0.8),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.Conv2D(1, kernel_size=4, padding='same'),
    tf.keras.layers.Activation('tanh'),
])

critic = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(img_shape)),
    tf.keras.layers.Conv2D(16, kernel_size=3, strides=2, padding="same"),
    tf.keras.layers.LeakyReLU(alpha=0.2),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Conv2D(32, kernel_size=3, strides=2, padding="same"),
    tf.keras.layers.BatchNormalization(momentum=0.8),
    tf.keras.layers.LeakyReLU(alpha=0.2),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Conv2D(64, kernel_size=3, strides=2, padding="same"),
    tf.keras.layers.BatchNormalization(momentum=0.8),
    tf.keras.layers.LeakyReLU(alpha=0.2),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Conv2D(128, kernel_size=3, strides=1, padding="same"),
    tf.keras.layers.BatchNormalization(momentum=0.8),
    tf.keras.layers.LeakyReLU(alpha=0.2),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1)
])

def wasserstein_loss(y_true, y_pred):
    return tf.reduce_mean(y_true*y_pred)


critic.compile(loss=wasserstein_loss, optimizer=tf.keras.optimizers.RMSprop(0.00005), metrics=['accuracy'])

critic.trainable = False
WGAN = tf.keras.models.Sequential([
    generator,
    critic
])
WGAN.compile(loss=wasserstein_loss, optimizer=tf.keras.optimizers.RMSprop(0.00005), metrics=['accuracy'])