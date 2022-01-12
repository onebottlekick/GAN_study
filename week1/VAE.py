import tensorflow as tf


def sampling(args):
    z_mean, z_log_var = args
    epsilon = tf.random.normal(shape=(tf.shape(z_mean)[0], 2), mean=0.0, stddev=1.0)
    return z_mean + tf.exp(z_log_var/2)*epsilon

x = tf.keras.layers.Input(shape=(784, ))
h = tf.keras.layers.Dense(256, activation='relu')(x)
z_mean = tf.keras.layers.Dense(2)(h)
z_log_var = tf.keras.layers.Dense(2)(h)
z = tf.keras.layers.Lambda(sampling)([z_mean, z_log_var])

encoder = tf.keras.Model(x, [z_mean, z_log_var, z])

input_decoder = tf.keras.layers.Input(shape=(2, ))
decoder_h = tf.keras.layers.Dense(256, activation='relu')(input_decoder)
x_decoded = tf.keras.layers.Dense(784, activation='sigmoid')(decoder_h)

decoder = tf.keras.Model(input_decoder, x_decoded)

output_combined = decoder(encoder(x)[2])

kl_loss = -0.5*tf.reduce_sum(1 + z_log_var - tf.exp(z_log_var) - tf.square(z_mean), axis=-1)
VAE = tf.keras.Model(x, output_combined)
VAE.add_loss(tf.reduce_mean(kl_loss)/784.)
VAE.compile(optimizer='rmsprop', loss='binary_crossentropy')


