import tensorflow as tf


encoder = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(28, 28)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(32)
])

decoder = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(32, )),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(784),
    tf.keras.layers.Reshape((28, 28), input_shape=(784, ))
])

AE = tf.keras.models.Sequential([
    encoder,
    decoder
])

if __name__ == '__main__':
    (X_train, _), (_, _) = tf.keras.datasets.mnist.load_data()
    AE.compile(loss='mse', optimizer='adam')
    AE.fit(X_train, X_train, shuffle=True, epochs=10, batch_size=32)
    AE.save('models/AE.h5')