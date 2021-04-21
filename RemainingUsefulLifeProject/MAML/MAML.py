import tensorflow as tf
import tensorflow.keras.backend as K


def get_model(input_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(24, activation='sigmoid',
                              input_shape=(input_size,)),
        tf.keras.layers.Dense(5),
        tf.keras.layers.Dense(3),
        tf.keras.layers.Dense(1)
    ])
    return model


class MAML:

    def __init__(self, input_size, alpha=0.0003, beta=0.0001):
        self.model = get_model(input_size)
        self.loss_object = tf.keras.losses.MeanSquaredError()
        self.inner_loop_optimizer = tf.keras.optimizers.Adam(learning_rate=alpha)
        self.outer_loop_optimizer = tf.keras.optimizers.Adam(learning_rate=beta)

    def inner_loop(self, x_s, y_s, epochs):
        for epoch in range(epochs):
            with tf.GradientTape() as tape:
                predicted = self.model(x_s)
                loss_value = self.loss_object(y_s, predicted)
            if epoch % 20 == 0:
                print("Epoch: {:03d}\t\tLoss: {:.2f}".format(epoch, loss_value.numpy().mean()))
            grads = tape.gradient(loss_value, self.model.trainable_variables)
            self.inner_loop_optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        print("Epoch: Final\tLoss: {:.2f}".format(loss_value.numpy().mean()))

    def outer_loop(self, x_t, y_t):
        with tf.GradientTape() as tape:
            predicted = self.model(x_t)
            loss_value = self.loss_object(y_t, predicted)
        grads = tape.gradient(loss_value, self.model.trainable_variables)
        self.outer_loop_optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        print("Loss after outer loop: {:.2f}".format(loss_value.numpy().mean()))

    def train(self, x_s, y_s, x_t, y_t, epochs):
        for epoch in range(epochs):
            self.inner_loop(x_s, y_s, 5)
            self.outer_loop(x_t, y_t)
