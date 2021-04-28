from abc import ABC

from tensorflow import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import time


def loss_function(y_pred, y):
    return keras.backend.mean(keras.losses.mean_squared_error(y, y_pred))


def np_to_tensor(list_of_numpy_objs):
    return (tf.convert_to_tensor(obj) for obj in list_of_numpy_objs)


class SineModel(keras.Model, ABC):

    def __init__(self):
        super().__init__()
        self.hidden1 = keras.layers.Dense(40, input_shape=(1,))
        self.hidden2 = keras.layers.Dense(40)
        self.out = keras.layers.Dense(1)

    def forward(self, x):
        x = keras.activations.relu(self.hidden1(x))
        x = keras.activations.relu(self.hidden2(x))
        x = self.out(x)
        return x

    def compute_loss(self, x, y, loss_fn=loss_function):
        logits = self.forward(x)
        mse = loss_fn(y, logits)
        return mse, logits

    def compute_gradients(self, x, y, loss_fn=loss_function):
        with tf.GradientTape() as tape:
            loss, _ = self.compute_loss(x, y, loss_fn)
        return tape.gradient(loss, self.trainable_variables), loss

    def train_batch(self, x, y, optimizer):
        tensor_x, tensor_y = np_to_tensor((x, y))
        gradients, loss = self.compute_gradients(tensor_x, tensor_y)
        self.apply_gradients(optimizer, gradients, self.trainable_variables)
        return loss

    def train_model(self, dataset, epochs=1, lr=0.001, log_steps=1000):
        optimizer = keras.optimizers.Adam(learning_rate=lr)
        for epoch in range(1, epochs + 1):
            losses = []
            total_loss = 0
            start = time.time()
            for i, sinusoid_generator in enumerate(dataset):
                x, y = sinusoid_generator.batch()
                loss = self.train_batch(x, y, optimizer)
                total_loss += loss
                curr_loss = total_loss / (i + 1.0)
                losses.append(curr_loss)

                if i % log_steps == 0 and i > 0:
                    print("Step {}: loss = {:.5f}, Time to run {} steps = {:.2f} seconds".format(
                        i, curr_loss, log_steps, time.time() - start
                    ))
                    start = time.time()
            plt.plot(losses)
            plt.title("Loss Vs Time steps")
            plt.show()

    @staticmethod
    def apply_gradients(optimizer, gradients, variables):
        optimizer.apply_gradients(zip(gradients, variables))
