from abc import ABC
from tensorflow.python.keras.layers import Dense
import tensorflow as tf
from tensorflow import keras
import numpy as np
from RemainingUsefulLifeProject.MAML.Models.maml import loss_function


def copy_model(model, x):
    copy = FFNModel(model.input_size, model.learning_rate)
    copy.__forward__(x[0:1])
    copy.set_weights(model.get_weights())
    return copy


class FFNModel(keras.Model, ABC):

    def __init__(self, input_size, lr=0.02):
        super().__init__()
        self.input_size = input_size
        self.learning_rate = lr
        self.optimizer = keras.optimizers.Adam(learning_rate=lr)

        self.hidden1 = Dense(50, input_shape=(input_size,))
        self.hidden2 = Dense(50)
        self.hidden3 = Dense(50)
        self.hidden4 = Dense(50)
        self.hidden5 = Dense(50)
        self.out = Dense(1)

    def __forward__(self, x):
        x = keras.activations.relu(self.hidden1(x))
        x = keras.activations.relu(self.hidden2(x))
        x = keras.activations.relu(self.hidden3(x))
        x = keras.activations.relu(self.hidden4(x))
        x = keras.activations.relu(self.hidden5(x))
        x = self.out(x)
        return x

    def get_model(self):
        return self

    def train(self, x, y, epochs):
        batch_size = 100
        x_batches, y_batches, losses = [], [], []
        self.__forward__(x)  # Initialize the weights.
        while batch_size - 100 < len(x):
            x_batches.append(x[batch_size - 100:batch_size])
            y_batches.append(y[batch_size - 100:batch_size])
            batch_size += 100
        for epoch in range(1, epochs+1):
            epoch_loss = []
            for x_batch, y_batch in zip(x_batches, y_batches):
                with tf.GradientTape() as tape:
                    loss, _ = self.compute_loss(x_batch, y_batch)
                gradients = tape.gradient(loss, self.trainable_variables)
                self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
                epoch_loss.append(loss)
            loss = np.mean(epoch_loss)
            print(" Loss: {}".format(loss))
            losses.append(loss)
        return losses

    def predict(self, x):
        return np.array(self.__forward__(x))

    def compute_loss(self, x, y, loss_fn=loss_function):
        logits = self.__forward__(x)
        mse = loss_fn(y, logits)
        return mse, logits
