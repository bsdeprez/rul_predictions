from abc import ABC
from tensorflow import keras
import tensorflow as tf

from RemainingUsefulLifeProject.Baseline.helpermethods import loss_function, np_to_tensor


class FFNModel(keras.Model, ABC):

    def __init__(self, input_size):
        super().__init__()
        self.hidden1 = keras.layers.Dense(40, input_shape=(input_size,))
        self.hidden2 = keras.layers.Dense(40)
        self.out = keras.layers.Dense(1)

    def train_model(self, x, y, epochs=1, lr=0.01, batch_steps=10):
        optimizer = keras.optimizers.Adam(learning_rate=lr)
        for epoch in range(1, epochs + 1):
            losses = []
            total_loss = 0

    def predict(self, x):
        return self(x)

    def __forward__(self, x):
        x = keras.activations.relu(self.hidden1(x))
        x = keras.activations.relu(self.hidden2(x))
        x = self.out(x)
        return x

    def __compute_loss__(self, x, y, loss_fn=loss_function):
        logits = self.__forward__(x)
        mse = loss_fn(y, logits)
        return mse, logits

    def __compute_gradients__(self, x, y, loss_fn=loss_function):
        with tf.GradientTape() as tape:
            loss, _ = self.__compute_loss__(x, y, loss_fn)
        return tape.gradient(loss, self.trainable_variables), loss

    def __train_batch__(self, x, y, optimizer):
        tensor_x, tensor_y = np_to_tensor((x, y))
        gradients, loss = self.__compute_gradients__(tensor_x, tensor_y)
        optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return loss
