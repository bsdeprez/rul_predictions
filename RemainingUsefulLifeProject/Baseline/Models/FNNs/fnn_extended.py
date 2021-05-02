from abc import ABC
from tensorflow import keras
import tensorflow as tf
import math
import numpy as np
from tqdm import tqdm


class FFNModel(keras.Model, ABC):

    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        self.hidden1 = keras.layers.Dense(24, input_shape=(input_size,))
        self.hidden2 = keras.layers.Dense(100)
        self.hidden3 = keras.layers.Dense(100)
        self.hidden4 = keras.layers.Dense(100)
        self.hidden5 = keras.layers.Dense(24)
        self.out = keras.layers.Dense(1)

    def train(self, dataset, epochs, val_split=0.1, batch_size=100, lr=0.01):
        """
        :param dataset: the training dataset, as a TensorSliceDataset.
        :param epochs: number of epochs to train the model.
        :param val_split: split between training and validation data.
        :param batch_size: The number of samples propagated through the network.
        :param lr: The learning rate of the training algorithm.
        """
        loss_fn = keras.losses.MeanSquaredError()
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        train_metric, val_metric = keras.metrics.MeanSquaredError(), keras.metrics.MeanSquaredError()
        train_dataset, val_dataset = self.__split_in_training_and_validation__(dataset, val_split)
        train_dataset, val_dataset = train_dataset.batch(batch_size), val_dataset.batch(batch_size)
        # Iterate over the epochs
        for epoch in range(1, epochs + 1):
            # Iterate over the batches of the dataset
            with tqdm(desc="[{:0>3d}/{:0>3d}]".format(epoch, epochs), total=len(train_dataset)) as progressbar:
                for x_batch_train, y_batch_train in train_dataset:
                    with tf.GradientTape() as tape:
                        train_loss, predicted = self.compute_loss(x_batch_train, y_batch_train, loss_fn)
                    grads = tape.gradient(train_loss, self.trainable_variables)
                    optimizer.apply_gradients(zip(grads, self.trainable_variables))
                    train_metric.update_state(y_batch_train, predicted)
                    progressbar.update()

                # Run a validation loop at the end of each epoch
                for x_batch_val, y_batch_val in val_dataset:
                    val_predicted = self.forward(x_batch_val)
                    val_metric.update_state(y_batch_val, val_predicted)

                # Update the progress-bar
                train_metric_value, val_metric_value = train_metric.result(), val_metric.result()
                progressbar.set_description("[{:0>3d}/{:0>3d}] loss: {:.2f} - validation loss: {:.2f}".format(
                    epoch, epochs, train_metric_value, val_metric_value
                ))
                progressbar.refresh()
                progressbar.update()

                # Reset metrics at the end of each epoch
                train_metric.reset_states()
                val_metric.reset_states()

    def predict(self, dataset, batch_size=100):
        truth = []
        predicted = []
        for x, y in dataset.batch(batch_size):
            preds = np.array(self.forward(x))
            predicted += list(preds.flatten())
            truth_y = np.array(y)
            truth += list(truth_y)
        return np.array(truth), np.array(predicted)

    def forward(self, x):
        """
        Runs x through the neural net and returns a predicted output.
        :param x: An input-tensor
        :returns: An output-tensor
        """
        x = keras.activations.relu(self.hidden1(x))
        x = keras.activations.relu(self.hidden2(x))
        x = keras.activations.relu(self.hidden3(x))
        x = keras.activations.relu(self.hidden4(x))
        x = keras.activations.relu(self.hidden5(x))
        x = self.out(x)
        return x

    def compute_loss(self, x, y, loss_fn):
        logits = self.forward(x)
        mse = loss_fn(y, logits)
        return mse, logits

    def copy_model(self, x):
        """
        Creates a new model with the same weights as this model.
        :param x: An input example. This is used to run a forward pass in order to initialize the weights.
        :return: A copy of the model.
        """
        copied_model = FFNModel(self.input_size)
        copied_model.forward(x)
        copied_model.set_weights(self.get_weights())
        return copied_model

    @staticmethod
    def __split_in_training_and_validation__(x, y, val_split):
        """
        :param x: training input
        :param y: training labels/output
        :param val_split: percentage
        :returns: train_dataset, val_dataset
        """
        split = math.floor(len(x) * val_split)
        x_s, y_s, x_val, y_val = x[split:], y[split:], x[:split], y[:split]
        train_dataset = tf.data.Dataset.from_tensor_slices((x_s, y_s))
        val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
        return train_dataset, val_dataset
