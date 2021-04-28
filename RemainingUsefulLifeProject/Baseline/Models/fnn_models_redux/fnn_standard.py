import tensorflow as tf
import math
import tensorflow.keras.layers as layers
from tqdm import tqdm


class FNNModel:

    def __init__(self, input_size, learning_rate=0.03):
        self.model = tf.keras.Sequential([
            layers.Dense(24, input_shape=(input_size,), activation="sigmoid"),
            layers.Dense(5),
            layers.Dense(3),
            layers.Dense(1)
        ])
        self.loss_fn = tf.keras.losses.MeanSquaredError()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.train_acc_metric = tf.keras.metrics.MeanSquaredError()
        self.val_acc_metric = tf.keras.metrics.MeanSquaredError()

    def train(self, x, y, epochs, val_split=0.1, batch_size=100):
        """
        :param x: training input
        :param y: training labels/output
        :param epochs: number of epochs to train the model
        :param val_split: split between training and validation data
        :param batch_size: The number of samples propagated through the network
        """
        previous_val_loss = 0
        patience = 5
        current_patience = patience
        delta = 1

        train_dataset, val_dataset = self.__split_in_training_and_validation__(x, y, val_split)
        train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)
        val_dataset = val_dataset.batch(batch_size)
        # Iterate over the epochs
        for epoch in range(1, epochs+1):
            # Iterate over the batches of the dataset
            with tqdm(desc="[{:0>3d}/{:0>3d}]".format(epoch, epochs), total=len(train_dataset)) as progressbar:
                for x_batch_train, y_batch_train in train_dataset:
                    with tf.GradientTape() as tape:
                        predicted = self.model(x_batch_train, training=True)
                        train_loss = self.loss_fn(y_batch_train, predicted)
                    grads = tape.gradient(train_loss, self.model.trainable_weights)
                    self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
                    # Update training metric
                    self.train_acc_metric.update_state(y_batch_train, predicted)
                    progressbar.update()

                # Run a validation loop at the end of each epoch
                for x_batch_val, y_batch_val in val_dataset:
                    val_predicted = self.model(x_batch_val, training=False)
                    # Update val metrics
                    self.val_acc_metric.update_state(y_batch_val, val_predicted)

                # Update the progress-bar
                train_metric, val_metric = self.train_acc_metric.result(), self.val_acc_metric.result()
                progressbar.set_description("[{:0>3d}/{:0>3d}] loss: {:.2f} - validation loss: {:.2f}".format(epoch, epochs,
                                                                                                       train_metric,
                                                                                                       val_metric))
                progressbar.refresh()
                progressbar.update()

                # Reset metrics at the end of each epoch
                self.train_acc_metric.reset_states()
                self.val_acc_metric.reset_states()

                # Check for early stopping
                if previous_val_loss < val_metric or previous_val_loss - val_metric < delta:
                    current_patience -= 1
                    previous_val_loss = val_metric
                else:
                    current_patience = patience
                if current_patience == 0:
                    print("Stopping early...")
                    break

    def predict(self, x):
        """
        :param x: input_vector
        :return: a tensor containing predicted values
        """
        return self.model.predict(x.values)

    @staticmethod
    def __split_in_training_and_validation__(x, y, val_split):
        """
        :param x: training input
        :param y: training labels/output
        :param val_split: percentage
        :returns: train_dataset, val_dataset
        """
        x, y = x.values, y.values
        split = math.floor(len(x) * val_split)
        x_s, y_s, x_val, y_val = x[split:], y[split:], x[:split], y[:split]
        train_dataset = tf.data.Dataset.from_tensor_slices((x_s, y_s))
        val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
        return train_dataset, val_dataset
