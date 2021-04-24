import tensorflow as tf
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Sequential


class FNNModel:

    def __init__(self, input_size):
        self.model = Sequential()
        self.model.add(Dense(24, input_dim=input_size, activation='sigmoid'))
        self.model.add(Dense(5))
        self.model.add(Dense(3))
        self.model.add(Dense(1))
        self.model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer='adam')

    def train(self, x, y, epochs):
        return self.model.fit(x, y, epochs=epochs, validation_split=0.1)

    def predict(self, x):
        return self.model.predict(x)