from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Sequential
from tensorflow import keras

from RemainingUsefulLifeProject.MAML.Models.maml import loss_function


def copy_model(model):
    copy = MyModel(model.input_size)
    copy.model.set_weights(model.model.get_weights())
    return copy


class MyModel:

    def __init__(self, input_size, lr=0.00001):
        self.input_size = input_size
        self.model = Sequential()
        self.model.add(Dense(200, input_dim=self.input_size, activation='relu'))
        self.model.add(Dense(250))
        self.model.add(Dense(200))
        self.model.add(Dense(1))
        optimizer = keras.optimizers.RMSprop(learning_rate=lr)
        self.model.compile(loss='mean_squared_error', optimizer=optimizer)

    def train(self, x, y, epochs):
        return self.model.fit(x, y, epochs=epochs, validation_split=0.1, batch_size=150)

    def predict(self, x):
        return self.model.predict(x)

    def compute_loss(self, x, y, loss_fn=loss_function):
        logits = self.model(x)
        mse = loss_fn(y, logits)
        return mse, logits
