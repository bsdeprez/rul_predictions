from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Sequential
from tensorflow import keras

from RemainingUsefulLifeProject.MAML.Models.maml import loss_function


def copy_model(model):
    copy = FFNModel(model.input_size)
    copy.model.set_weights(model.model.get_weights())
    return copy


class FFNModel:

    def __init__(self, input_size, lr=0.01):
        self.input_size = input_size
        self.model = Sequential()
        self.model.add(Dense(24, input_dim=self.input_size, activation='relu'))
        self.model.add(Dense(5))
        self.model.add(Dense(3))
        self.model.add(Dense(1))
        optimizer = keras.optimizers.Adam(learning_rate=lr)
        self.model.compile(loss='mean_squared_error', optimizer=optimizer)

    def train(self, x, y, epochs):
        return self.model.fit(x, y, epochs=epochs, validation_split=0.1, callbacks=[
            keras.callbacks.EarlyStopping(patience=10, monitor='val_loss', restore_best_weights=True)
        ])

    def predict(self, x):
        return self.model.predict(x)

    def compute_loss(self, x, y, loss_fn=loss_function):
        logits = self.model(x)
        mse = loss_fn(y, logits)
        return mse, logits
