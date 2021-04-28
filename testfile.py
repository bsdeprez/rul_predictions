import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from RemainingUsefulLifeProject.Data.datareader import DataObject


class SimpleModel:

    def __init__(self, input_size):
        self.model = Sequential()
        self.model.add(Dense(5))
        self.model.add(Dense(1))
        self.model.build((None, input_size))
        print(self.model.summary())

    def train(self, x, y):
        self.model.compile(optimizer="Adam", loss="mse")
        history = self.model.fit(x, y, epochs=2)
        return history

    def __inner_loop__(self):
        print("Hollw world")


def get_datasets(data_object):
    x_train = data_object.train_df[data_object.sensor_names].values
    y_train = data_object.train_df['RUL'].values
    x_test = data_object.test_df.groupby('unit_nr').last(1)[data_object.sensor_names].values
    y_test = data_object.truth_df['RUL'].values
    return x_train, y_train, x_test, y_test


filepath = "Data/CMAPSSData/"
FD001 = DataObject('FD001', filepath=filepath)
x_train, y_train, _, _ = get_datasets(FD001)
alpha, beta = 0.0001, 0.0003

model = SimpleModel(x_train.shape[1])
h = model.train(x_train, y_train)
print(h.history['loss'])
