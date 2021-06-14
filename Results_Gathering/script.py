import numpy as np
import pandas as pd
from keras.layers import Dense
from keras.models import Sequential

from Results_Gathering.Data.DAO import DataObject
from Results_Gathering.Models.Standard_Model import FFNModel, copy_model

"""
PARAMETERS
"""
learning_rate = 0.01


def get_data(train_df, test_df, dao):
    train_x, train_y = train_df[dao.sensor_names].values, train_df['RUL'].values
    test_df = test_df.groupby(by='unit_nr').last().reset_index()
    test_x, test_y = test_df[dao.sensor_names].values, test_df['RUL'].values
    if len(test_y) > 100:
        index = np.random.choice(len(test_y), 100, replace=False)
        test_x, test_y = test_x[index], test_y[index]
    return train_x, train_y, test_x, test_y


# Create the dataset used for training the baseline.
data_path = "../Data/CMAPSSData/"
FD002 = DataObject("FD002", data_path)
train_datasets = []
for condition in (1, 2, 3):
    train_per_condition, test_per_condition = FD002.datasets[condition]
    train_datasets.append(train_per_condition)
train = pd.concat(train_datasets)
x_train, y_train, _, _ = get_data(train, test_per_condition, FD002)

# Create and train the model
model = FFNModel(len(FD002.sensor_names), learning_rate)
model.train(x_train, y_train, epochs=2)

# Create the data for transfer learning.
gathered_scores, test_data, train_data, models = {}, {}, {}, {}
for condition in FD002.conditions[:1]:
    gathered_scores[condition] = {'r2': [], 'phm': [], 'mse': []}
    train, test = FD002.datasets[condition]
    x_train, y_train, x_test, y_test = get_data(train, test, FD002)
    test_data[condition] = (x_test, y_test)
    train_data[condition] = (x_train, y_train)

    # Remove the output layer of the model
    raw_model = model.get_model()
    copied_model = Sequential()
    for layer in raw_model.layers[:-1]:
        copied_model.add(layer)
    # Freeze the existing layers
    for layer in copied_model.layers:
        layer.trainable = False
    # Add new layers to the frozen network
    copied_model.add(Dense(50, name="new_layer_1"))
    copied_model.add(Dense(1, name="new_output_layer"))
    copied_model.compile(loss="mean_squared_error", optimizer=model.optimizer)






