import pandas as pd
from keras.layers import Dense
from keras.models import Sequential

from Results_Gathering.Data.DAO import DataObject
from Results_Gathering.Data.Data_Helper import get_data
from Results_Gathering.Helper_functions.Plotter import write_gathered_scores
from Results_Gathering.Helper_functions.Scoring_functions import *
from Results_Gathering.Models.MAML import train_maml
from Results_Gathering.Models.Standard_Model import FFNModel, copy_model

"""
PARAMETERS
"""
learning_rate = 0.01
epochs = 50

"""
TRANSFER LEARNING
"""
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
model.train(x_train, y_train, epochs=epochs)
# Create the data for transfer learning.
gathered_scores, test_data, train_data, models = {}, {}, {}, {}
for condition in FD002.conditions:
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
    models[condition] = copied_model
# Train the new models on their respective data set
for step in range(epochs):
    print(" ===================== STEP {} ===================== ".format(step))
    for condition in FD002.conditions:
        x_test, y_test = test_data[condition]
        predicted = models[condition].predict(x_test)
        r2, mse, phm = r2_score(y_test, predicted), mse_score(y_test, predicted), phm_score(y_test, predicted)
        gathered_scores[condition]['r2'].append(r2)
        gathered_scores[condition]['mse'].append(mse)
        gathered_scores[condition]['phm'].append(phm)
    for condition in FD002.conditions:
        x_train, y_train = train_data[condition]
        models[condition].fit(x_train, y_train, epochs=1)
# Write the scores of transfer learning
write_gathered_scores(gathered_scores, "Standard Model", "Transfer Learning", title="Transfer Learning")

"""
MAML
"""
# Create the dataset used for training the MAML.
train_list, test_list = {}, {}
for condition in FD002.conditions:
    train, test = FD002.datasets[condition]
    x_train, y_train, x_test, y_test = get_data(train, test, FD002)
    train_list[condition], test_list[condition] = (x_train, y_train), (x_test, y_test)
train_set = []
for condition in (1, 2, 3):
    train_set.append(train_list[condition])
# Create the model and train it.
model = FFNModel(len(FD002.sensor_names), learning_rate)
maml = train_maml(model, copy_model, epochs, train_set, learning_rate)
# Test the maml
gathered_scores, models = {}, {}
for condition in FD002.conditions:
    gathered_scores[condition] = {'r2': [], 'phm': [], 'mse': []}
    models[condition] = copy_model(maml, train_list[condition][0])
for epoch in range(1, epochs+1):
    print(" ====================== epoch {:0>2d}/{:0>2d} ======================".format(epoch, epochs))
    for condition in FD002.conditions:
        x_train, y_train = train_list[condition]
        x_test, y_test = train_list[condition]
        models[condition].train(x_train, y_train, epochs=1)
        y_hat = models[condition].predict(x_test)
        r2, mse, phm = r2_score(y_test, y_hat), mse_score(y_test, y_hat), phm_score(y_test, y_hat)
        gathered_scores[condition]['r2'].append(r2)
        gathered_scores[condition]['mse'].append(mse)
        gathered_scores[condition]['phm'].append(phm)
write_gathered_scores(gathered_scores, "Standard Model", "MAML", title="MAML")


