import pandas as pd
from keras.layers import Dense
from keras.models import Sequential

from Results_Gathering.Data.DAO import DataObject
from Results_Gathering.Data.Data_Helper import get_data
from Results_Gathering.Helper_functions.Plotter import write_gathered_scores
from Results_Gathering.Helper_functions.Scoring_functions import *
from Results_Gathering.Models.MAML import train_maml
from Results_Gathering.Models.Standard_Model import FFNModel, copy_model

# ============================================
# PARAMETERS
# ============================================
learning_rate = 0.01
epochs = 50
units_in_training_data = 260
folders = "Experiment 1",

# ============================================
# Gathering data
# ============================================
data_path = "../Data/CMAPSSData/"
FD002 = DataObject("FD002", data_path)
drop_columns = ['s_1', 's_5', 's_6', 's_10', 's_15', 's_16', 's_18', 's_19', 's_9', 's_14']
FD002.drop_columns(drop_columns)

# Create the dataset for the first training part of Transfer learning
train_datasets, test_datasets = [], []
for condition in (1, 2, 3):
    train_per_condition, test_per_condition = FD002.datasets[condition]
    train_datasets.append(train_per_condition)
    test_datasets.append(test_per_condition)
train = pd.concat(train_datasets)
test = pd.concat(test_datasets)
x_TL, y_TL, x_original, y_original = get_data(train, test_per_condition, FD002)

# Create the dataset for the first training part of MAML
MAML_set = []
for condition in (1, 2, 3):
    train_per_condition, test_per_condition = FD002.datasets[condition]
    x, y, _, _ = get_data(train_per_condition, test_per_condition, FD002)
    MAML_set.append((x, y))

# Create the dataset for the training part afterwards
train_list, test_list = {}, {}
for condition in (4, 5, 6):
    train, test = FD002.datasets[condition]
    train = train[train['unit_nr'] <= units_in_training_data]
    x_train, y_train, x_test, y_test = get_data(train, test, FD002)
    train_list[condition], test_list[condition] = (x_train, y_train), (x_test, y_test)

# ============================================
# Transfer learning
# ============================================
# Create and train the model
model = FFNModel(len(FD002.sensor_names), learning_rate)
model.train(x_TL, y_TL, epochs=epochs)
# Create the data for transfer learning.
gathered_scores, general_scores, models = {}, {}, {}
for condition in (4, 5, 6):
    gathered_scores[condition] = {'r2': [], 'phm': [], 'mse': []}
    general_scores[condition] = {}
    raw_model = model.get_model()
    copied_model = Sequential()
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
    for condition in (4, 5, 6):
        x_test, y_test = test_list[condition]
        predicted = models[condition].predict(x_test)
        r2, mse, phm = r2_score(y_test, predicted), mse_score(y_test, predicted), phm_score(y_test, predicted)
        gathered_scores[condition]['r2'].append(r2)
        gathered_scores[condition]['mse'].append(mse)
        gathered_scores[condition]['phm'].append(phm)
    for condition in (4, 5, 6):
        x_train, y_train = train_list[condition]
        models[condition].fit(x_train, y_train, epochs=1)
# Calculate the score for the original problem
for condition in (4, 5, 6):
    predicted = models[condition].predict(x_original)
    r2, mse, phm = r2_score(y_original, predicted), mse_score(y_original, predicted), phm_score(y_original, predicted)
    general_scores[condition]['r2'], general_scores[condition]['mse'], general_scores[condition]['phm'] = r2, mse, phm
# Write the scores of transfer learning
write_gathered_scores(gathered_scores, general_scores, *folders, "Transfer Learning", title="Transfer Learning")

# ============================================
# MAML
# ============================================
# Create the model and train it.
model = FFNModel(len(FD002.sensor_names), learning_rate)
maml = train_maml(model, copy_model, epochs, MAML_set, learning_rate)
gathered_scores, models = {}, {}
for condition in (4, 5, 6):
    gathered_scores[condition] = {'r2': [], 'phm': [], 'mse': []}
    general_scores[condition] = {}
    models[condition] = copy_model(maml, train_list[condition][0])
for epoch in range(1, epochs+1):
    print(" ====================== epoch {:0>2d}/{:0>2d} ======================".format(epoch, epochs))
    for condition in (4, 5, 6):
        x_train, y_train = train_list[condition]
        x_test, y_test = train_list[condition]
        models[condition].train(x_train, y_train, epochs=1)
        y_hat = models[condition].predict(x_test)
        r2, mse, phm = r2_score(y_test, y_hat), mse_score(y_test, y_hat), phm_score(y_test, y_hat)
        gathered_scores[condition]['r2'].append(r2)
        gathered_scores[condition]['mse'].append(mse)
        gathered_scores[condition]['phm'].append(phm)
for condition in (4, 5, 6):
    predicted = models[condition].predict(x_original)
    r2, mse, phm = r2_score(y_original, predicted), mse_score(y_original, predicted), phm_score(y_original, predicted)
    general_scores[condition]['r2'], general_scores[condition]['mse'], general_scores[condition]['phm'] = r2, mse, phm
write_gathered_scores(gathered_scores, general_scores, *folders, "MAML", title="MAML")
