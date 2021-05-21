import numpy as np
import pandas as pd
import os
from RemainingUsefulLifeProject.Baseline.Models.Redux.Standard import FFNModel
from RemainingUsefulLifeProject.Data.cmapss_dao import DataObject
from RemainingUsefulLifeProject.Data.plotter import __get_directory__
from RemainingUsefulLifeProject.Data.scoring_methods import r2_score, phm_score, mse_score

np.random.seed(42)

lr = 0.01
epochs = 50

def get_data(train_df, test_df, dao):
    test_df = test_df.groupby(by='unit_nr').last().reset_index()
    x_tr, y_tr = train_df[dao.sensor_names].values, train_df['RUL'].values
    x_te, y_te = test_df[dao.sensor_names].values, test_df['RUL'].values
    if len(y_te) > 100:
        index = np.random.choice(len(y_te), 100, replace=False)
        x_te, y_te = x_te[index], y_te[index]
    return x_tr, y_tr, x_te, y_te

def create_sets(dao):
    train_list, test_list = {}, {}
    for cond in dao.conditions:
        train, test = dao.datasets[cond]
        x_tr, y_tr, x_te, y_te = get_data(train, test, dao)
        train_list[cond] = (x_tr, y_tr)
        test_list[cond] = (x_te, y_te)
    return train_list, test_list


# Read in the data and create the dataset
filepath = "../../../../../Data/CMAPSSData/"
FD002 = DataObject('FD002', filepath, normalize=False)
train_sets, test_sets = create_sets(FD002)
gathered_scores, models = {}, {}
# Create and train the model

weight_path = "../../../../Saved Models/ReduxModelTrainedOnCombinedDataset/model"

# Let's train a model on the first 3 datasets.
train_1, test_1 = FD002.datasets[1]
train_2, _ = FD002.datasets[2]
train_3, _ = FD002.datasets[3]
train_set = pd.concat((train_1, train_2, train_3))
x_train, y_train, _, _ = get_data(train_set, test_1, FD002)
# Create and train model:
model = FFNModel(len(FD002.sensor_names), lr=lr)
model.train(x_train, y_train, epochs=epochs)

for condition in FD002.conditions:
    x_test, y_test = test_sets[condition]
    y_hat = model.predict(x_test)
    r2, mse, phm = r2_score(y_test, y_hat), mse_score(y_test, y_hat), phm_score(y_test, y_hat)
    print(" ============== CONDITION {} ==============".format(condition))
    print(" r2: {}".format(r2))
    print(" mse: {}".format(mse))
    print(" phm: {}".format(phm))
