import os
import numpy as np
import pandas as pd
from RemainingUsefulLifeProject.Baseline.Models.Redux.Standard import FFNModel, copy_model
from RemainingUsefulLifeProject.Data.cmapss_dao import DataObject
from RemainingUsefulLifeProject.Data.plotter import __get_directory__
from RemainingUsefulLifeProject.Data.scoring_methods import r2_score, mse_score, phm_score
from RemainingUsefulLifeProject.MAML.Models.maml import train_maml

lr = 0.1

def get_data(train_df, test_df, dao):
    test_df = test_df.groupby(by='unit_nr').last().reset_index()
    train_x, train_y = train_df[dao.sensor_names].values, train_df['RUL'].values
    test_x, test_y = test_df[dao.sensor_names].values, test_df['RUL'].values
    if len(test_y) > 100:
        index = np.random.choice(len(test_y), 100, replace=False)
        test_x, test_y = test_x[index], test_y[index]
    return train_x, train_y, test_x, test_y


def create_sets(dao):
    train_list, test_list = {}, {}
    for cond in dao.conditions:
        train, test = dao.datasets[cond]
        x_tr, y_tr, x_te, y_te = get_data(train, test, dao)
        train_list[cond] = (x_tr, y_tr)
        test_list[cond] = (x_te, y_te)
    return train_list, test_list


def write_gathered_scores(scores, title="Scores Neural Network"):
    folders = "Results", "Redux Model", "MAML", "Baseline Model", "Standard"
    folder = __get_directory__(*folders)
    for cond in scores.keys():
        file_title = "{} - condition {}.csv".format(title, cond)
        file_location = os.path.join(folder, file_title)
        r2_results, mse_results, phm_results = scores[cond]['r2'], scores[cond]['mse'], scores[cond]['phm']
        for i in range(len(r2_results)):
            with open(file_location, 'w+') as file:
                file.write("epoch;r2;mse;phm\n")
                file.write("{};{};{};{}\n".format(i, r2_results[i], mse_results[i], phm_results[i]))


# Read in the data and create the dataset
filepath = "../../../../../Data/CMAPSSData/"
FD002 = DataObject('FD002', filepath)
train_sets, test_sets = create_sets(FD002)

# Create training set:
train_set = []
for condition in range(1, 4):
    train_set.append(train_sets[condition])

# Create and train model:
model = FFNModel(len(FD002.sensor_names), lr=lr)
maml = train_maml(model, copy_model, epochs=50, dataset=train_set, lr_inner=lr)

# Test the maml:
gathered_scores, models = {}, {}
for condition in FD002.conditions:
    gathered_scores[condition] = {'r2': [], 'phm': [], 'mse': []}
    models[condition] = copy_model(maml, train_sets[condition][0])

for epoch in range(1, 51):
    print(" ====================== epoch {:0>2d}/{:0>2d} ======================".format(epoch, 50))
    for condition in FD002.conditions:
        x_train, y_train = train_sets[condition]
        x_test, y_test = test_sets[condition]
        models[condition].train(x_train, y_train, epochs=1)
        y_hat = models[condition].predict(x_test)
        r2, mse, phm = r2_score(y_test, y_hat), mse_score(y_test, y_hat), phm_score(y_test, y_hat)
        gathered_scores[condition]['r2'].append(r2)
        gathered_scores[condition]['mse'].append(mse)
        gathered_scores[condition]['phm'].append(phm)

write_gathered_scores(gathered_scores)
