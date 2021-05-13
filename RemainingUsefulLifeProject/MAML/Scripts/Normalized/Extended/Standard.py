from RemainingUsefulLifeProject.Baseline.Models.Extended import FFNModel, copy_model
from RemainingUsefulLifeProject.Data.cmapss_dao import DataObject
import math
import numpy as np
import os
from RemainingUsefulLifeProject.Data.plotter import __get_directory__
from RemainingUsefulLifeProject.Data.scoring_methods import r2_score, mse_score, phm_score
from RemainingUsefulLifeProject.MAML.Models.maml import train_maml


def get_data(train_df, test_df, dao):
    train_x, train_y = train_df[dao.sensor_names].values, train_df['RUL'].values
    test_df = test_df.groupby(by='unit_nr').last().reset_index()
    test_x, test_y = test_df[dao.sensor_names].values, test_df['RUL'].values
    if len(test_y) > 100:
        index = np.random.choice(len(test_y), 100, replace=False)
        test_x, test_y = test_x[index], test_y[index]
    return train_x, train_y, test_x, test_y


def write_gathered_scores(scores, title="Scores MAML"):
    folders = "Results", "MAML", "Extended", "Standard"
    folder = __get_directory__(*folders)
    for key in scores.keys():
        file_title = "{} - Condition {}.csv".format(title, key)
        file_location = os.path.join(folder, file_title)
        r2_sc, mse_sc, phm_sc = scores[key]['r2'], scores[key]['mse'], scores[key]['phm']
        with open(file_location, 'w+') as file:
            file.write("epoch;r2;mse;phm\n")
            for epoch in range(len(r2_sc)):
                file.write("{};{};{};{}\n".format(epoch, r2_sc[epoch], mse_sc[epoch], phm_sc[epoch]))


filepath = "../../../../../Data/CMAPSSData/"
FD001 = DataObject("FD001", filepath=filepath)
FD002 = DataObject("FD002", filepath)

# CREATE TRAINING DATASET
train_dataset = []
for condition in (1, 2, 3):
    train, test = FD002.datasets[condition]
    x, y, _, _ = get_data(train, test, FD002)
    pieces = math.floor(len(x) / 1000)
    x, y = np.array_split(x, pieces), np.array_split(y, pieces)
    for i in range(len(x)):
        train_dataset.append((x[i], y[i]))

# CREATE AND TRAIN MODEL
model = FFNModel(len(FD002.sensor_names))
maml = train_maml(model, copy_model, epochs=50, dataset=train_dataset)

# TEST THE MAML
gathered_scores = {}
test_data = {}
train_data = {}
models = {}
for condition in FD002.conditions:
    gathered_scores[condition] = {'r2': [], 'phm': [], 'mse': []}
    train, test = FD002.datasets[condition]
    x_train, y_train, x_test, y_test = get_data(train, test, FD002)
    test_data[condition] = (x_test, y_test)
    train_data[condition] = (x_train, y_train)
    models[condition] = copy_model(maml)

for step in range(50):
    print(" ==================== STEP {} ====================".format(step))
    for condition in FD002.conditions:
        x_test, y_test = test_data[condition]
        predicted = models[condition].predict(x_test)
        r2 = r2_score(y_test, predicted)
        mse = mse_score(y_test, predicted)
        phm = phm_score(y_test, predicted)
        gathered_scores[condition]['r2'].append(r2)
        gathered_scores[condition]['mse'].append(mse)
        gathered_scores[condition]['phm'].append(phm)
    for condition in FD002.conditions:
        x_train, y_train = train_data[condition]
        models[condition].train(x_train, y_train, epochs=1)

write_gathered_scores(gathered_scores)


