from RemainingUsefulLifeProject.Baseline.Models.FeedForwardNeuralNetworks.Baseline import FFNModel, copy_model
from RemainingUsefulLifeProject.Data.cmapss_dao import DataObject
import math
import numpy as np
import random

from RemainingUsefulLifeProject.Data.plotter import __get_directory__
from RemainingUsefulLifeProject.MAML.Models.maml import train_maml


def get_data(train_df, test_df, dao):
    train_x, train_y = train_df[dao.sensor_names].values, train_df['RUL'].values
    test_df = test_df.groupby(by='unit_nr').last().reset_index()
    test_x, test_y = test_df[dao.sensor_names].values, test_df['RUL'].values
    return train_x, train_y, test_x, test_y

def write_gathered_scores(gathered_scores):
    folders = "Results", "MAML", "Baseline", "Standard"
    folder = __get_directory__(*folders)
    print(folder)


filepath = "../../../../../../Data/CMAPSSData/"
FD001 = DataObject("FD001", filepath=filepath)
FD002 = DataObject("FD002", filepath)

train_dataset = []
for cond in (1, 2, 3):
    train, test = FD002.datasets[cond]
    x, y, _, _ = get_data(train, test, FD002)
    pieces = math.floor(len(x)/1000)
    x, y = np.array_split(x, pieces), np.array_split(y, pieces)
    for i in range(len(x)):
        train_dataset.append((x, y))

gathered_scores = {}
for condition in FD002.conditions:
    gathered_scores[condition] = {'r2': [], 'phm': [], 'mse': []}

for condition in FD002.conditions:
    scores = gathered_scores[condition]
    scores['r2'].append(random.randint(0, 10))
    scores['mse'].append(random.randint(10, 20))
    scores['phm'].append(random.randint(20, 30))

print(gathered_scores)

write_gathered_scores(gathered_scores)
