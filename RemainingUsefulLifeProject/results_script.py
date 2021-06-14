import numpy as np
import pandas as pd
from RemainingUsefulLifeProject.Baseline.Models.FNNs.Baseline import FFNModel
from RemainingUsefulLifeProject.Data.cmapss_dao import DataObject


def get_data(train_df, test_df, dao):
    train_x, train_y = train_df[dao.sensor_names].values, train_df['RUL'].values
    test_df = test_df.groupby(by='unit_nr').last().reset_index()
    test_x, test_y = test_df[dao.sensor_names].values, test_df['RUL'].values
    if len(test_y) > 100:
        index = np.random.choice(len(test_y), 100, replace=False)
        test_x, test_y = test_x[index], test_y[index]
    return train_x, train_y, test_x, test_y


filepath = "../Data/CMAPSSData/"
FD002 = DataObject("FD002", filepath)

# =========================================================
# CLASSIC MODEL AND TRANSFER LEARNING
# =========================================================
# Step 1: Creating the data set for training
train_dfs = []
for condition in (1, 2, 3):
    train, test = FD002.datasets[condition]
    train_dfs.append(train)
train_set = pd.concat(train_dfs)
train_x, train_y, _, _ = get_data(train, test, FD002)

model = FFNModel(len(FD002.sensor_names))

