from Results_Gathering.Data.DAO import DataObject
from Results_Gathering.Helper_functions.Plotter import __get_directory__
from Results_Gathering.Models.Standard_Model import FFNModel
import numpy as np

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



