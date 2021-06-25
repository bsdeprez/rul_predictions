import os

from Results_Gathering.Data.DAO import DataObject
from Results_Gathering.Models.Standard_Model import FFNModel


learning_rate = 0.01
filepath = "../Data/CMAPSSData/"
FD002 = DataObject("FD002", filepath)
drop_columns = ['s_1', 's_5', 's_6', 's_10', 's_15', 's_16', 's_18', 's_19', 's_9', 's_14']
FD002.drop_columns(drop_columns)

print(len(FD002.sensor_names))
model = FFNModel(len(FD002.sensor_names), learning_rate)
print(model.get_model().summary())