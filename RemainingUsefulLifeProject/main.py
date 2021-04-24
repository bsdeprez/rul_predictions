import os
import random
import pandas as pd
import tensorflow as tf
from RemainingUsefulLifeProject.Baseline.Models.fnn_models.fnn_standard import FNNModel
from RemainingUsefulLifeProject.Data.dataobject import CustomDataObject
from RemainingUsefulLifeProject.MAML.MAML import MAML


def get_data(data_object, N):
    train = data_object.train_list[0]
    test = data_object.test_list[0]

    train_sample = random.sample(range(0, len(train)), N)
    test_sample = random.sample(range(0, len(test)), N)
    x_train = train[data_object.sensor_names].values[train_sample]
    y_train = train['RUL'].values[train_sample]
    x_test = test[data_object.sensor_names].values[test_sample]
    y_test = test['RUL'].values[test_sample]
    return x_train, y_train, x_test, y_test


dao = CustomDataObject()
x_s, y_s, x_t, y_t = get_data(dao, 5)
maml = MAML(x_s.shape[1])
maml.train(x_s, y_s, x_t, y_t, 100)
