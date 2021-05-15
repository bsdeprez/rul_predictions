from RemainingUsefulLifeProject.Baseline.Models.FNNs.Baseline import FFNModel
from RemainingUsefulLifeProject.Baseline.Models.Test import MyModel
from RemainingUsefulLifeProject.Data.cmapss_dao import DataObject
import numpy as np
import random
import matplotlib.pyplot as plt

def get_data(train_df, test_df, dao):
    train_x, train_y = train_df[dao.sensor_names].values, train_df['RUL'].values
    test_df = test_df.groupby(by='unit_nr').last().reset_index()
    test_x, test_y = test_df[dao.sensor_names].values, test_df['RUL'].values
    if len(test_y) > 100:
        index = np.random.choice(len(test_y), 100, replace=False)
        test_x, test_y = test_x[index], test_y[index]
    return train_x, train_y, test_x, test_y


filepath = "../Data/CMAPSSData/"
FD001 = DataObject('FD001', filepath=filepath)
train, test = FD001.datasets[1]
x_train, y_train, _, _ = get_data(train, test, FD001)
model = MyModel(len(FD001.sensor_names))
history = model.train(x_train, y_train, epochs=100)
plt.plot(history.history['loss'][1:], label='loss')
plt.plot(history.history['val_loss'][1:], label='val_loss')
plt.legend(loc='upper left')
plt.show()
