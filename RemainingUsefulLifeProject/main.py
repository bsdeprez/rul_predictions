from RemainingUsefulLifeProject.Baseline.Models.Redux.Standard import FFNModel
from RemainingUsefulLifeProject.Data.cmapss_dao import DataObject
import numpy as np
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
FD001.add_kinking_point(130)
train, test = FD001.datasets[1]
x_train, y_train, x_test, y_test = get_data(train, test, FD001)

model = FFNModel(len(FD001.sensor_names), lr=0.03)

"""losses = model.train(x_train, y_train, epochs=250)
plt.plot(losses)
plt.title = "History"
plt.show()"""

"""x = test[test['unit_nr'] == 3]
test_x, test_y = x[FD001.sensor_names].values, x['RUL'].values
y_hat = model.predict(test_x)
plt.plot(test_y, label="Truth")
plt.plot(y_hat, label="Predicted")
plt.legend(loc='upper right')
plt.show()"""
for sensor in FD001.sensor_names:
    data = test[test['unit_nr'] == 3][sensor]
    plt.plot(data)
    plt.title = sensor
    plt.show()
