from Baseline.Models.linearreggression import LinearRegressionModel
from Data.datareader import DataObject
from Data.plotter import *

filepath = '../../../../Data/CMAPSSData/'
FD001 = DataObject('FD001', filepath=filepath)

x_train = FD001.train_df[FD001.sensor_names].values
y_train = FD001.train_df['RUL'].values
x_test = FD001.test_df.groupby('unit_nr').last(1)[FD001.sensor_names].values
y_test = FD001.truth_df['RUL'].values

model = LinearRegressionModel()
model.train(x_train, y_train)
y_predicted = model.predict(x_test)
plot_predicted_v_true(y_test, y_predicted)
plot_difference(y_test, y_predicted)
