from Baseline.Models.linearreggression import LinearRegressionModel
from Data.datareader import DataObject
from Data.plotter import *
from Data.scoring_methods import *

filepath = '../../../../Data/CMAPSSData/'
FD001 = DataObject('FD001', filepath=filepath)

chosen_sensors = ['s_2', 's_3', 's_4', 's_7', 's_11', 's_12', 's_15']
drop_sensors = [sensor for sensor in FD001.sensor_names if sensor not in chosen_sensors]
FD001.drop_columns(drop_sensors)

x_train = FD001.train_df[FD001.sensor_names].values
y_train = FD001.train_df['RUL'].values
x_test = FD001.test_df.groupby('unit_nr').last(1)[FD001.sensor_names].values
y_test = FD001.truth_df['RUL'].values

model = LinearRegressionModel()
model.train(x_train, y_train)
y_predicted = model.predict(x_test)
plot_predicted_v_true(y_test, y_predicted)
plot_difference(y_test, y_predicted)

print("============= Scores =============")
print("R2: {}".format(r2_score(y_test, y_predicted)))
print("MSE: {}".format(mse_score(y_test, y_predicted)))
print("PHM: {}".format(phm_score(y_test, y_predicted)))
