from RemainingUsefulLifeProject.Baseline.Models.maml.SimpleModel import CustomModel
from RemainingUsefulLifeProject.Data.datareader import DataObject

filepath = "../Data/CMAPSSData/"
FD001 = DataObject('FD001', filepath=filepath)

x_train = FD001.train_df[FD001.sensor_names].values
y_train = FD001.train_df['RUL'].values
x_test = FD001.test_df.groupby('unit_nr').last(1)[FD001.sensor_names].values
y_test = FD001.truth_df['RUL'].values

model = CustomModel(x_train.shape[1])
model.train(x_train, y_train, 5)

weights = model.get_weights()
model.set_weights(weights)

