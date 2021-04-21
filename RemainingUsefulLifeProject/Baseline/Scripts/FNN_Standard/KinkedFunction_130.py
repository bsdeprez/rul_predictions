from RemainingUsefulLifeProject.Baseline.Models.fnn_models.fnn_standard import FNNModel
from RemainingUsefulLifeProject.Data.dataobject import DataObject
from RemainingUsefulLifeProject.Data.plotter import *
from RemainingUsefulLifeProject.Data.scoring_methods import *

filepath = '../../../../Data/CMAPSSData/'
FD001 = DataObject('FD001', filepath=filepath)
FD003 = DataObject('FD003', filepath=filepath)

FD001.add_kinking_function(130)
FD001.add_kinking_function(130)


def get_datasets(data_object):
    x_train = data_object.train_df[data_object.sensor_names].values
    y_train = data_object.train_df['RUL'].values
    x_test = data_object.test_df.groupby('unit_nr').last(1)[data_object.sensor_names].values
    y_test = data_object.truth_df['RUL'].values
    return x_train, y_train, x_test, y_test

def test_data(model, x_test, y_test, dataset, trained_on=""):
    y_predicted = model.predict(x_test).flatten()
    plot_predicted_v_true(y_test, y_predicted, "Accuracy {}".format(dataset),
                          "Standard FNN\\Kinked 130\\Trained on {}".format(trained_on), show=False)
    plot_difference(y_test, y_predicted, "Difference distribution {}".format(dataset),
                    "Standard FNN\\Kinked 130\\Trained on {}".format(trained_on), show=False)
    print_scores(y_test, y_predicted, dataset, "Standard FNN\\Kinked 130\\Trained on {}".format(trained_on), dataset)


x_train_FD001, y_train_FD001, x_test_FD001, y_test_FD001 = get_datasets(FD001)
x_train_FD003, y_train_FD003, x_test_FD003, y_test_FD003 = get_datasets(FD003)

model = FNNModel(input_size=len(FD001.sensor_names))
history = model.train(x_train_FD001, y_train_FD001, epochs=125)
plot_history(history)
print(" =================== TRAINED ON {} ===================".format(FD001.name))
test_data(model, x_test_FD001, y_test_FD001, FD001.name, FD001.name)
test_data(model, x_test_FD003, y_test_FD003, FD003.name, FD001.name)
print("")
history = model.train(x_train_FD003, y_train_FD003, epochs=125)
plot_history(history)
print(" =================== TRAINED ON {} ===================".format(FD003.name))
test_data(model, x_test_FD001, y_test_FD001, FD001.name, FD003.name)
test_data(model, x_test_FD003, y_test_FD003, FD003.name, FD003.name)
