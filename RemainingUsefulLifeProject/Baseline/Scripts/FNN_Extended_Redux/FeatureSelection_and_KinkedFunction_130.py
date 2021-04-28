from RemainingUsefulLifeProject.Baseline.Models.fnn_models_redux.fnn_extended import FNNModel
from RemainingUsefulLifeProject.Data.dataobject import DataObject
from RemainingUsefulLifeProject.Data.plotter import *
from RemainingUsefulLifeProject.Data.scoring_methods import print_scores

filepath = '../../../../Data/CMAPSSData/'
FD001 = DataObject('FD001', filepath=filepath)
FD002 = DataObject('FD002', filepath=filepath)

drop_sensors = ['s_1', 's_5', 's_10', 's_16', 's_18', 's_19']
FD001.drop_columns(drop_sensors)
FD002.drop_columns(drop_sensors)

FD001.add_kinking_function(130)
FD002.add_kinking_function(130)

def get_datasets(data_object):
    x_train, y_train = data_object.train_df[data_object.sensor_names], data_object.train_df['RUL']
    x_test, y_test = data_object.test_df.groupby('unit_nr').last(1)[data_object.sensor_names], data_object.truth_df[
        'RUL']
    return x_train, y_train, x_test, y_test


def test_data(model, x_test, y_test, dataset, trained_on=""):
    y_predicted = model.predict(x_test).flatten()
    plot_predicted_v_true(y_test, y_predicted, "Accuracy {}".format(dataset),
                          "Redux\\Extended FNN\\Feature Selection And Kinked 130\\Trained on {}".format(trained_on), show=False)
    plot_difference(y_test, y_predicted, "Difference distribution {}".format(dataset),
                    "Redux\\Extended FNN\\Feature Selection And Kinked 130\\Trained on {}".format(trained_on), show=False)
    print_scores(y_test, y_predicted, dataset, "Redux\\Extended FNN\\Feature Selection And Kinked 130\\Trained on {}".format(trained_on), dataset)


x_s_FD001, y_s_FD001, x_t_FD001, y_t_FD001 = get_datasets(FD001)
x_s_FD002, y_s_FD002, x_t_FD002, y_t_FD002 = get_datasets(FD002)

FNN = FNNModel(input_size=len(FD001.sensor_names))
FNN.train(x_s_FD001, y_s_FD001, epochs=125)
print(" =================== TRAINED ON {} ===================".format(FD001.name))
test_data(FNN, x_t_FD001, y_t_FD001, FD001.name, FD001.name)
test_data(FNN, x_t_FD002, y_t_FD002, FD002.name, FD001.name)
print("")
FNN.train(x_s_FD002, y_s_FD002, epochs=125)
print(" =================== TRAINED ON {} ===================".format(FD002.name))
test_data(FNN, x_t_FD001, y_t_FD001, FD001.name, FD002.name)
test_data(FNN, x_t_FD002, y_t_FD002, FD002.name, FD002.name)
