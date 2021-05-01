from RemainingUsefulLifeProject.Baseline.Models.FNNs.fnn_extended import FFNModel
from RemainingUsefulLifeProject.Data.dataobject import CustomDataObject

from tensorflow.python.framework.ops import disable_eager_execution

from RemainingUsefulLifeProject.Data.plotter import plot_predicted_v_true, plot_difference
from RemainingUsefulLifeProject.Data.scoring_methods import print_scores

filepath = '../../../../../Data/Custom/'

FD001 = CustomDataObject('FD001', filepath)
FD002 = CustomDataObject('FD002', filepath)

drop_sensors = ['s_1', 's_5', 's_10', 's_16', 's_18', 's_19']
FD001.drop_columns(drop_sensors)
FD002.drop_columns(drop_sensors)


def get_data(dao, condition):
    x_train, y_train = dao.train_dfs[condition][dao.sensor_names], dao.train_dfs[condition]['RUL']
    test = dao.test_dfs[condition].groupby(by='unit_nr').last()
    x_test, y_test = test[dao.sensor_names], test['RUL']
    return x_train.values, y_train.values, x_test.values, y_test.values

def print_results(model, x_test, y_test, tested_on, trained_on):
    y_predicted = model.predict(x_test)
    plot_predicted_v_true(y_test, y_predicted, "Accuracy {}".format(tested_on),
                          "Extended FNN\\Feature Selection\\Trained on {}".format(trained_on), show=False)
    plot_difference(y_test, y_predicted, 'Difference distribution {}'.format(tested_on),
                    "Extended FNN\\Feature Selection\\Trained on {}".format(trained_on), show=False)
    print_scores(y_test, y_predicted, tested_on, "Extended FNN\\Feature Selection\\Trained on {}".format(trained_on), tested_on)


xs_1, ys_1, xt_1, yt_1 = get_data(FD001, "5")
xs_2, ys_2, xt_2, yt_2 = get_data(FD002, "1")

print(xs_1.shape)
fnn1 = FFNModel(xs_1.shape[1])
fnn2 = FFNModel(xs_2.shape[1])
fnn1.train(xs_1, ys_1, epochs=30)
fnn2.train(xs_2, ys_2, epochs=30)
print(" =================== TRAINED ON CONDITION 5 ===================")
print_results(fnn1, xt_1, yt_1, tested_on="Condition 5", trained_on="Condition 5")
print_results(fnn1, xt_2, yt_2, tested_on="Condition 3", trained_on="Condition 5")
print("")
print(" =================== TRAINED ON CONDITION 3 ===================")
print_results(fnn2, xt_1, yt_1, tested_on="Condition 5", trained_on="Condition 3")
print_results(fnn2, xt_2, yt_2, tested_on="Condition 3", trained_on="Condition 3")
