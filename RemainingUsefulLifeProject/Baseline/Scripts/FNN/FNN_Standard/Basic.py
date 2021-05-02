from RemainingUsefulLifeProject.Baseline.Models.FNNs.fnn_standard import FFNModel
from RemainingUsefulLifeProject.Data.dataobject import CustomDataObject
from RemainingUsefulLifeProject.Data.plotter import plot_predicted_v_true, plot_difference
from RemainingUsefulLifeProject.Data.scoring_methods import print_scores
import tensorflow as tf

filepath = '../../../../../Data/Custom/'
FD001 = CustomDataObject('FD001', filepath)
FD002 = CustomDataObject('FD002', filepath)


def get_data(dao, condition):
    x_train, y_train = dao.train_dfs[condition][dao.sensor_names], dao.train_dfs[condition]['RUL']
    test_df = dao.test_dfs[condition].groupby(by='unit_nr').last()
    x_test, y_test = test_df[dao.sensor_names], test_df['RUL']
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    return train_dataset, test_dataset, x_test.shape[1]


def make_dataset(dao, train_dataset, test_dataset, input_size_set):
    for condition in dao.conditions:
        train_data, test_data, size = get_data(dao, condition)
        train_dataset[condition] = train_data
        test_dataset[condition] = test_data
        input_size_set.add(size)
    return train_dataset, test_dataset, input_size_set

def print_results(model, test_dataset, tested_on, trained_on):
    truth, predicted = model.predict(test_dataset)
    plot_predicted_v_true(truth, predicted, "Accuracy {}".format(tested_on),
                          "Standard FFN\\Standard\\Trained on {}".format(trained_on), show=False)
    plot_difference(truth, predicted, "Difference distribution {}".format(tested_on),
                    "Standard FFN\\Standard\\Trained on {}".format(trained_on), show=False)
    print_scores(truth, predicted, tested_on, "Standard FFN\\Standard\\Trained on {}".format(trained_on), tested_on)


train_FD001, test_FD001, input_size = make_dataset(FD001, {}, {}, set())
train_FD002, test_FD002, input_size = make_dataset(FD002, {}, {}, input_size)
if len(input_size) != 1:
    print("Multiple input-shapes found...")
    exit(1)

size = next(iter(input_size))
ffn = FFNModel(size)
ffn.train(train_FD001["5"], epochs=20)
conditions = list(FD002.conditions)
conditions.sort()
for test_cond in conditions:
    print_results(ffn, test_FD002[test_cond], tested_on="FD002 - Condition {}".format(test_cond), trained_on="FD001 - Condition 5")
for train_cond in conditions:
    fnn = FFNModel(size)
    fnn.train(train_FD002[train_cond], epochs=50)
    for test_cond in conditions:
        if test_cond != train_cond:
            print_results(ffn, test_FD002[test_cond], tested_on="FD002 - Condition {}".format(test_cond),
                          trained_on="FD002 - Condition {}".format(train_cond))
    print_results(ffn, test_FD001['5'], tested_on="FD001 - Condition 5",
                  trained_on="FD002 - Condition {}".format(train_cond))
