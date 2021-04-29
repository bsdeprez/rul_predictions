from RemainingUsefulLifeProject.Baseline.Models.FNNs.fnn_standard import FFNModel
from RemainingUsefulLifeProject.Baseline.Models.MAML.maml import train_maml
from RemainingUsefulLifeProject.Data.dataobject import CustomDataObject
import tensorflow as tf

filepath = '../../../../Data/Custom/'

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
        train_data, test_data, input_size = get_data(dao, condition)
        train_dataset.append(train_data)
        test_dataset.append(test_data)
        input_size_set.add(input_size)
    return train_dataset, test_dataset, input_size_set


train, test, input_sizes = make_dataset(FD001, [], [], set())
train, test, input_sizes = make_dataset(FD002, train, test, input_sizes)
if len(input_sizes) != 1:
    print("Multiple input-shapes found!")
    exit(1)

input_size = next(iter(input_sizes))
ffnn = FFNModel(input_size)

