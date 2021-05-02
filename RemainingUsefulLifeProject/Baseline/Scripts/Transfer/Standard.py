from RemainingUsefulLifeProject.Baseline.Models.FNNs.fnn_standard import FFNModel
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
        train_data, test_data, size = get_data(dao, condition)
        train_dataset[condition] = train_data
        test_dataset[condition] = test_data
        input_size_set.add(size)
    return train_dataset, test_dataset, input_size_set


# Train on FD001
# => score every condition from FD002
# Train on each condition of FD002
#   => score every other condition from FD002, as well as FD001
train_FD001, test_FD001, input_size = make_dataset(FD001, {}, {}, set())
train_FD002, test_FD002, input_size = make_dataset(FD002, {}, {}, input_size)
if len(input_size) != 1:
    print("Multiple input-shapes found!")
    exit(1)
# Start by training on FD001
model = FFNModel(next(iter(input_size)))
model.train(train_FD001["5"], test_FD001["5"], 1)
