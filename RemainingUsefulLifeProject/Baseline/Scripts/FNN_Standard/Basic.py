from RemainingUsefulLifeProject.Data.dataobject import DataObject

filepath = '../../../../Data/CMAPSSData/'
FD001 = DataObject('FD001', filepath=filepath)
FD002 = DataObject('FD002', filepath=filepath)

def get_datasets(data_object):
    x_train, y_train = data_object.train_df[data_object.sensor_names], data_object.train_df['RUL']
    x_test, y_test = data_object.test_df.groupby('unit_nr').last(1)[data_object.sensor_names], \
                     data_object.truth_df['RUL']
    return x_train, y_train, x_test, y_test
