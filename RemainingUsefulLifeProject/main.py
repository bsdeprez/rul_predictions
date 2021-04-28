import random
from RemainingUsefulLifeProject.Data.dataobject import CustomDataObject
from RemainingUsefulLifeProject.Data.plotter import *
from RemainingUsefulLifeProject.MAML.MAML import MAML


def get_data(data_object, N):
    train = data_object.train_list[1]
    test = data_object.test_list[2]

    x_train = train[data_object.sensor_names].values
    y_train = train['RUL'].values
    x_test = test[data_object.sensor_names].values
    y_test = test['RUL'].values
    return x_train, y_train, x_test, y_test

def test_data(model, x_test, y_test, dataset):
    y_predicted = model.predict(x_test).flatten()
    plot_predicted_v_true(y_test, y_predicted, "Accuracy MAML", "MAML\\Standard", show=True)
    plot_difference(y_test, y_predicted, "Difference MAML", "MAML\\Standard", show=True)


dao = CustomDataObject()
x_s, y_s, x_t, y_t = get_data(dao, 5)
maml = MAML(x_s.shape[1])
maml.train(x_s, y_s, x_t, y_t, 500)
test_data(maml.model, x_t, y_t, "b")

