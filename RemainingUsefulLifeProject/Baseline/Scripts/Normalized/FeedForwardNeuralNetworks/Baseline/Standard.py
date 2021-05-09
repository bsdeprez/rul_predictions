"""
THINGS TO EDIT WHEN COPYING THIS SCRIPT:
Imported model
arguments in save_results
"""
from RemainingUsefulLifeProject.Baseline.Models.FeedForwardNeuralNetworks.Baseline import FFNModel
from RemainingUsefulLifeProject.Data.cmapss_dao import DataObject
from RemainingUsefulLifeProject.Data.plotter import plot_history, plot_predictions, plot_differences, print_scores
import numpy as np


def get_data(train_df, test_df, dao):
    train_x, train_y = train_df[dao.sensor_names].values, train_df['RUL'].values
    test_df = test_df.groupby(by='unit_nr').last().reset_index()
    test_x, test_y = test_df[dao.sensor_names].values, test_df['RUL'].values
    if len(test_y) > 100:
        index = np.random.choice(len(test_y), 100, replace=False)
        test_x, test_y = test_x[index], test_y[index]
    return train_x, train_y, test_x, test_y


def save_results(his, y_true, y_pred, trained_on="", tested_on="", new_file=True):
    folders = "Images", "Feedforward Neural Networks", "Baseline", "Normalised", "Standard"
    if his:
        plot_history(his, *folders, title='History Training {}'.format(trained_on))
    plot_predictions(y_true, y_pred, *folders,
                     title="Predictions - Trained on {}, tested on {}".format(trained_on, tested_on))
    plot_differences(y_true, y_pred, *folders,
                     title='Differences - Trained on {}, tested on {}'.format(trained_on, tested_on))
    print_scores(y_true, y_pred, *folders, title="Scores - Trained on {}".format(trained_on, tested_on),
                 header="==== Trained on {}, Tested on {} ====".format(trained_on, tested_on), new_file=new_file)


def train_model(dao_1, dao_2, train_dao, train_condition):
    # Train the model
    train, test = train_dao.datasets[train_condition]
    x_train, y_train, _, _ = get_data(train, test, train_dao)
    model = FFNModel(len(train_dao.sensor_names), lr=0.15)
    history = model.train(x_train, y_train, epochs=50)

    for i, condition in enumerate(dao_1.conditions):
        train, test = dao_1.datasets[condition]
        _, _, x_test, y_test = get_data(train, test, dao_1)
        predicted = model.predict(x_test).flatten()
        if i == 0:
            save_results(history, y_test, predicted, "{} - Condition {}".format(train_dao.name, train_condition),
                         "{} - Condition {}".format(dao_1.name, condition))
        else:
            save_results(None, y_test, predicted, "{} - Condition {}".format(train_dao.name, train_condition),
                         "{} - Condition {}".format(dao_1.name, condition), new_file=False)
    for condition in dao_2.conditions:
        train, test = dao_2.datasets[condition]
        _, _, x_test, y_test = get_data(train, test, dao_2)
        predicted = model.predict(x_test).flatten()
        save_results(None, y_test, predicted, "{} - Condition {}".format(train_dao.name, train_condition),
                     "{} - Condition {}".format(dao_2.name, condition), new_file=False)


# =====================================================================================================================#
# Firstly: train the model on FD001 and test it.
filepath = "../../../../../../Data/CMAPSSData/"
FD001 = DataObject("FD001", filepath=filepath)
FD002 = DataObject("FD002", filepath)

train_model(FD001, FD002, FD001, 1)
# =====================================================================================================================#
# Secondly: train the model on each FD002 condition
for cond in FD002.conditions:
    train_model(FD001, FD002, FD002, cond)