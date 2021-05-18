import numpy as np

from RemainingUsefulLifeProject.Baseline.Models.FNNs.Baseline import FFNModel
from RemainingUsefulLifeProject.Data.cmapss_dao import DataObject
from RemainingUsefulLifeProject.Data.plotter import *
from RemainingUsefulLifeProject.Data.plotter import __get_directory__


def get_data(train_df, test_df, dao):
    train_x, train_y = train_df[dao.sensor_names].values, train_df['RUL'].values
    test_df = test_df.groupby(by='unit_nr').last().reset_index()
    test_x, test_y = test_df[dao.sensor_names].values, test_df['RUL'].values
    if len(test_y) > 100:
        index = np.random.choice(len(test_y), 100, replace=False)
        test_x, test_y = test_x[index], test_y[index]
    return train_x, train_y, test_x, test_y

def write_gathered_scores(scores, title="Scores Baseline"):
    folders = "Results", "Standard Model", "Classic Neural Network", "Baseline Model", "Standard"
    folder = __get_directory__(*folders)
    for key in scores.keys():
        file_title = "{} - Condition {}.csv".format(title, key)
        file_location = os.path.join(folder, file_title)
        r2_sc, mse_sc, phm_sc = scores[key]['r2'], scores[key]['mse'], scores[key]['phm']
        with open(file_location, 'w+') as file:
            file.write("epoch;r2;mse;phm\n")
            for epoch in range(len(r2_sc)):
                file.write("{};{};{};{}\n".format(epoch, r2_sc[epoch], mse_sc[epoch], phm_sc[epoch]))

def save_results(y_true, y_pred, trained_on="", tested_on="", new_file=True):
    folders = "Images", "Standard Model", "Classic Neural Network", "Baseline Model", "Standard"
    plot_predictions(y_true, y_pred, *folders,
                     title="Predictions - Trained on {}, tested on {}".format(trained_on, tested_on))
    plot_differences(y_true, y_pred, *folders,
                     title='Differences - Trained on {}, tested on {}'.format(trained_on, tested_on))
    print_scores(y_true, y_pred, *folders, title="Scores - Trained on {}".format(trained_on, tested_on),
                 header="==== Trained on {}, Tested on {} ====".format(trained_on, tested_on), new_file=new_file)

def train_model(dao_1, dao_2, train_dao, train_condition, scores):
    train, test = train_dao.datasets[train_condition]
    x_train, y_train, x_test, y_test = get_data(train, test, train_dao)
    model = FFNModel(len(train_dao.sensor_names), lr=0.05)
    for epoch in range(50):
        model.train(x_train, y_train, 1)
        predicted = model.predict(x_test)
        r2_sc, mse_sc, phm_sc = r2_score(y_test, predicted), mse_score(y_test, predicted), phm_score(y_test, predicted)
        if scores is not None:
            scores[train_condition]['r2'].append(r2_sc)
            scores[train_condition]['mse'].append(mse_sc)
            scores[train_condition]['phm'].append(phm_sc)

    for i, cond in enumerate(dao_1.conditions):
        train, test = dao_1.datasets[cond]
        _, _, x_test, y_test = get_data(train, test, dao_1)
        predicted = model.predict(x_test).flatten()
        if i == 0:
            save_results(y_test, predicted, "{} - Condition {}".format(train_dao.name, train_condition),
                         "{} - Condition {}".format(dao_1.name, cond))
        else:
            save_results(y_test, predicted, "{} - Condition {}".format(train_dao.name, train_condition),
                         "{} - Condition {}".format(dao_1.name, cond), new_file=False)
    for cond in dao_2.conditions:
        train, test = dao_2.datasets[cond]
        _, _, x_test, y_test = get_data(train, test, dao_2)
        predicted = model.predict(x_test).flatten()
        save_results(y_test, predicted, "{} - Condition {}".format(train_dao.name, train_condition),
                     "{} - Condition {}".format(dao_2.name, cond), new_file=False)


filepath = "../../../../../Data/CMAPSSData/"
FD001 = DataObject("FD001", filepath=filepath)
FD002 = DataObject("FD002", filepath=filepath)

gathered_scores = {}
for condition in FD002.conditions:
    gathered_scores[condition] = {'r2': [], 'phm': [], 'mse': []}

print(" ============================= TRAINING FD001 Condition 1 ============================= ")
train_model(FD001, FD002, FD001, 1, None)
for condition in FD002.conditions:
    print(" ============================= TRAINING FD002 Condition {} ============================= ".format(condition))
    train_model(FD001, FD002, FD002, condition, gathered_scores)

write_gathered_scores(gathered_scores)