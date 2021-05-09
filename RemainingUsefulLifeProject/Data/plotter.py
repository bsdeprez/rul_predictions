import matplotlib.pyplot as plt
import os

from RemainingUsefulLifeProject.Data.scoring_methods import r2_score, mse_score, phm_score


def __get_directory__(*args):
    plotter_file = os.path.abspath(__file__)
    base_path = plotter_file.split("RemainingUsefulLifeProject")[0]
    for arg in args:
        base_path = os.path.join(base_path, arg)
    if not os.path.isdir(base_path):
        os.makedirs(base_path)
    return base_path


def plot_history(history, *args, title="", display=False):
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.plot(history.history['loss'], label="training loss")
    plt.plot(history.history['val_loss'], label='validation loss')
    plt.legend(loc='upper right')
    plt.title = title
    folder = __get_directory__(*args)
    plt.savefig(os.path.join(folder, title))
    if display:
        plt.show()
    else:
        plt.close()


def plot_predictions(y_true, y_predicted, *args, title="", display=False):
    plt.plot(range(len(y_true)), y_true, label="True values")
    plt.plot(range(len(y_true)), y_predicted, label="Predicted values")
    plt.title = title
    folder = __get_directory__(*args, "predictions")
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(folder, title))
    if display:
        plt.show()
    else:
        plt.close()


def plot_differences(y_true, y_predicted, *args, title="scores", display=False):
    difference = y_true - y_predicted
    plt.bar(range(len(difference)), height=difference, label="difference")

    plt.ylabel("difference")
    plt.legend(loc='upper right')
    plt.title = title
    folder = __get_directory__(*args, "differences")
    plt.savefig(os.path.join(folder, title))
    if display:
        plt.show()
    else:
        plt.close()


def print_scores(y_true, y_predicted, *args, title="", display=False, header="", new_file=True):
    folder = __get_directory__(*args)
    r2 = r2_score(y_true, y_predicted)
    mse = mse_score(y_true, y_predicted)
    phm = phm_score(y_true, y_predicted)
    filename = os.path.join(folder, "{}.txt".format(title))
    output = "{}\nmse: {}\nr2: {}\nphm: {}\n".format(header, mse, r2, phm)
    extension = 'a+'
    if new_file: extension = 'w+'
    with open(filename, extension) as file:
        file.write(output)
    if display: print(output)
