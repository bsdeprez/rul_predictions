import matplotlib.pyplot as plt
import os


def plot_difference(y_true, y_predicted, title="", folder="", show=True):
    folder = __get_images_directory__(folder)
    d = y_predicted - y_true
    plt.hist(d)
    plt.title(title)
    plt.savefig(folder + title)
    if show:
        plt.show()
    else:
        plt.close()

def __get_images_directory__(folder):
    base = os.getcwd().split('\\rul_predictions')[0] + "\\rul_predictions"
    folder = "{}\\Images\\{}\\".format(base, folder)
    if not os.path.isdir(folder):
        os.makedirs(folder)
    return folder


def plot_predicted_v_true(y_true, y_predicted, title="", folder="", show=True):
    folder = __get_images_directory__(folder)
    plt.scatter(y_true, y_predicted, c='crimson')
    p1 = max(max(y_predicted), max(y_true))
    p2 = min(min(y_predicted), min(y_true))
    plt.plot([p1, p2], [p1, p2], 'b-')
    plt.gca().invert_xaxis()
    plt.xlabel('True values')
    plt.ylabel('Predictions')
    plt.axis('equal')
    plt.title(title)
    plt.savefig(folder+title)
    if show:
        plt.show()
    else:
        plt.close()

def plot_history(history):
    plt.figure(figsize=(13,5))
    plt.plot(range(1, len(history.history['loss'])+1), history.history['loss'], label='train')
    plt.plot(range(1, len(history.history['val_loss'])+1), history.history['val_loss'], label='validate')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()