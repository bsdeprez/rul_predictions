import sklearn.metrics as m
import numpy as np
import os

from RemainingUsefulLifeProject.Data.plotter import __get_images_directory__


def r2_score(y_true, y_predicted):
    return m.r2_score(y_true, y_predicted)

def phm_score(y_true, y_predicted):
    d = y_predicted - y_true
    d = np.where(d < 0, -d/10, d/13)
    s = np.exp(d) - 1
    return np.sum(s)

def mse_score(y_true, y_predicted):
    return m.mean_squared_error(y_true, y_predicted)

def print_scores(y_true, y_predicted, dataset, location="", title=""):
    print(" ============= {} =============".format(dataset))
    print(" R2: {}".format(r2_score(y_true, y_predicted)))
    print(" MSE: {}".format(mse_score(y_true, y_predicted)))
    print(" PHM: {}".format(phm_score(y_true, y_predicted)))
    # If a location was given, save the scores to a txt.file
    if len(location) != 0:
        location = __get_images_directory__(location) + "\\{}.txt".format(title)
        with open(location, 'w+') as file:
            file.write(" ============= {} =============\n".format(dataset))
            file.write(" R2: {}\n".format(r2_score(y_true, y_predicted)))
            file.write(" MSE: {}\n".format(mse_score(y_true, y_predicted)))
            file.write(" PHM: {}\n".format(phm_score(y_true, y_predicted)))
