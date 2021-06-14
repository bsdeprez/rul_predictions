import os
import matplotlib.pyplot as plt


def __get_directory__(*args):
    plotter_file = os.path.abspath(__file__)
    base_path = plotter_file.split("Results_Gathering")[0]
    for arg in args:
        base_path = os.path.join(base_path, arg)
    if not os.path.isdir(base_path):
        os.makedirs(base_path)
    return base_path
