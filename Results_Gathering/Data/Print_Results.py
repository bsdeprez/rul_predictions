import os
import pandas as pd
import matplotlib.pyplot as plt
from Results_Gathering.Helper_functions.Plotter import __get_directory__

results_folder = "Results", "Experiment 7"
save_folders = "Images", "Experiment 7"
filepath_TL = __get_directory__(*results_folder, "Transfer Learning")
filepath_MAML = __get_directory__(*results_folder, "MAML")
save_path = __get_directory__(*save_folders)

for file_TL, file_MAML in zip(os.listdir(filepath_TL), os.listdir(filepath_MAML)):
    location_TL, location_MAML = os.path.join(filepath_TL, file_TL), os.path.join(filepath_MAML, file_MAML)
    type = file_TL.split(".")[1]
    if type == "csv":
        condition = file_TL.split("Condition ")[-1].split(".")[0]
        df_TL = pd.read_csv(location_TL, sep=";")
        df_MAML = pd.read_csv(location_MAML, sep=";")

        # Get baseline data
        baseline_mse, baseline_phm = [], []
        for epoch in range(len(df_TL['epoch'])):
            x_mse, y_mse = epoch, df_TL['mse'].values[0]
            x_phm, y_phm = epoch, df_TL['phm'].values[0]

        # Plot r2 score
        ax_r2 = df_TL[['epoch', 'r2']].plot(x='epoch', y='r2', label="Transfer Learning")
        df_MAML[['epoch', 'r2']].plot(x='epoch', y='r2', label="MAML", ax=ax_r2)
        plt.title("R2 score - condition {}".format(condition))
        plt.savefig(os.path.join(save_path, "R2 score - Condition {}".format(condition)))
        plt.close()

        # Plot mse-score
        ax_mse = df_TL[['epoch', 'mse']].plot(x='epoch', y='mse', label='Transfer Learning')
        df_MAML[['epoch', 'mse']].plot(x='epoch', y='mse', label='MAML', ax=ax_mse)
        plt.title("MSE Score - Condition {}".format(condition))
        plt.savefig(os.path.join(save_path, "MSE score - Condition {}".format(condition)))
        plt.close()

        # Plot phm-score
        ax_mse = df_TL[['epoch', 'phm']].plot(x='epoch', y='phm', label='Transfer Learning')
        df_MAML[['epoch', 'phm']].plot(x='epoch', y='phm', label='MAML', ax=ax_mse)
        plt.title("PHM Score - Condition {}".format(condition))
        plt.savefig(os.path.join(save_path, "PHM score - Condition {}".format(condition)))
        plt.close()
