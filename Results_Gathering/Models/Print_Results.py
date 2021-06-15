import os
import pandas as pd
import matplotlib.pyplot as plt
from Results_Gathering.Helper_functions.Plotter import __get_directory__

results_folder = "Results", "Standard Model"
filepath_TL = __get_directory__(*results_folder, "Transfer Learning")
filepath_MAML = __get_directory__(*results_folder, "MAML")

for file_TL, file_MAML in zip(os.listdir(filepath_TL), os.listdir(filepath_MAML)):
    location_TL, location_MAML = os.path.join(filepath_TL, file_TL), os.path.join(filepath_MAML, file_MAML)
    condition = file_TL.split("Condition ")[-1].split(".")[0]
    df_TL = pd.read_csv(location_TL, sep=";")
    df_MAML = pd.read_csv(location_MAML, sep=";")

    # Get baseline data
    baseline_mse, baseline_phm = [], []
    for epoch in range(len(df_TL['epoch'])):
        x_mse, y_mse = epoch, df_TL['mse'].values[0]
        x_phm, y_phm = epoch, df_TL['phm'].values[0]
        baseline_mse.append((x_mse, y_mse))
        baseline_phm.append((x_phm, y_phm))
    df_baseline_mse = pd.DataFrame(baseline_mse, columns=['epoch', 'mse'])
    df_baseline_phm = pd.DataFrame(baseline_phm, columns=['epoch', 'phm'])

    # Plot r2 score
    ax_r2 = df_TL[['epoch', 'r2']].plot(x='epoch', y='r2', label="Transfer Learning")
    df_MAML[['epoch', 'r2']].plot(x='epoch', y='r2', label="MAML", ax=ax_r2)
    plt.title("R2 score - condition {}".format(condition))
    plt.show()

    # Plot mse-score
    ax_mse = df_TL[['epoch', 'mse']].plot(x='epoch', y='mse', label='Transfer Learning')
    df_MAML[['epoch', 'mse']].plot(x='epoch', y='mse', label='MAML', ax=ax_mse)
    df_baseline_mse[['epoch', 'mse']].plot(x='epoch', y='mse', label='Baseline', ax=ax_mse)
    plt.title("MSE Score - Condition {}".format(condition))
    plt.show()

    # Plot phm-score
    ax_mse = df_TL[['epoch', 'phm']].plot(x='epoch', y='phm', label='Transfer Learning')
    df_MAML[['epoch', 'phm']].plot(x='epoch', y='phm', label='MAML', ax=ax_mse)
    df_baseline_phm[['epoch', 'phm']].plot(x='epoch', y='phm', label='Baseline', ax=ax_mse)
    plt.title("PHM Score - Condition {}".format(condition))
    plt.show()
