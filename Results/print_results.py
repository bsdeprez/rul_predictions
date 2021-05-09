import os
import pandas as pd
import matplotlib.pyplot as plt

filepath_FFNN = "FeedForward Neural Network\\Baseline\\Standard"
filepath_TL = "Transfer Learning\\Baseline\\Standard"
filepath_MAML = "MAML\\Baseline\\Standard"

for file_FFNN, file_TL, file_MAML in zip(os.listdir(filepath_FFNN), os.listdir(filepath_TL), os.listdir(filepath_MAML)):
    location_FFNN = os.path.join(os.path.abspath(""), filepath_FFNN, file_FFNN)
    location_TL = os.path.join(os.path.abspath(""), filepath_TL, file_TL)
    location_MAML = os.path.join(os.path.abspath(""), filepath_MAML, file_MAML)
    condition = file_FFNN.split("Condition ")[-1].split(".")[0]
    df_FFNN = pd.read_csv(location_FFNN, sep=";")
    df_TL = pd.read_csv(location_TL, sep=";")
    df_MAML = pd.read_csv(location_MAML, sep=";")

    # Plot r2-score
    ax_r2 = df_FFNN[['epoch', 'r2']].plot(x='epoch', y='r2', label="Feedforward neural network")
    df_TL[['epoch', 'r2']].plot(x='epoch', y="r2", label="Transfer learning", ax=ax_r2)
    df_MAML[['epoch', 'r2']].plot(x='epoch', y='r2', label="Maml", ax=ax_r2)
    plt.title("R2 score - condition {}".format(condition))
    plt.show()

    # Plot mse-score
    ax_mse = df_FFNN[['epoch', 'mse']].plot(x='epoch', y='mse', label="Feedforward neural network")
    df_TL[['epoch', 'mse']].plot(x='epoch', y='mse', label="Transfer learning", ax=ax_mse)
    df_MAML[['epoch', 'mse']].plot(x='epoch', y='mse', label="Maml", ax=ax_mse)
    plt.title("MSE score - condition {}".format(condition))
    plt.show()

    # Plot phm-score
    ax_phm = df_FFNN[['epoch', 'phm']].plot(x='epoch', y='phm', label="Feedforward neural network")
    df_TL[['epoch', 'phm']].plot(x='epoch', y='phm', label="Transfer learning", ax=ax_phm)
    df_MAML[['epoch', 'phm']].plot(x='epoch', y='phm', label="Maml", ax=ax_phm)
    plt.title("PHM score - condition {}".format(condition))
    plt.show()