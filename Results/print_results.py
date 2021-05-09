import os
import pandas as pd
import matplotlib.pyplot as plt

filepath = "MAML\\Baseline\\Standard"
for file in os.listdir(filepath):
    location = os.path.join(os.path.abspath(""), filepath, file)
    df = pd.read_csv(location, sep=";")
    condition = file.split("Condition ")[-1].split(".")[0]

    r2 = df[['epoch', 'r2']]
    mse = df[['epoch', 'mse']]
    phm = df[['epoch', 'phm']]
    r2.plot(x='epoch', y='r2', title='r2 score - {}'.format(condition))
    plt.show()
    mse.plot(x='epoch', y='mse', title='mse score - {}'.format(condition))
    plt.show()
    phm.plot(x="epoch", y="phm", title="phm score - {}".format(condition))
    plt.show()
