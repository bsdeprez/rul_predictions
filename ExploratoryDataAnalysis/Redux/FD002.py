import warnings
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import math
import os
import random

filepath = "../../Data/CMAPSSData/"
index_columns = ['unit_nr', 'time_cycles']
setting_columns = ['setting_1', 'setting_2', 'setting_3']
sensor_columns = ['s_{}'.format(i) for i in range(1, 22)]
columns = index_columns + setting_columns + sensor_columns

train_df = pd.read_csv(os.path.join(filepath, 'train_FD002.txt'), sep='\s+', header=None, names=columns)

warnings.filterwarnings('ignore')
sns.jointplot(data=train_df[setting_columns], x='setting_1', y='setting_3', hue='setting_2', palette='coolwarm')
plt.close()

corr_matrix = train_df[setting_columns].corr()
sns.heatmap(data=corr_matrix, annot=True, cmap='coolwarm')
plt.close()

m, b = np.polyfit(train_df['setting_1'], train_df['setting_2'], 1)
plt.plot(train_df['setting_1'], train_df['setting_2'], 'o', label='Datapoints')
plt.plot(train_df['setting_1'], m*train_df['setting_1'] + b, label='Best fit')
plt.xlabel('setting 1')
plt.legend(loc='upper left')
plt.ylabel('setting 2')
plt.show()

