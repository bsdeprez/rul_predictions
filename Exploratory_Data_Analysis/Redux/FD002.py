import math
import warnings
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import os
import re

from sklearn.preprocessing import MinMaxScaler

'''
Read in the data
'''
filepath = "../../Data/CMAPSSData/"
result_path = "../Results/FD002/"
index_columns = ['unit_nr', 'time_cycles']
setting_columns = ['setting_1', 'setting_2', 'setting_3']
sensor_columns = ['s_{}'.format(i) for i in range(1, 22)]
columns = index_columns + setting_columns + sensor_columns
train_df = pd.read_csv(os.path.join(filepath, 'train_FD002.txt'), sep='\s+', header=None, names=columns)

"""
Plot the different conditions
"""
warnings.filterwarnings('ignore')
sns.jointplot(data=train_df[setting_columns], x='setting_1', y='setting_3', hue='setting_2', palette='coolwarm')
plt.close()
corr_matrix = train_df[setting_columns].corr()
sns.heatmap(data=corr_matrix, annot=True, cmap='coolwarm')
plt.close()
m, b = np.polyfit(train_df['setting_1'], train_df['setting_2'], 1)
plt.plot(train_df['setting_1'], train_df['setting_2'], 'o', label='Datapoints')
plt.plot(train_df['setting_1'], m * train_df['setting_1'] + b, label='Best fit')
plt.xlabel('setting 1')
plt.legend(loc='upper left')
plt.ylabel('setting 2')
plt.close()

"""
Describe FD002
"""
describe_FD002 = os.path.join(result_path, "describe_FD002.txt")
# write_df_to_file(train_df.describe().round(1).T, describe_FD002, True)

"""
Split on conditions
"""
conditions = {'0,0,100': 1, '42,1,100': 2, '25,1,60': 3, '20,1,100': 4, '35,1,100': 5, '10,0,100': 6}
condition_list = []
for setting1, setting2, setting3 in train_df[setting_columns].values:
    key = "{},{},{}".format(round(setting1), round(setting2), round(setting3))
    condition_list.append(conditions[key])
conditions_df = pd.DataFrame(condition_list, columns=['condition'])
train_df = train_df.merge(conditions_df, left_index=True, right_index=True)

"""
Add RUL columns
"""""
max_cycle = train_df.groupby(by='unit_nr')['time_cycles'].max()
result_frame = train_df.merge(max_cycle.to_frame(name='max'), left_on='unit_nr', right_index=True)
train_df['RUL'] = result_frame['max'] - result_frame['time_cycles']

'''
Write the descriptions of each condition to a file
'''
descriptions = []
for condition in np.unique(train_df['condition']):
    df_slice = train_df[train_df['condition'] == condition]
    description = df_slice.describe().round(1)
    description = description.rename(index={'mean': 'mean {}'.format(condition), 'std': 'std {}'.format(condition)}).T
    descriptions.append(description[['mean {}'.format(condition), 'std {}'.format(condition)]])
df = descriptions[0].T
for i in range(1, 3):
    df = df.append(descriptions[i].T)
df = df.T
path_df = os.path.join(result_path, "describe_FD002_per_condition_part_1.txt")
def write_file(data, path):
    data.to_csv("./temp.txt", sep="&")
    with open("./temp.txt", 'r') as file:
        lines = file.readlines()

    lines = [line[:-1] for line in lines]
    write_line = ""
    for line in lines:
        fields = line.split('&')
        fields = [field.replace(" ", "\\ ") for field in fields]
        fields = [re.sub(r"s_([0-9]+)", r"s_{\1}", field) for field in fields]
        fields = ["$ {} $".format(field) for field in fields]
        fields[-1] = "{}\\\\ \n".format(fields[-1])
        result_line = fields[0]
        for i in range(1, len(fields)):
            result_line += " & {}".format(fields[i])
        write_line += result_line

    with open(path, 'w+') as file:
        file.write(write_line)

    os.remove("./temp.txt")


# write_file(df, path_df)
df = descriptions[3].T
for i in range(4, 6):
    df = df.append(descriptions[i].T)
df = df.T
path_df = os.path.join(result_path, "describe_FD002_per_condition_part_2.txt")
# write_file(df, path_df)

"""
Plot a sampled version of the data per condition
"""
drop_columns = ['s_1', 's_5', 's_6', 's_10', 's_15', 's_16', 's_18', 's_19']
for i in drop_columns:
    sensor_columns.remove(i)
datasets = {}
for condition in np.unique(train_df['condition']):
    df_slice = train_df[train_df['condition'] == condition].reset_index()
    scaler = MinMaxScaler()
    df_slice[sensor_columns] = pd.DataFrame(scaler.fit_transform(df_slice[sensor_columns]))
    df_slice[sensor_columns] = 2 * df_slice[sensor_columns] - 1

    n_cols, n_rows = 2, math.ceil(len(sensor_columns)/2)
    fig, axes = plt.subplots(n_rows, n_cols)
    fig.set_figheight(15)
    fig.set_figwidth(15)
    df_list = []
    for column in sensor_columns:
        data = [column, 'RUL', 'unit_nr']
        df_list.append(df_slice[data])
    for r in range(n_rows):
        for c in range(n_cols):
            count = 2*r + c
            if count < len(df_list):
                for unit_nr in df_list[count]['unit_nr'].unique():
                    if unit_nr % 10 == 0:
                        sensor = df_list[count].columns[0]
                        df = df_list[count][df_list[count]['unit_nr'] == unit_nr]
                        axes[r][c].plot(df['RUL'], df[sensor])
                        axes[r][c].set_title("Sensor {}".format(sensor))
            axes[r][c].set_xlim(max(df['RUL']), min(df['RUL']))
    fig.tight_layout()
    print(condition)
    plt.show()

