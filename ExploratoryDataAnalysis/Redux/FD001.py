import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import math
import os
import random

from sklearn.preprocessing import MinMaxScaler


def write_df_to_file(df, file, index=False):
    temp_file = os.path.join(result_path, "temp.txt")
    df.to_csv(temp_file, sep='&', index=index)
    with open(temp_file, 'r') as f:
        lines = f.readlines()
    with open(file, 'w+') as f:
        for line in lines:
            reworked_line = "{}".format(line.split('\n')[0])
            divided = reworked_line.split('&')
            divided = ["${}$ &".format(i) for i in divided]
            reworked_line = "".join(divided)[:-1] + '\\\\ \n'
            f.write(reworked_line)


# Read in the data
filepath = "../../Data/CMAPSSData/"
result_path = "../Results/FD001/"
index_columns = ['unit_nr', 'time_cycles']
setting_columns = ['setting_1', 'setting_2', 'setting_3']
sensor_columns = ["s_{%d}" % i for i in range(1, 22)]
columns = index_columns + setting_columns + sensor_columns
train_df = pd.read_csv(os.path.join(filepath, 'train_FD001.txt'), sep='\s+', header=None, names=columns)
test_df = pd.read_csv(os.path.join(filepath, 'test_FD001.txt'), sep='\s+', header=None, names=columns)
RUL_df = pd.read_csv(os.path.join(filepath, 'RUL_FD001.txt'), sep='\s+', header=None, names=['RUL'])
# Scaling the data
scaler = MinMaxScaler()
train_df[sensor_columns] = pd.DataFrame(scaler.fit_transform(train_df[sensor_columns]))
train_df[sensor_columns] = 2 * train_df[sensor_columns] - 1

# Add RUL columns
max_cycle = train_df.groupby(by='unit_nr')['time_cycles'].max()
result_frame = train_df.merge(max_cycle.to_frame(name='max'), left_on='unit_nr', right_index=True)
train_df['RUL'] = result_frame['max'] - result_frame['time_cycles']

head_FD001 = os.path.join(result_path, "head_FD001.txt")
# write_df_to_file(train_df.head().round(4).T, head_FD001, True)
describe_FD001 = os.path.join(result_path, "describe_FD001.txt")
# write_df_to_file(train_df.describe().round(2).T, describe_FD001, True)

std = train_df[sensor_columns].describe().T.round(4)['std']
plt.bar(std.index, std.values)
plt.close()  # plt.show()

drop_columns = ['s_{1}', 's_{5}', 's_{6}', 's_{10}', 's_{16}', 's_{18}', 's_{19}']
for i in drop_columns:
    sensor_columns.remove(i)
corr_matrix = train_df[sensor_columns+['RUL']].corr()
plt.figure(figsize=(20, 20))
ticks = ["s_{}".format(column.split('}')[0].split('{')[1]) for column in sensor_columns]
sns.heatmap(data=corr_matrix, annot=True, vmin=-1, vmax=1, center=0, cmap='coolwarm', xticklabels=ticks, yticklabels=ticks)
plt.yticks(rotation=0)
plt.close()   # plt.show()

sensor_names = {}
for sensor in sensor_columns:
    x = sensor.split('}')[0].split('{')[1]
    sensor_names[sensor] = "s_{}".format(x)

ncols = 2
nrows = math.ceil(len(sensor_columns)/2)
fig, axes = plt.subplots(nrows, ncols)
fig.set_figheight(15)
fig.set_figwidth(15)
df_list = []
for column in sensor_columns:
    data = [column, 'time_cycles', 'unit_nr']
    df_list.append(train_df[data])

count = 0
for r in range(nrows):
    for c in range(ncols):
        for i in df_list[count]['unit_nr'].unique():
            if i % 10 == 0:
                sensor = df_list[count].columns[0]
                df = df_list[count][df_list[count]['unit_nr'] == i]
                axes[r][c].plot(df['time_cycles'], df[sensor])
                axes[r][c].title.set_text(sensor_names[sensor])
        count += 1
plt.tight_layout()
plt.close()

data = train_df[train_df['unit_nr'] == 1][['RUL', 'time_cycles']]
data_kink = train_df[train_df['unit_nr'] == 1]['RUL'].clip(upper=130)
data.plot(x='RUL', y='time_cycles', label='RUL without upper limit')
data_kink.plot(x='RUL', label='RUL with upper limit')
plt.xlabel('Remaining Useful Life')
plt.ylabel('Time cycles')
plt.legend(loc='upper right')
plt.close()

sns.scatterplot(x=train_df['s_{9}'], y=train_df['s_{14}'])
plt.xlabel('s_9')
plt.ylabel('s_14')
plt.show()
