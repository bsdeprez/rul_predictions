import math
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import re
from sklearn.cluster import KMeans

'''
Read in the data
'''
filepath = "../../Data/CMAPSSData/"
result_path = "../Results/FD003/"
index_columns = ['unit_nr', 'time_cycles']
setting_columns = ['setting_1', 'setting_2', 'setting_3']
sensor_columns = ['s_{}'.format(i) for i in range(1, 22)]
columns = index_columns + setting_columns + sensor_columns
train_df = pd.read_csv(os.path.join(filepath, 'train_FD003.txt'), sep='\s+', header=None, names=columns)
# Add RUL columns
max_cycle = train_df.groupby(by='unit_nr')['time_cycles'].max()
result_frame = train_df.merge(max_cycle.to_frame(name='max'), left_on='unit_nr', right_index=True)
train_df['RUL'] = result_frame['max'] - result_frame['time_cycles']

"""
Print the description
"""
description = train_df.describe().round(1).T
description.to_csv("./temp.txt", sep='&')
with open("./temp.txt", 'r') as file:
    lines = file.readlines()
write_line = ""
for line in lines:
    fields = line.split('&')
    fields = [field.replace(" ", "\\ ") for field in fields]
    fields = [re.sub(r"s_([0-9]+)", r"s_{\1}", field) for field in fields]
    fields = ["$ {} $".format(field) for field in fields]
    fields[-1] = "{}\\\\ \n".format(fields[-1])
    results_line = fields[0]
    for i in range(1, len(fields)):
        results_line += " & {}".format(fields[i])
    write_line += results_line
with open("../Results/FD003/describe_FD003.txt", 'w+') as file:
    file.write(write_line)
os.remove("./temp.txt")

"""
Create the correlation matrix
"""
drop_columns = ['s_1', 's_5', 's_6', 's_10', 's_16', 's_18', 's_19']
for i in drop_columns:
    sensor_columns.remove(i)
corr_matrix = train_df[sensor_columns + ['RUL']].corr()
plt.figure(figsize=(20, 20))
ticks = sensor_columns + ['RUL']
sns.heatmap(data=corr_matrix, annot=True, vmin=-1, vmax=1, center=0, cmap='coolwarm', xticklabels=ticks,
            yticklabels=ticks)
plt.yticks(rotation=0)
plt.close()
# plt.show()

"""
Check correlation between
    s_7 and s_12
    s_8 and s_13
    s_9 and s_14
"""


def check_corr(sensor_1, sensor_2):
    sns.scatterplot(x=train_df[sensor_1], y=train_df[sensor_2])
    plt.xlabel(sensor_1), plt.ylabel(sensor_2)
    plt.close()


check_corr('s_7', 's_12')
check_corr('s_8', 's_13')
check_corr('s_9', 's_14')

drop_columns = ['s_7', 's_8', 's_9']
for i in drop_columns:
    sensor_columns.remove(i)
"""
Create the scatter plots
"""
n_cols, n_rows = 2, math.ceil(len(sensor_columns) / 2)
fig, axes = plt.subplots(n_rows, n_cols, squeeze=False)
fig.set_figheight(15)
fig.set_figwidth(15)
df_list = []
for column in sensor_columns:
    data = [column, 'RUL', 'unit_nr']
    df_list.append(train_df[data])
for row in range(0, n_rows):
    for col in range(0, n_cols):
        count = 2 * row + col
        if count < len(df_list):
            for i in df_list[count]['unit_nr'].unique():
                if i % 10 == 0:
                    sensor = df_list[count].columns[0]
                    df = df_list[count][df_list[count]['unit_nr'] == i]
                    axes[row][col].plot(df['RUL'], df[sensor])
                    axes[row][col].title.set_text(sensor)
            axes[row][col].invert_xaxis()
plt.tight_layout()
plt.close()

"""
Clustering
"""
df_kmeans = train_df.copy()
sensors = ['s_15', 's_20', 's_21']
km = KMeans(n_clusters=2, init='k-means++')
y_km = km.fit_predict(df_kmeans[sensors])
df_kmeans['cluster'] = pd.DataFrame(y_km)

fig, axes = plt.subplots(3, 1, squeeze=False)
fig.set_figheight(15), fig.set_figwidth(15)
for row in range(0, 3):
    sensor = sensors[row]
    axes[row][0].scatter(df_kmeans['RUL'], df_kmeans[sensor], c=df_kmeans['cluster'], cmap='Set1')
    axes[row][0].invert_xaxis()
    axes[row][0].title.set_text(sensor)
plt.tight_layout()
plt.show()