import pandas as pd
import os
import re
import numpy as np
import math
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

def batch_generator(x, y, batch_size):
    permutations = np.random.permutation(len(x))
    x, y = x[permutations], y[permutations]
    number_of_batches = math.floor(len(x)/batch_size)
    for i in range(number_of_batches):
        start = batch_size*i
        stop = batch_size*(i+1)
        yield x[start:stop], y[start:stop]

def split_in_validation_set(x, y, val_split):
    permutations = np.random.permutation(len(x))
    x, y = x[permutations], y[permutations]
    split = math.floor(val_split*len(x))
    return x[split:], y[split:], x[:split], y[:split]

def get_data_per_condition(dao, condition):
    train_data, test_data = dao.train_dfs[condition], dao.test_dfs[condition]
    train_sensors, test_sensors = train_data[dao.sensor_names], test_data[dao.sensor_names]
    train_rul, test_rul = train_data["RUL"], test_data["RUL"]
    train_sensors, train_rul = np.array(train_sensors.values, dtype=np.float32), np.array(train_rul.values, dtype=np.float32)
    test_sensors, test_rul = np.array(test_sensors.values, dtype=np.float32), np.array(test_rul.values, dtype=np.float32)
    return (train_sensors, train_rul), (test_sensors, test_rul)

def get_dataset(dao):
    train, test = {}, {}
    for condition in dao.conditions:
        train_data, test_data = get_data_per_condition(dao, condition)
        train[condition] = train_data
        test[condition] = test_data
    return train, test

class DataObject:
    def __init__(self, dataset, filepath='../Data/CMAPSSData/'):
        self.name = dataset
        files = {
            'train_file': "{}train_{}.txt".format(filepath, dataset),
            'test_file': "{}test_{}.txt".format(filepath, dataset),
            'truth_file': "{}RUL_{}.txt".format(filepath, dataset)
        }
        for _, file in files.items():
            if not os.path.exists(file):
                raise FileNotFoundError("{} doesn't exist".format(file))
        self.index_names = ['unit_nr', 'time_cycle']
        self.setting_names = ['setting_{}'.format(i) for i in range(1, 4)]
        self.sensor_names = ['s_{}'.format(i) for i in range(1, 22)]
        column_names = self.index_names + self.setting_names + self.sensor_names
        self.train_df = pd.read_csv(files['train_file'], sep='\s+', header=None, names=column_names)
        self.test_df = pd.read_csv(files['test_file'], sep='\s+', header=None, names=column_names)
        self.truth_df = pd.read_csv(files['truth_file'], sep='\s+', header=None, names=['RUL'])

        self.train_df = self.__create_RUL_column__(self.train_df)
        self.__normalize__(self.sensor_names)

    def __normalize__(self, columns):
        scaler = MinMaxScaler()
        scaler.fit(self.train_df[columns])
        self.train_df[columns] = 2 * pd.DataFrame(scaler.transform(self.train_df[columns])) - 1
        self.test_df[columns] = 2 * pd.DataFrame(scaler.transform(self.test_df[columns])) - 1

    def drop_columns(self, columns):
        for column in columns:
            if column in self.sensor_names: self.sensor_names.remove(column)
            if column in self.setting_names: self.setting_names.remove(column)
            if column in self.index_names: self.index_names.remove(column)
        self.train_df.drop(columns, axis=1, errors='ignore', inplace=True)
        self.test_df.drop(columns, axis=1, errors='ignore', inplace=True)
        self.truth_df.drop(columns, axis=1, errors='ignore', inplace=True)

    def add_kinking_function(self, kink):
        self.train_df['RUL'].clip(upper=kink, inplace=True)
        self.truth_df['RUL'].clip(upper=kink, inplace=True)

    @staticmethod
    def __create_RUL_column__(df):
        max_cycle = df.groupby(by='unit_nr')['time_cycle'].max()
        result_frame = df.merge(max_cycle.to_frame(name='max'), left_on='unit_nr', right_index=True)
        df['RUL'] = result_frame['max'] - result_frame['time_cycle']
        return df

    def split_on_fault_modes(self):
        sensors = ['s_15', 's_20', 's_21']
        copy = self.train_df.copy()
        km = KMeans(n_clusters=2, init='k-means++')
        y_km = km.fit_predict(copy[sensors])
        copy['cluster'] = pd.DataFrame(y_km)
        series = copy[copy['RUL'] <= 15].groupby(by='unit_nr')['cluster'].mean()
        copy = pd.DataFrame(series).rename(columns={0: 'unit_nr', 1: 'cluster'})
        self.train_df = self.train_df.merge(copy, on='unit_nr')

    def split_on_condition(self, df):
        conditions = {'35,1,100': 1, '42,1,100': 2, '25,1,60': 3, '20,1,100': 4, '0,0,100': 5, '10,0,100': 6}
        copy_df = df.copy()
        condition_list = []
        for setting1, setting2, setting3 in df[self.setting_names].values:
            key = "{},{},{}".format(round(setting1), round(setting2), round(setting3))
            condition_list.append(conditions[key])
        conditions_df = pd.DataFrame(condition_list, columns=['condition'])
        copy_df = copy_df.merge(conditions_df, left_index=True, right_index=True)
        return copy_df

class CustomDataObject:

    def __init__(self, dataset, filepath='../Data/Custom/'):
        self.index_names = ['unit_nr', 'time_cycle']
        self.setting_names = ['setting_{}'.format(i) for i in range(1, 4)]
        self.sensor_names = ['s_{}'.format(i) for i in range(1, 22)]
        self.train_dfs = {}
        self.test_dfs = {}
        filepath = os.path.join(filepath, dataset)
        self.conditions = set()
        for file in os.listdir(filepath):
            match = re.match(r".+_condition_([0-9]+).+", file)
            if match:
                self.conditions.add(match.group(1))
        for condition in self.conditions:
            train_file = "train_condition_{}.txt".format(condition)
            test_file = "test_condition_{}.txt".format(condition)
            train_df = pd.read_csv(os.path.join(filepath, train_file), sep=';')
            test_df = pd.read_csv(os.path.join(filepath, test_file), sep=';')
            self.train_dfs[condition] = train_df
            self.test_dfs[condition] = test_df

    def add_kinking_function(self, kink):
        for condition in self.conditions:
            self.train_dfs[condition]['RUL'].clip(upper=kink, inplace=True)
            self.test_dfs[condition]['RUL'].clip(upper=kink, inplace=True)

    def drop_columns(self, columns):
        for column in columns:
            if column in self.sensor_names: self.sensor_names.remove(column)
            if column in self.setting_names: self.setting_names.remove(column)
            if column in self.index_names: self.index_names.remove(column)
        for condition in self.conditions:
            self.train_dfs[condition].drop(columns, axis=1, errors='ignore', inplace=True)
            self.test_dfs[condition].drop(columns, axis=1, errors='ignore', inplace=True)
