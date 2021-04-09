import pandas as pd
import os

from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler


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
