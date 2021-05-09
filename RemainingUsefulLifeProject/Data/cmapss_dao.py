import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class DataObject:

    def __init__(self, dataset, filepath='../Data/CMAPSSData/', normalize=True):
        self.name = dataset
        train_file = os.path.join(filepath, "train_{}.txt".format(dataset))
        test_file = os.path.join(filepath, "test_{}.txt".format(dataset))
        truth_file = os.path.join(filepath, "RUL_{}.txt".format(dataset))
        for file in (train_file, test_file, truth_file):
            if not os.path.exists(file):
                raise FileNotFoundError("{} doesn't exist".format(file))

        self.index_names = ['unit_nr', 'time_cycle']
        self.setting_names = ['setting_{}'.format(i) for i in range(1, 4)]
        self.sensor_names = ['s_{}'.format(i) for i in range(1, 22)]

        column_names = self.index_names + self.setting_names + self.sensor_names
        self.train_df = pd.read_csv(train_file, sep='\s+', header=None, names=column_names)
        self.test_df = pd.read_csv(test_file, sep='\s+', header=None, names=column_names)
        self.truth_df = pd.read_csv(truth_file, sep='\s+', header=None, names=['RUL'])

        self.__create_condition_column__()
        self.conditions = self.train_df['condition'].unique()
        self.__create_RUL_column__()
        self.datasets = {}
        if normalize:
            self.normalize()
        else:
            self.split_in_dataset()

    def add_kinking_point(self, kink):
        for condition in self.conditions:
            train, test = self.datasets[condition]
            train['RUL'].clip(upper=kink, inplace=True)
            test['RUL'].clip(upper=kink, inplace=True)
            self.datasets[condition] = (train, test)

    def drop_columns(self, columns):
        for column in columns:
            if column in self.sensor_names: self.sensor_names.remove(column)
            if column in self.setting_names: self.setting_names.remove(column)
            if column in self.index_names: self.index_names.remove(column)
        for condition in self.conditions:
            train, test = self.datasets[condition]
            train.drop(columns, axis=1, errors='ignore', inplace=True)
            test.drop(columns, axis=1, errors='ignore', inplace=True)
            self.datasets[condition] = (train, test)

    def normalize(self):
        for condition in self.conditions:
            train_df = self.train_df[self.train_df['condition'] == condition].reset_index()
            test_df = self.test_df[self.test_df['condition'] == condition].reset_index()
            train_norm, test_norm = self.__normalize__(train_df, test_df)
            train_df[self.sensor_names], test_df[self.sensor_names] = train_norm, test_norm
            self.datasets[condition] = (train_df, test_df)

    def split_in_dataset(self):
        for condition in self.conditions:
            train_df = self.train_df[self.train_df['condition'] == condition].reset_index()
            test_df = self.test_df[self.test_df['condition'] == condition].reset_index()
            self.datasets[condition] = (train_df, test_df)

    def __create_RUL_column__(self):
        # RUL for train_df
        max_cycle = self.train_df.groupby(by='unit_nr')['time_cycle'].max()
        result_frame = self.train_df.merge(max_cycle.to_frame(name='max'), left_on='unit_nr', right_index=True)
        self.train_df['RUL'] = result_frame['max'] - result_frame['time_cycle']
        self.train_df = self.train_df.astype({'RUL': 'float32'})
        # RUL for test_df
        max_cycle = self.test_df.groupby(by='unit_nr')['time_cycle'].max()
        result_frame = self.test_df.merge(max_cycle.to_frame(name='max'), left_on='unit_nr', right_index=True)
        self.truth_df['unit_nr'] = self.truth_df.index + 1
        result_frame = result_frame.merge(self.truth_df, on="unit_nr")
        self.test_df['RUL'] = result_frame['max'] - result_frame['time_cycle'] + result_frame['RUL']
        self.test_df = self.test_df.astype({'RUL': 'float32'})

    def __create_condition_column__(self):
        conditions = {'0,0,100': 1, '42,1,100': 2, '25,1,60': 3, '20,1,100': 4, '35,1,100': 5, '10,0,100': 6}

        def __split_on_conditions__(df):
            condition_list = []
            for setting1, setting2, setting3 in df[self.setting_names].values:
                key = "{},{},{}".format(round(setting1), round(setting2), round(setting3))
                condition_list.append(conditions[key])
            conditions_df = pd.DataFrame(condition_list, columns=['condition'])
            return df.merge(conditions_df, left_index=True, right_index=True)

        self.train_df = __split_on_conditions__(self.train_df)
        self.test_df = __split_on_conditions__(self.test_df)

    def __normalize__(self, train_df, test_df):
        scaler = MinMaxScaler()
        scaler.fit(train_df[self.sensor_names])
        train_df, test_df = train_df.copy(), test_df.copy()
        train_df = 2 * pd.DataFrame(scaler.transform(train_df[self.sensor_names])) - 1
        test_df = 2 * pd.DataFrame(scaler.transform(test_df[self.sensor_names])) - 1
        return train_df, test_df
