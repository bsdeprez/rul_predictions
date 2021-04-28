import os
import pandas as pd
from RemainingUsefulLifeProject.Data.dataobject import DataObject

filepath = '../Data/CMAPSSData/'
custom_path = "../Data/Custom/"

def save_file(condition, train_or_test, folder, dataset_name, df):
    folder_path = os.path.join(folder, dataset_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    file_path = os.path.join(folder_path, "{}_condition_{}.txt".format(train_or_test, condition))
    pd.DataFrame.to_csv(df, file_path, sep=';', header=True, index=False)


def convert_file(folder, dataset):
    dao = DataObject(dataset)
    train_df = dao.split_on_condition(dao.train_df)
    test_df = dao.split_on_condition(dao.test_df)

    max_cycle = test_df.groupby(by='unit_nr')['time_cycle'].max()
    result_frame = test_df.merge(max_cycle.to_frame(name='max'), left_on='unit_nr', right_index=True)
    truth_df = dao.truth_df
    truth_df['unit_nr'] = truth_df.index + 1
    result_frame = result_frame.merge(truth_df, on="unit_nr")
    test_df['RUL'] = result_frame['max'] - result_frame['time_cycle'] + result_frame['RUL']
    for condition in train_df['condition'].unique():
        train_df_to_save = train_df[train_df['condition'] == condition]
        test_df_to_save = test_df[test_df['condition'] == condition]
        save_file(condition, "train", folder, dataset, train_df_to_save)
        save_file(condition, "test", folder, dataset, test_df_to_save)


convert_file(custom_path, 'FD001')
convert_file(custom_path, 'FD002')
