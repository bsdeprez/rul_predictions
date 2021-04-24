import os
import pandas as pd

from RemainingUsefulLifeProject.Data.dataobject import DataObject

train_path = "Data/Custom/Train/"
test_path = "Data/Custom/Test/"

FD002 = DataObject("FD002", filepath="Data/CMAPSSData/")
FD002.__normalize__(FD002.sensor_names)
RUL = FD002.truth_df['RUL']
test_df = FD002.test_df.groupby('unit_nr').last(1).reset_index()
test_df['RUL'] = RUL

per_condition = FD002.split_on_condition(test_df)
for condition in per_condition['condition'].unique():
    df = per_condition[per_condition['condition'] == condition].drop('condition', axis=1)
    file_name = "FD002_condition_{}.txt".format(condition)
    df.to_csv(os.path.join(test_path, file_name), index=False, encoding='utf-8')
    print(condition)
