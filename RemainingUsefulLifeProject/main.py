from RemainingUsefulLifeProject.Data.datareader import DataObject
import os

filepath = "../Data/CMAPSSData/"
custom_datasets = "../Data/Custom/Per_Condition"
FD002 = DataObject('FD002', filepath=filepath)
x_train = FD002.train_df[FD002.sensor_names].values
y_train = FD002.train_df['RUL'].values
train = FD002.split_on_condition(FD002.test_df)

for i in range(1, 7):
    df_per_condition = train[train['condition'] == i]
    filename = "FD002_test_condition_{}.csv".format(i)
    if len(df_per_condition.index) > 0:
        path = os.path.join(custom_datasets, filename)
        df_per_condition.to_csv(path, encoding='utf-8', index=False)
