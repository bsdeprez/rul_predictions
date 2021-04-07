from Data.datareader import DataObject

FD001 = DataObject('FD001')
FD001.add_kinking_function(130)
print(FD001.train_df.head())
