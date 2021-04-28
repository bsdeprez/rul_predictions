from RemainingUsefulLifeProject.Data.dataobject import CustomDataObject

dao = CustomDataObject('FD001')

dao.add_kinking_function(5)
dao.drop_columns(['condition'])
print(dao.train_dfs['5'])

