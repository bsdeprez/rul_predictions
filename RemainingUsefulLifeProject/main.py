from RemainingUsefulLifeProject.Data.cmapss_dao import DataObject

filepath = "../Data/CMAPSSData/"

FD001 = DataObject("FD001", filepath=filepath)
FD001.add_kinking_point(130)
FD001.drop_columns(['s_21'])
train, test = FD001.datasets[1]
print(train)
