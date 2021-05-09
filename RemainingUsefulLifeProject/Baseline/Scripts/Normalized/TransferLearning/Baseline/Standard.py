from RemainingUsefulLifeProject.Baseline.Models.FeedForwardNeuralNetworks.Baseline import FFNModel
from RemainingUsefulLifeProject.Data.cmapss_dao import DataObject

filepath = "../../../../../../Data/CMAPSSData/"
FD001 = DataObject("FD001", filepath=filepath)
FD002 = DataObject("FD002", filepath)

model = FFNModel(len(FD001.sensor_names), lr=0.15)

# First: train the model on condition 1, 2 and 3 of dataset FD002
