import pandas as pd 

def loadDataset(datasetName, run, dir_path):
    data_path = dir_path + datasetName + "_" + str(run) + "_"
    data_train = pd.read_csv(data_path+"train.csv", header=0)
    data_train = data_train.to_numpy()
    data_valid = pd.read_csv(data_path+"valid.csv", header=0)
    data_valid = data_valid.to_numpy()
    data_test = pd.read_csv(data_path+"test.csv", header=0)
    data_test = data_test.to_numpy()
    return data_train, data_valid, data_test