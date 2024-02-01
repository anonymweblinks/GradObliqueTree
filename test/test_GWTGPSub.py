import torch 
import numpy as np
import pandas as pd
import time
import json 

import sys
sys.path.append('./src/')

from dataset import loadDataset
from treeFunc import readTreePath, update_c
from warmStart import CART_Reg_warmStart
from GWTFunc import treeOptbyGRADwithC
from GPSubFunc import NDTGradOpt





if __name__ == "__main__":
    ################## main Code ##################
    
    ## Args 
    data_num_start = int(sys.argv[1])
    data_num_end = int(sys.argv[2])
    runs_num_start = int(sys.argv[3])
    runs_num_end = int(sys.argv[4])
    
    # tree depth 
    treeDepth = int(sys.argv[5])                    #  2 4 8 
    epochNum = int(sys.argv[6])                     #  1000 
    device_arg =  str(sys.argv[7])                  #  "cuda:0" or "cpu"
    device = torch.device(device_arg)
    startNum = int(sys.argv[8]) 

    
    ##  data
    datasetPath = "./data/"
    
    
    Datasets_names = [ "auto-mpg", "automobile", "communities-and-crime", "computer-hardware", "concrete-slump-test-compressive", "concrete-slump-test-flow",
    "concrete-slump-test-slump", "housing", "hybrid-price", "lpga-2008", "lpga-2009",  "yacht-hydrodynamics",
    "abalone", "ailerons", "airfoil-self-noise", "cpu-act", "cpu-small", "elevators", "kin8nm", "parkinsons-telemonitoring-motor", 
    "parkinsons-telemonitoring-total", "vote-for-clinton", "friedman-artificial", "sgemm_product" ]


    ## read the treePath from the HDF5 file
    indices_flags_dict = readTreePath(treeDepth, device)

    datasetNum = len(Datasets_names)
    print("Starting: Total {} datasets".format(datasetNum))
    
    
    for datasetIdx in range(data_num_start-1, data_num_end):
        print("############# Dataset[{}]: {} #############".format(datasetIdx, Datasets_names[datasetIdx]))
        for run in range(runs_num_start, runs_num_end+1):
            print("####### Run: {} #######".format(run))
            torch.manual_seed(run)
            np.random.seed(run)
    
            data_train, data_valid, data_test = loadDataset(Datasets_names[datasetIdx], run, datasetPath)
            
            p = data_train.shape[1] - 1
            X_train = torch.from_numpy(data_train[:, 0:p] * 1.0).float()
            Y_train = torch.from_numpy(data_train[:, p] * 1.0).float()
            X_valid = torch.from_numpy(data_valid[:, 0:p] * 1.0).float()
            Y_valid = torch.from_numpy(data_valid[:, p] * 1.0).float()
            X_test = torch.from_numpy(data_test[:, 0:p] * 1.0).float()
            Y_test = torch.from_numpy(data_test[:, p] * 1.0).float()

            # X_train = X_train.to(device, non_blocking=True)
            # Y_train = Y_train.to(device, non_blocking=True)
            # X_valid = X_valid.to(device, non_blocking=True)
            # Y_valid = Y_valid.to(device, non_blocking=True)
            # X_test = X_test.to(device, non_blocking=True)
            # Y_test = Y_test.to(device, non_blocking=True)

            X = torch.cat((X_train, X_valid), 0)
            Y = torch.cat((Y_train, Y_valid), 0)
            X = X.to(device, non_blocking=True)
            Y = Y.to(device, non_blocking=True)
            X_test = X_test.to(device, non_blocking=True)
            Y_test = Y_test.to(device, non_blocking=True)



            if run == runs_num_start:
                print("dataset:{};    n_train:{};    n_valid:{};    n_test:{};    p:{}\n".format(Datasets_names[datasetIdx], X_train.shape[0], X_valid.shape[0], X_test.shape[0], X_train.shape[1]))



            # Use a subset of the data to warm start the optimization
            a_init, b_init, c_init = CART_Reg_warmStart(X, Y, treeDepth, device)
            cart_warmStart_dict = {"a": a_init, "b": b_init, "c": c_init}

            startTime = time.perf_counter()
            r2_train_GPSub, r2_test_GPSub, r2_train_GWT, r2_test_GWT, elapsedTime_GWT, Tree_GPSub, Tree_GWT = NDTGradOpt(X, Y, X_test, Y_test, treeDepth, indices_flags_dict, epochNum, device, startNum, cart_warmStart_dict, None, None)
            print("r2_train_GPSub: {};   r2_test_GPSub: {}".format(r2_train_GPSub, r2_test_GPSub))
            elapsedTime_GPSub = time.perf_counter()-startTime
            print("\nelapsedTime: {}\n".format(elapsedTime_GPSub))


            print("treeGWT: ", Tree_GWT)
            print("treeGMSub: ", Tree_GPSub)

