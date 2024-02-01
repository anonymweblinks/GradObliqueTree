import torch 
import sys 
import copy 
import time

sys.path.append('./src/')

from warmStart import CART_Reg_warmStart
from GWTFunc import  multiStartTreeOptbyGRAD_withC, treeOptbyGRADwithC
from treeFunc import update_c, objv_cost, get_branch_nodes


def NDTGradOpt(X_train, Y_train, X_test, Y_test, treeDepth, indices_flags_dict, epochNum, device, startNum, cart_warmStart_dict, NDTWarmStart, RsWs):
    # Regression Tree
    objv_TreeMH, Tree_MH, objv_TreeOT, Tree_OT, elapsedTime_OT = RT(X_train, Y_train, treeDepth, indices_flags_dict, epochNum, device, startNum, cart_warmStart_dict, NDTWarmStart, RsWs)
    objv_MSE_train_OT, r2_train_OT = objv_cost(X_train, Y_train, treeDepth, Tree_OT)
    objv_MSE_test_OT, r2_test_OT = objv_cost(X_test, Y_test, treeDepth, Tree_OT)
    print("\nFinal Results...")
    print("\nobjv_MSE_train_GWT: {};    objv_MSE_test_GWT: {}".format(objv_MSE_train_OT, objv_MSE_test_OT))
    print("r2_train_GWT: {};    r2_test_GWT: {}".format(r2_train_OT, r2_test_OT))
    print(f"elapsedTime_GWT is {elapsedTime_OT}")

    objv_MSE_train_MH, r2_train_MH = objv_cost(X_train, Y_train, treeDepth, Tree_MH)
    objv_MSE_test_MH, r2_test_MH = objv_cost(X_test, Y_test, treeDepth, Tree_MH)
    print("\nobjv_MSE_train_GPSub: {};    objv_MSE_test_GPSub: {}".format(objv_MSE_train_MH, objv_MSE_test_MH))
    return r2_train_MH, r2_test_MH, r2_train_OT, r2_test_OT, elapsedTime_OT, Tree_MH, Tree_OT

def RT(X, y, treeDepth, indices_flags_dict, epochNum, device, startNum, cart_warmStart_dict, NDTWarmStart, RsWs):
    n, p = X.shape
    Tb = 2 ** treeDepth - 1                            # branch node size                     
    Tleaf = 2 ** treeDepth                             # leaf node size        

    a = torch.zeros((Tb, p), device= device, dtype=torch.float32)
    b = torch.zeros((Tb), device= device, dtype=torch.float32)

    print("\nOptimizing for One-time or One-tree...")
    startTime = time.perf_counter() 

    objv_TreeOT, tree_Dmax_OT = multiStartTreeOptbyGRAD_withC(X, y, treeDepth, indices_flags_dict, epochNum, device, cart_warmStart_dict, NDTWarmStart, RsWs, startNum)
    elapsedTime_OT = time.perf_counter() - startTime

    TreeOT = copy.deepcopy(tree_Dmax_OT)
    #
    ## 
    print("Optimizing for moving horizon...")
    a, b = RT_inner(X, y, 1, treeDepth,indices_flags_dict, a, b, 1, epochNum, device,  TreeOT, startNum)
    treeDepth_tensor = torch.tensor(treeDepth, device=device)
    TreeMH = {"a": a, "b": b, "c": torch.zeros(Tleaf, device=device, dtype=torch.float32), "D": treeDepth_tensor}
    c_TreeMH = update_c(X, y, treeDepth_tensor, TreeMH)
    TreeMH = {"a": a, "b": b, "c": c_TreeMH, "D": treeDepth_tensor}
    objv_TreeMH, _ = objv_cost(X, y, treeDepth, TreeMH)

    if objv_TreeOT <= objv_TreeMH:
        print("objv_TreeOT <= objv_TreeMH")

    return objv_TreeMH, TreeMH, objv_TreeOT, tree_Dmax_OT, elapsedTime_OT




def RT_inner(X, y, dc, Dmax, indices_flags_dict, a_s, b_s, ind, epochNum, device, treeOT_allNodesWS, startNum):

    H_ind = 2 if Dmax-dc >= 1 else 1
    
    H_ind_tensor = torch.tensor(H_ind, device=device)

    BranchNodesList = get_branch_nodes(ind, H_ind)
    a_branchNodesWS = treeOT_allNodesWS["a"][BranchNodesList, :]
    b_branchNodesWS = treeOT_allNodesWS["b"][BranchNodesList]
    Tleaf = 2 ** H_ind
    tree_branchNodeWS =  {"a": a_branchNodesWS, "b": b_branchNodesWS, "c": torch.zeros(Tleaf, device=device, dtype=torch.float32), "D": H_ind_tensor}
    c_branchNodesWS = update_c(X, y, H_ind, tree_branchNodeWS)
    tree_branchNodeWS =  {"a": a_branchNodesWS, "b": b_branchNodesWS, "c": c_branchNodesWS, "D": H_ind_tensor}


    # CART Regression Tree Warm Start
    a_init, b_init, c_init = CART_Reg_warmStart(X, y, H_ind, device)
    cart_warmStart_dict = {"a": a_init, "b": b_init, "c": c_init}
    cart_warmStart_Hind = copy.deepcopy(cart_warmStart_dict) 
    # reduced-sample warm start
    reduced_number = X.shape[0] // 10 if X.shape[0] > 1000 else X.shape[0] // 2
    objv_cartWarmStart_reduced, tree_reduced = treeOptbyGRADwithC(H_ind, indices_flags_dict, 10000, X[:reduced_number,:], y[:reduced_number], device, cart_warmStart_Hind, None, None,  20.0, 0)
    c_RsWs = update_c(X[:reduced_number,:], y[:reduced_number], H_ind, tree_reduced)
    RsWs = {"a": tree_reduced["a"], "b": tree_reduced["b"], "c": c_RsWs}

    objv_iter, tree = multiStartTreeOptbyGRAD_withC(X, y, H_ind, indices_flags_dict, epochNum, device, cart_warmStart_Hind, tree_branchNodeWS, RsWs, startNum)
    a_s[ind-1, :] = tree["a"][0, :]
    b_s[ind-1] = tree["b"][0]


    # reach the depth 
    if dc == Dmax:
        return a_s, b_s
    
    # node index for next layer 
    node_l = 2 * ind
    node_r = 2 * ind + 1
    yes = torch.matmul(X, a_s[ind-1, :]) < b_s[ind-1]
    if y[yes].shape[0] > 1 and torch.unique(y[yes]).numel() > 1:
        a_s, b_s = RT_inner(X[yes, :], y[yes], dc + 1, Dmax, indices_flags_dict, a_s, b_s, node_l, epochNum, device, treeOT_allNodesWS, startNum)
    else:
        pass
    if y[~yes].shape[0] > 1 and torch.unique(y[~yes]).numel() > 1:
        a_s, b_s = RT_inner(X[~yes, :], y[~yes], dc + 1, Dmax, indices_flags_dict, a_s, b_s, node_r, epochNum, device, treeOT_allNodesWS, startNum)
    else:
        pass

    return a_s, b_s


