
from sklearn.tree import _tree
from sklearn import tree
import numpy as np
import torch 

## retrieve the parameters abc of the trained tree model
def regTreeWarmStart(model, treeDepth):
    tree_ = model.tree_                  
    branchNode_inputDepth = 2**(treeDepth) - 1
    Fitted_treeDepth = model.get_depth()
    branchNode_FittedDepth = 2**(Fitted_treeDepth) - 1

    leafNode_inputDepth = 2**(treeDepth)
    leafNode_FittedDepth = 2**(Fitted_treeDepth)
    if Fitted_treeDepth != treeDepth:
        print("Fitted_treeDepth != treeDepth")
        a = [0]*branchNode_inputDepth
        b = [0]*branchNode_inputDepth
        c = [0]*leafNode_inputDepth
    else:
        a = [0]*branchNode_FittedDepth
        b = [0]*branchNode_FittedDepth
        c = [0]*leafNode_FittedDepth

    def warmStartPara(node, ind):

        if tree_.n_node_samples[node] == 1 and ind <= branchNode_FittedDepth:
            node_l = 2 * ind
            node_r = 2 * ind + 1
            a[ind-1] = 0
            b[ind-1] = 0
            c[node_l-1-branchNode_FittedDepth] = 0
            c[node_r-1-branchNode_FittedDepth] = tree_.value[node].squeeze()

        else:
            if tree_.feature[node] != _tree.TREE_UNDEFINED:
                featureIdx = tree_.feature[node]
                threshold = tree_.threshold[node]
                a[ind-1] = featureIdx
                b[ind-1] = -threshold

                node_l = 2 * ind
                node_r = 2 * ind + 1
                warmStartPara(tree_.children_left[node], node_l)
                warmStartPara(tree_.children_right[node], node_r)
            else:
                if Fitted_treeDepth == treeDepth:
                    c[ind-1-branchNode_FittedDepth] = (tree_.value[node].squeeze())
                else:
                    depthDiff = treeDepth - Fitted_treeDepth
                    peudo_ind_r = ind 
                    for _ in range(depthDiff):
                        peudo_ind_r = 2 * peudo_ind_r + 1
                    c[peudo_ind_r-1-branchNode_inputDepth] = (tree_.value[node].squeeze())

    warmStartPara(0, 1)
    return a, b, c

def CART_Reg_warmStart(X, Y, treeDepth, device):
    model = tree.DecisionTreeRegressor(max_depth=treeDepth, min_samples_leaf=1, random_state=0)
    if device == torch.device('cuda'):
        X_np, Y_np = X.cpu().numpy(), Y.cpu().numpy()
    else:
        X_np, Y_np = X, Y
    model = model.fit(X_np, Y_np)
    p = X.shape[1]
    a, b, c = regTreeWarmStart(model,treeDepth)
    a_all = np.eye(p, dtype="float32")[a]
    return a_all, np.asarray(b,dtype="float32"), np.asarray(c, dtype="float32")



