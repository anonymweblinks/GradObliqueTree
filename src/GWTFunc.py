import torch 
import copy 
import sys 
sys.path.append('./src/')

from treeFunc import objv_cost, update_c

class scaledSigmoid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, scale):
        ctx.save_for_backward(input)
        ctx.scale = scale
        return 1/(1+torch.exp(-scale*input))

    @staticmethod
    def backward(ctx, grad_output):
        scale = ctx.scale
        input, = ctx.saved_tensors
        gradient = grad_output * (scale * 1/(1+torch.exp(-scale*input)) * (1 - 1/(1+torch.exp(-scale*input))))
        grad_scale = None 
        return gradient, grad_scale
    
class branchNodeNet(torch.nn.Module):
    def __init__(self, treeDepth: int, p: int, scale: float) -> None:
        super().__init__()
        self.depth = treeDepth
        self.featNum = p
        self.treesize = 2 ** (self.depth + 1) - 1
        self.branchNodeNum = 2 ** (self.depth) - 1
        self.scale = scale

        self.linear1 = torch.nn.Linear(self.featNum, self.branchNodeNum)
        self.scaledSigmoid = scaledSigmoid.apply
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = self.scaledSigmoid(-x, self.scale)
        return x



@torch.jit.script
def objectiveFuncwithC(branchOutput: torch.Tensor, Y_train: torch.Tensor, c_leafLable: torch.Tensor,  treeDepth: int, indices_tensor: torch.Tensor, flags_tensor: torch.Tensor) -> torch.Tensor: 

    if treeDepth == 1:
        Indicator1 = branchOutput 
        Y_train_diff = (Y_train.view(-1, 1) - c_leafLable.view(1, -1)).pow_(2)  # n x 2 elements
        objv = torch.sum(Indicator1 * Y_train_diff[:, :1] + (1 - Indicator1) * Y_train_diff[:, 1:])


    else:
        indicators_complement = 1 - branchOutput
        Y_train_diff = (Y_train.view(-1, 1) - c_leafLable.view(1, -1)).pow_(2)
        indicators_stack = torch.stack((indicators_complement, branchOutput))
        selected_indicators = indicators_stack[flags_tensor, :, indices_tensor]
        indicator_pairs = selected_indicators.prod(dim=1).transpose(0, 1)
        objv = (indicator_pairs * Y_train_diff).sum()

    return objv




def treeOptbyGRADwithC(treeDepth, indices_flags_dict, epochNum, X_train, Y_train, device, CARTwarmStart, NDTWarmStart, RsWs_MH, scaleFactor, idx_start):

    ## hyperparameters
    learningRate = 0.01
    stop = 25

    ##  net
    p = X_train.shape[1]
    scale = torch.tensor([scaleFactor], device=device)
    net = branchNodeNet(treeDepth, p, scale).to(device, non_blocking=True)
    # Script the custom network class

    ### initialize weight and bias of net
    if idx_start == 0:
        ## CART Regression Tree Warm Start
        a = CARTwarmStart["a"]
        b = CARTwarmStart["b"]
        c = CARTwarmStart["c"]
    elif NDTWarmStart is not None and idx_start == 1:
        ## take tree_branchNodeWS as the initial warm start for the current node ind
        a = NDTWarmStart["a"]
        b = NDTWarmStart["b"]
        c = NDTWarmStart["c"]
    elif RsWs_MH is not None and idx_start == 2:
        ## take reduced-sample warm start as the initial warm start for the current node ind in Moving-horizon strategy
        a = RsWs_MH["a"]
        b = RsWs_MH["b"]
        c = RsWs_MH["c"]
    else:
        ## Random Initialization
        a = torch.rand(net.linear1.weight.shape, device=device)*(2.0)+(-1.0)
        b = torch.rand(net.linear1.bias.shape, device=device)*(2.0)+(-1.0)
        c = torch.rand([2 ** treeDepth], device=device)*(torch.max(Y_train)-torch.min(Y_train))+torch.min(Y_train)      

    customWeigt = torch.as_tensor(a, device=device)
    customBias = torch.as_tensor(b, device=device)
    net.linear1.weight = torch.nn.Parameter(customWeigt)
    net.linear1.bias = torch.nn.Parameter(customBias)

    ## optimized parameters
    c_leafLable = torch.as_tensor(c, device=device).requires_grad_()

    ## Optimizer 
    optimizer = torch.optim.Adam(list(net.parameters())+[c_leafLable], lr=learningRate)


    objvList = []
    consec_decreases = 0

    # load the indices_tensor and flags_tensor from the indices_flags_dict

    indices_tensor = indices_flags_dict["D"+str(treeDepth)]["indices_tensor"]
    flags_tensor = indices_flags_dict["D"+str(treeDepth)]["flags_tensor"]


    for epoch in range(epochNum):

        optimizer.zero_grad(set_to_none=True)

        # net 
        branchOutput = net(X_train)
        objv = objectiveFuncwithC(branchOutput, Y_train, c_leafLable, treeDepth, indices_tensor, flags_tensor)

        objvList.append(objv.item())
        objv.backward()

        optimizer.step()

        currLR = optimizer.param_groups[-1]['lr']

        ## early stopping for learning rate change 
        if epoch > int(0.2*epochNum) and objvList[-2]-objvList[-1] < 1e-8:
            consec_decreases += 1
        else:
            consec_decreases = 0

        if consec_decreases == 5:
            if currLR > 1e-4:
                consec_decreases = 0
                optimizer.param_groups[0]['lr'] = currLR*0.9
            else:
                optimizer.param_groups[0]['lr'] = learningRate

        elif consec_decreases == stop and epoch > int(0.9*epochNum):
            print(f"Stopped early at epoch {epoch + 1} - objv unchanged for {consec_decreases} consecutive epochs!")
            break 
        
    with torch.no_grad():    
        a_grad = net.linear1.weight * 1.0        # requires_grad is not set to True after times 1.0
        b_grad = net.linear1.bias * (-1.0)
        c_grad = c_leafLable * 1.0               # requires_grad is not set to True after times 1.0
        treeDict = {"a": a_grad, "b": b_grad, "c": c_grad}
        
    return objv, treeDict





def multiStartTreeOptbyGRAD_withC(X_train, Y_train, treeDepth, indices_flags_dict, epochNum, device, CARTwarmStart, NDTWarmStart, RsWs_MH, startNum):

    objvmin = 1e10
    treeOpt = None

    for idx_start in range(startNum):

        scaleFactor = 5 + 20 * torch.rand(1, device=device)
        print("scaleFactor is {}".format(scaleFactor))

        objvCurrent, treeCurrent = treeOptbyGRADwithC(treeDepth, indices_flags_dict, epochNum, X_train, Y_train, device, CARTwarmStart, NDTWarmStart, RsWs_MH, scaleFactor, idx_start)
        
        c = update_c(X_train, Y_train, treeDepth, treeCurrent)
        treeUpcateC = {"a": treeCurrent["a"], "b": treeCurrent["b"], "c": c}

        objvCurrent, r2Current = objv_cost(X_train, Y_train, treeDepth, treeUpcateC) 

        if objvCurrent < objvmin:
            objvmin = objvCurrent
            treeOpt = copy.deepcopy(treeUpcateC)
        print("objvCurrent: {};   in idx_start {}".format(objvCurrent, idx_start))
        print("objvmin: {};   in idx_start {}\n".format(objvmin, idx_start))

    return objvmin, treeOpt
















