# %% [code]


"""### Import"""

import numpy as np
import time 
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import linalg as LA
import pdb
import torch
from torch.optim import Optimizer
from torch.nn.functional import sigmoid
#import pdb
#from sgdModif import SGD




"""### Utils ADL"""

class meanStd(object):
    def __init__(self):
        self.mean     = 0.0
        self.mean_old = 0.0
        self.std      = 0.001
        self.count    = 0.0
        self.minMean  = 100.0
        self.minStd   = 100.0
        self.M_old    = 0.0
        self.M        = 0.0
        self.S        = 0.0
        self.S_old    = 0.0
        
    def calcMeanStd(self, data, cnt = 1):
        self.data     = data
        self.mean_old = copy.deepcopy(self.mean)
        self.M_old    = self.count*self.mean_old
        self.M        = self.M_old + data
        self.S_old    = copy.deepcopy(self.S)
        if self.count > 0:
            self.S    = self.S_old + ((self.count*data - self.M_old)**2)/(self.count*(self.count + cnt))
        
        self.count   += cnt
        self.mean     = self.mean_old + np.divide((data-self.mean_old),self.count)
        self.std      = np.sqrt(self.S/self.count)
        
        if (self.std != self.std).any():
            print('There is NaN in meanStd')
            pdb.set_trace()
    
    def resetMinMeanStd(self):
        self.minMean = copy.deepcopy(self.mean)
        self.minStd  = copy.deepcopy(self.std)
        
    def calcMeanStdMin(self):
        if self.mean < self.minMean:
            self.minMean = copy.deepcopy(self.mean)
        if self.std < self.minStd:
            self.minStd = copy.deepcopy(self.std)

def probitFunc(meanIn,stdIn):
    stdIn += 0.0001  # for safety
    out = meanIn/(torch.ones(1) + (np.pi/8)*stdIn**2)**0.5
    
    return out

def generateWeightXavInit(nInput,nNode,nOut,nNewNode):
    copyNet         = smallAdl(nInput,nNode,nOut)
    newWeight       = copyNet.linear.weight.data[0:nNewNode]
    newWeightNext   = copyNet.linear.weight.data[:,0:nNewNode]
    newOutputWeight = copyNet.linearOutput.weight.data[:,0:nNewNode]


    return newWeight, newOutputWeight, newWeightNext

def deleteRowTensor(x,index):
    x = x[torch.arange(x.size(0))!=index] 
    
    return x

def deleteColTensor(x,index):
    x = x.transpose(1,0)
    x = x[torch.arange(x.size(0))!=index]
    x = x.transpose(1,0)
    
    return x

"""### Network"""

class smallAdl(nn.Module):
    def __init__(self, no_input, no_hidden, classes):
        super(smallAdl, self).__init__()
        # hidden layer
        self.linear = nn.Linear(no_input, no_hidden,  bias=True)
        self.activation = nn.Sigmoid()
        nn.init.xavier_uniform_(self.linear.weight)
        self.linear.bias.data.zero_()
        
        # softmax layer
        self.linearOutput = nn.Linear(no_hidden, classes,  bias=True)
        nn.init.xavier_uniform_(self.linearOutput.weight)
        self.linearOutput.bias.data.zero_()
        
    def forward(self, x):
        x  = self.linear(x)
        h  = self.activation(x)
        x  = self.linearOutput(h)
        
        h2 = (h.clone().detach())**2
        x2 = self.linearOutput(h2)
        
    
        return x, h.clone().detach(), x2.clone().detach()

def createSmallAdl(no_input,no_hidden,classes):
    obj = smallAdl(no_input,no_hidden,classes)
    
    return obj

def adlFeedforwardTest(netList,x,votingWeight,device):
    # feedforward to all layers
    with torch.no_grad():
        classes = netList[0].linearOutput.weight.shape[0]
        
        nData   = x.shape[0]
        y       = torch.zeros(nData,classes)
        yList   = []
        hList   = []

        minibatch_data = x.to(device)
        minibatch_data = minibatch_data.type(torch.float)
        tempVar = minibatch_data

        for netLen in range(len(netList)):
            currnet          = netList[netLen]
            obj              = currnet.eval()
            obj              = obj.to(device)
            tempY, tempVar,_ = obj(tempVar)
            hList            = hList + [tempVar.tolist()]
            y                = y + tempY*votingWeight[netLen]
            if votingWeight[netLen] == 0:
                yList        = yList + [[]]
            else:
                yList        = yList + [F.softmax(tempY,dim=1).tolist()]
                
       
           

    return y, yList, hList

def adlFeedforwardBiasVar(netList,netWinIdx,x,y,device):
    # feedforward from the input to the winning layer
    # y in one hot vector form, float, already put in device
    with torch.no_grad():
        minibatch_data  = x.to(device)
        minibatch_data  = minibatch_data.type(torch.float)
        minibatch_label = y
        
        tempVar = minibatch_data
        for netLen in range(len(netList)):
            currnet               = netList[netLen]
            obj                   = currnet.eval()
            obj                   = obj.to(device)
            tempY, tempVar,tempY2 = obj(tempVar)
            
            if netLen == 0:
                tempVar2          = (tempVar.clone().detach())**2
            else:
                tempY2,tempVar2,_ = obj(tempVar2)
                
            if netLen == netWinIdx:
                break
        
        tempY    = F.softmax(tempY,dim=1)
        tempY2   = F.softmax(tempY2,dim=1)
        bias     = torch.norm((tempY - minibatch_label)**2)
        variance = torch.norm(tempY2 - tempY**2)
   
    return bias.tolist(), variance.tolist(), tempVar

def adlFeedforwardTrain(netWin,xWin,device):
    # feed forward only on winning layer
    minibatch_data = xWin.to(device)
    minibatch_data = minibatch_data.type(torch.float)
    minibatch_data.requires_grad_()
    
    netWin = netWin.train()
    netWin = netWin.to(device)
    y,_,_  = netWin(minibatch_data)
    
    return y

"""### Train and Test"""

def adlTrain(netList,netWinIdx,xWin,x,y,nClass,miuX,miuBias,miuVar,lr,criterion,device,epoch=1):
    
    print('Adjust ',netWinIdx+1,'-th hidden layer')
    
    # flags
    growNode  = False
    pruneNode = False
    
    # shuffle the data
    nData = x.shape[0]
    shuffled_indices = torch.randperm(nData)
    
    # label for bias var calculation
    y_biasVar = F.one_hot(y, num_classes =nClass).float()
    
    for iData in range(0,nData):
        # load data
        
        indices                 = shuffled_indices[iData:iData+1]
               
        minibatch_xWin          = xWin[indices]
        minibatch_xWin          = minibatch_xWin.to(device)
               
        minibatch_label         = y[indices]
        minibatch_label         = minibatch_label.to(device)
        minibatch_label_biasVar = y_biasVar[indices]
        minibatch_label_biasVar = minibatch_label_biasVar.to(device)
        
        minibatch_x             = x[indices]
        minibatch_x             = minibatch_x.to(device)
        
        # calculate mean of input
        miuX.calcMeanStd(minibatch_x)
        
        # get bias and variance
        outProbit = probitFunc(miuX.mean,miuX.std)
        bias, variance, nodeSignificance = adlFeedforwardBiasVar(netList,netWinIdx,
                                                                 outProbit,minibatch_label_biasVar,device)
        
        # calculate mean of bias
        miuBias[netWinIdx].calcMeanStd(bias)
        if miuBias[netWinIdx].count < 1 or growNode:
            miuBias[netWinIdx].resetMinMeanStd()
        else:
            miuBias[netWinIdx].calcMeanStdMin()
        
        # calculate mean of variance
        miuVar[netWinIdx].calcMeanStd(variance)
        if miuVar[netWinIdx].count < 20 or pruneNode:
            miuVar[netWinIdx].resetMinMeanStd()
        else:
            miuVar[netWinIdx].calcMeanStdMin()
        
        # growing
        growNode = growNodeIdentification(bias,miuBias[netWinIdx].minMean,miuBias[netWinIdx].minStd,
                                          miuBias[netWinIdx].mean,miuBias[netWinIdx].std)
        if growNode and miuBias[netWinIdx].count >= 1:
            # grow a node
            netList = nodeGrowing(netList,netWinIdx,1)
        
        # pruning
        pruneNode = pruneNodeIdentification(variance,miuVar[netWinIdx].minMean,miuVar[netWinIdx].minStd,
                                            miuVar[netWinIdx].mean,miuVar[netWinIdx].std)
        if (pruneNode and not growNode and miuVar[netWinIdx].count >= 20 and 
           netList[netWinIdx].linear.weight.data.shape[0] > netList[netWinIdx].linearOutput.weight.data.shape[0]):
            pruneIdx = findLeastSignificantNode(nodeSignificance)
            
            # prune a node
            netList  = nodePruning(netList,netWinIdx,pruneIdx)
            
        # active learning
        # if not growNode and not pruneNode and activeLearn:
            # active learning can be executed if there is no growing and pruning and active learning is triggered
        
        # declare parameters to be trained
        netOptim  = []
        netOptim  = netOptim + list(netList[netWinIdx].parameters())
        optimizer =torch.optim.SGD(netOptim, lr = lr, momentum = 0.95) #, weight_decay = 5e-4)
        # optimizer = torch.optim.Adam(netOptim, lr = 0.05, weight_decay = 5e-4)
        
        # feedforward
        scores    = adlFeedforwardTrain(netList[netWinIdx],minibatch_xWin,device)
        
        # calculate loss
        minibatch_label = minibatch_label.long()
        loss            = criterion(scores,minibatch_label)
        
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print('Bias: ',miuBias[netWinIdx].mean)
    print('Variance: ',miuVar[netWinIdx].mean)
    
    return netList, miuX, miuBias, miuVar

def adlTest(netList,votingWeight,test_data,test_label,batch_size,criterion,device):
    # load data
    test_data  = test_data.to(device)
    test_label = test_label.to(device)
    test_label = test_label.long()
    
    # testing
    start_test              = time.time()
    scores,scoresList,_     = adlFeedforwardTest(netList,test_data,votingWeight,device)
    rawPredicted, predicted = torch.max(F.softmax(scores.data,dim=1), 1)
    #print(predicted)
    end_test                = time.time()

    loss          = criterion(scores,test_label)
    residualError = torch.tensor([1.0]) - rawPredicted
    correct       = (predicted == test_label).sum().item()
    accuracy      = 100*correct/(predicted == test_label).shape[0]  # 1: correct, 0: wrong
    testing_time  = end_test - start_test
    F_matrix      = (predicted != test_label).int().tolist()  # 1: wrong, 0: correct
    
    lossList      = []
    F_matrixList  = []
    for netLen in range(len(netList)):
        if votingWeight[netLen] == 0:
            F_matrixList = F_matrixList + [[]]
            lossList     = lossList + [-1]  # -1 loss value indicate that the layer is already pruned
        else:
            _, predicted = torch.max(torch.FloatTensor(scoresList[netLen]).data,1)
            F_matrixList = F_matrixList + [(predicted != test_label).int().tolist()]  # 1: wrong, 0: correct
            loacalLoss   = criterion(torch.FloatTensor(scoresList[netLen]),test_label)
            lossList     = lossList + [loacalLoss.tolist()]
        
    print('Testing Accuracy: {}'.format(accuracy))
    print('Testing Loss: {}'.format(loss))
    print('Testing Time: {}'.format(testing_time))
    
    return scores, scoresList, loss, lossList, residualError, accuracy, testing_time, F_matrix, F_matrixList

def updateVotingWeight(votingWeight,dynamicDecreasingFactor,decreasingFactor,FmatrixList):
    for idx in range(0,len(votingWeight)):
        currFmat = FmatrixList[idx]
        for iData in range(0,len(currFmat)):
            if currFmat[iData] == 1:  # detect wrong prediction
                # penalty
                dynamicDecreasingFactor[idx] = np.maximum(dynamicDecreasingFactor[idx] - 
                                                          decreasingFactor, decreasingFactor)
                votingWeight[idx]            = np.maximum(votingWeight[idx]*dynamicDecreasingFactor[idx], 
                                                          decreasingFactor)
            elif currFmat[iData] == 0:  # detect correct prediction
                # reward
                dynamicDecreasingFactor[idx] = np.minimum(dynamicDecreasingFactor[idx] + decreasingFactor, 1)
                votingWeight[idx]            = np.minimum(votingWeight[idx]*(1 + dynamicDecreasingFactor[idx]), 1)
    
    votingWeight = (votingWeight/np.sum(votingWeight)).tolist()
    
    return votingWeight, dynamicDecreasingFactor

"""### Network Evaluation"""

def driftDetection(fMat,alphaDrift,alphaWarning,driftStatusOld):
    driftStatus = 0  # 0: no drift, 1: warning, 2: drift
    
    nData = len(fMat)
    F_max = np.max(fMat)
    
    if F_max != 0:  # all predictions are correct, no need to check drift
        F_min = np.min(fMat)
        miu_F = np.mean(fMat)
        errorBoundF = np.sqrt((1/(2*nData))*np.log(1/alphaDrift))

        cutPointCandidate = [int(nData/4),int(nData/2),int(nData*3/4)]
        cutPoint = 0

        # confirm the cut point
        for iCut in cutPointCandidate:
            miu_G = np.mean(fMat[0:iCut])
            nG    = len(fMat[0:iCut])
            errorBoundG = np.sqrt((1/(2*nG))*np.log(1/alphaDrift))
            if (miu_F + errorBoundF) <= (miu_G + errorBoundG):
                cutPoint = iCut
                print('A cut point is detected cut: ', cutPoint)
                break

        # confirm drift
        if cutPoint != 0:
            errorBoundDrift = (F_max - F_min)*np.sqrt(((nData - cutPoint)/(2*cutPoint*nData))*
                                                      np.log(1/alphaDrift))
            if (miu_G - miu_F) >= errorBoundDrift:
                print('H0 is rejected with size: ', errorBoundDrift)
                print('Status: DRIFT')
                driftStatus = 2
            else:
                errorBoundWarning = (F_max - F_min)*np.sqrt(((nData - cutPoint)/(2*cutPoint*nData))*
                                                            np.log(1/alphaWarning))
                if (miu_G - miu_F) >= errorBoundWarning and driftStatusOld != 1:
                    print('H0 is rejected with size: ', errorBoundWarning)
                    print('Status: WARNING')
                    driftStatus = 1
                else:
                    print('H0 is NOT rejected')
                    print('Status: STABLE')
                    driftStatus = 0
        else:
            print('Status: STABLE')
    else:
        print('Status: STABLE')
    
    return driftStatus

def winLayerIdentifier(votWeight):
    idx = 0
    #idx = np.argmax(np.asarray(votWeight)/(np.asarray(allLoss) + 0.001))
    idx = np.argmax(np.asarray(votWeight))
    
    return idx

def growNodeIdentification(bias,minMeanBias,minStdBias,meanBias,stdBias):
    growNode = False
    
    dynamicKsigmaGrow = 1.3*np.exp(-bias) + 0.7
    growCondition1    = minMeanBias + dynamicKsigmaGrow*minStdBias
    growCondition2    = meanBias + stdBias
    
    if growCondition2 > growCondition1:
        growNode = True
    
    return growNode

def pruneNodeIdentification(var,minMeanVar,minStdVar,meanVar,stdVar):
    pruneNode = False
    
    dynamicKsigmaPrune = 1.3*np.exp(-var) + 0.7
    pruneCondition1    = minMeanVar + 2*dynamicKsigmaPrune*minStdVar
    pruneCondition2    = meanVar + stdVar
    
    if pruneCondition2 > pruneCondition1:
        pruneNode = True
    
    return pruneNode

def findLeastSignificantNode(nodeSig):
    leastSigIdx = torch.argmin(torch.abs(nodeSig)).tolist()
    
    return leastSigIdx

"""### Evolving"""

def layerGrowing(netList,votWeight,dyDecFactor,avgBias,avgVar):
    nInput      = netList[-1].linearOutput.in_features
    nOutput     = netList[-1].linearOutput.out_features
    netList     = netList + [createSmallAdl(nInput,nOutput,nOutput)]
    votWeight   = votWeight + [1.0]
    dyDecFactor = dyDecFactor + [1.0]
    votWeight   = (votWeight/np.sum(votWeight)).tolist()
    avgBias     = avgBias + [meanStd()]
    avgVar      = avgVar + [meanStd()]
    print('*** ADD a new LAYER ***')
    
    return netList, votWeight, dyDecFactor, avgBias, avgVar

def layerPruning(yList,votingWeight,pruneThreshold):
    prunedLayerList = []
    nLayer = np.count_nonzero(votingWeight)
    for i in range(0,len(yList)):
        if votingWeight[i] == 0:
            continue
        
        for j in range(i+1,len(yList)):
            if votingWeight[j] == 0:
                continue
            
            A = torch.FloatTensor(yList[i]).transpose(0,1)
            B = torch.FloatTensor(yList[j]).transpose(0,1)
            nOutput = A.shape[0]
            MICI = []
            for k in range(0,nOutput):
                varianceA = np.var(A[k].tolist())
                varianceB = np.var(B[k].tolist())
                corrAB = np.corrcoef(A[k].tolist(),B[k].tolist())[0][1]
                
                if (corrAB != corrAB).any():
                    print('There is NaN in LAYER pruning')
                    corrAB = 0.0
                
                mici = (varianceA + varianceB - np.sqrt((varianceA + varianceB)**2 - 
                                                        4*varianceA*varianceB*(1-corrAB**2)))/2
                    
                print('mici of ',i,'-th layer and ',j,'-th layer and ',k,'-th output is: ',mici)
                MICI.append(mici)

            if np.max(np.abs(MICI)) < pruneThreshold:
                print('layer ',i+1, 'and layer ',j+1, 'are highly correlated with MICI ', np.max(np.abs(MICI)))
                if votingWeight[i] < votingWeight[j]:
                    prunedLayerList.append(i)
                    votingWeight[i] = 0
                    print('\\\ hidden LAYER ',i+1, 'is PRUNED ///')
                else:
                    prunedLayerList.append(j)
                    votingWeight[j] = 0
                    print('\\\ hidden LAYER ',j+1, 'is PRUNED ///')
                
                nLayer -= 1
                if nLayer <= 1:
                    break
    
    votingWeight = (votingWeight/np.sum(votingWeight)).tolist()
    
    return votingWeight

def removeLastLayer(netList,votWeight,dyDecFactor,avgBias,avgVar):
    while votWeight[-1] == 0:
        del netList[-1]
        del votWeight[-1]
        del dyDecFactor[-1]
        del avgBias[-1]
        del avgVar[-1]
        print('### A LAST hidden LAYER is REMOVED ###')
    
    return netList, votWeight, dyDecFactor, avgBias, avgVar

def nodeGrowing(netList,winIdx,nNewNode):
    if winIdx <= (len(netList)-1):
        netList      = copy.deepcopy(netList)
        
        nInputWin    = netList[winIdx].linear.weight.shape[1]
        nNodeWin     = netList[winIdx].linear.weight.shape[0]
        nOutput      = netList[winIdx].linearOutput.weight.shape[0]
        nNewNodeCurr = nNodeWin + nNewNode

        # grow node for current layer, output
        newWeight, newOutputWeight,_         = generateWeightXavInit(nInputWin,nNewNodeCurr,nOutput,nNewNode)
        netList[winIdx].linear.weight.data   = torch.cat((netList[winIdx].linear.weight.data,
                                                          newWeight),0)  # grow input weights
        netList[winIdx].linear.bias.data     = torch.cat((netList[winIdx].linear.bias.data,
                                                          torch.zeros(nNewNode)),0)  # grow input bias
        netList[winIdx].linear.out_features  = nNewNodeCurr
        del netList[winIdx].linear.weight.grad
        del netList[winIdx].linear.bias.grad
        
        # grow input weight of linearOutput
        netList[winIdx].linearOutput.weight.data = torch.cat((netList[winIdx].linearOutput.weight.data,
                                                                newOutputWeight),1)
        netList[winIdx].linearOutput.in_features = nNewNodeCurr
        del netList[winIdx].linearOutput.weight.grad
        del netList[winIdx].linearOutput.bias.grad

        if winIdx != (len(netList)-1):
            nextIdx       = winIdx + 1
            nInputNext    = netList[nextIdx].linear.weight.shape[1]
            nNodeNext     = netList[nextIdx].linear.weight.shape[0]
            nOutputNext   = netList[nextIdx].linearOutput.weight.shape[0]
            nNewInputNext = nInputNext + nNewNode

            # grow input weight of next layer
            _,_,newWeightNext = generateWeightXavInit(nNewInputNext,nNodeNext,nOutputNext,nNewNode)
            netList[nextIdx].linear.weight.data = torch.cat((netList[nextIdx].linear.weight.data,newWeightNext),1)
            del netList[nextIdx].linear.weight.grad

            # update input features
            netList[nextIdx].linear.in_features = nNewInputNext
           
            
        print('+++ GROW a hidden NODE +++')
    else:
        raise IndexError
    
    return copy.deepcopy(netList)

def nodePruning(netList,winIdx,pruneIdx):
    if winIdx <= (len(netList)-1):
        netList      = copy.deepcopy(netList)
        
        nNodeWin     = netList[winIdx].linear.weight.shape[0]
        nPrunedNode  = 1
        nNewNodeCurr = nNodeWin - nPrunedNode  # prune a node

        # prune node for current layer, output
        netList[winIdx].linear.weight.data  = deleteRowTensor(netList[winIdx].linear.weight.data,
                                                           pruneIdx)  # prune input weights
        netList[winIdx].linear.bias.data    = deleteRowTensor(netList[winIdx].linear.bias.data,
                                                           pruneIdx)  # prune input bias
        netList[winIdx].linear.out_features = nNewNodeCurr
        del netList[winIdx].linear.weight.grad
        del netList[winIdx].linear.bias.grad

        # prune input weight of linearOutput
        netList[winIdx].linearOutput.weight.data = deleteColTensor(netList[winIdx].linearOutput.weight.data,pruneIdx)
        netList[winIdx].linearOutput.in_features = nNewNodeCurr
        del netList[winIdx].linearOutput.weight.grad
        del netList[winIdx].linearOutput.bias.grad

        if winIdx != (len(netList)-1):
            nextIdx       = winIdx + 1
            nInputNext    = netList[nextIdx].linear.weight.shape[1]
            nNewInputNext = nInputNext - nPrunedNode

            # prune input weight of next layer
            netList[nextIdx].linear.weight.data = deleteColTensor(netList[nextIdx].linear.weight.data,pruneIdx)
            del netList[nextIdx].linear.weight.grad

            # update input features
            netList[nextIdx].linear.in_features = nNewInputNext
        
        print('--- Hidden NODE No: ',pruneIdx,' is PRUNED ---')
        
    else:
        raise IndexError
    
    return copy.deepcopy(netList)



# initialization
criterion    = nn.CrossEntropyLoss()
device       = torch.device('cpu')
averageInput = meanStd()
bufferData   = torch.Tensor().float()
bufferLabel  = torch.Tensor().long()

# performance
Accuracy     = meanStd()
testingTime  = meanStd()
trainingTime = meanStd()

# flags
driftStatus = 0
growCount   = 0
growLayer   = 0



