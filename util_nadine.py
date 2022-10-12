import numpy as np
import pandas as pd
import time 
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import linalg as LA
import scipy
from scipy import io
from scipy.stats.distributions import chi2
import pdb

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
class anomalyData(object):
    def __init__(self,nInput):
        self.Lambda               = 0.98               # Forgetting factor
        self.StabilizationPeriod  = 20                 # The length of stabilization period.
        self.indexStableExecution = nInput
        self.na                   = 10                 # number of consequent anomalies to be considered as change
        self.Threshold1           = chi2.ppf(0.99, df = nInput)
        self.Threshold2           = chi2.ppf(0.999,df = nInput)
        self.indexkAnomaly        = 0
        self.invCov               = torch.eye(nInput,nInput)
        self.center               = torch.zeros(1,nInput)
        self.caCounter            = 0
        self.anomalyData          = torch.Tensor().float()      # Identified anoamlies input
        self.anomalyLabel         = torch.Tensor().long()       # Identified anoamlies target
        self.anomalyIndices       = torch.Tensor().long()        # indices of Identified anoamlies target
        self.ChangePoints         = []                          # Index of identified change points
        
    def reset(self):
        self.indexkAnomaly  = 0
        self.invCov         = torch.eye(nInput,nInput)
        self.center         = torch.zeros(1,nInput)
        self.caCounter      = 0
        self.ChangePoints   = []
        self.anomalyIndices = torch.Tensor().long()
        
    def updateCenterCov(self,x):  
        # (InvCov,center,indexkAnomaly,Lambda,x)
        with torch.no_grad():
            default_Eff_Number = 200
            indexOfSample      = np.min([self.indexkAnomaly,default_Eff_Number])
            temp1              = self.mahalDist(x)
            temp1              = temp1 + (self.indexkAnomaly - 1)/self.Lambda
            multiplier         = ((self.indexkAnomaly)/((self.indexkAnomaly - 1)*self.Lambda))
            invCov             = (self.invCov - (torch.matmul(torch.matmul(self.invCov,(x - self.center).transpose(0,1)),
                                                              torch.matmul((x - self.center),self.invCov))/temp1))
            self.invCov        = multiplier*invCov
            self.center        = self.Lambda*self.center + (1.0 - self.Lambda)*x
        
    def updateAnomaly(self, x, y, indice, avgX, score, cnt = 1):
        with torch.no_grad():
            self.indexkAnomaly += cnt

            if self.indexkAnomaly <= self.indexStableExecution:
                self.center = avgX

            elif self.indexkAnomaly > self.indexStableExecution:
                mahaldist        = self.mahalDist(x)
                sortedScore,_    = torch.sort(F.softmax(score,dim=1),descending=True)
                sortedScore      = sortedScore.squeeze(dim=0).tolist()
                decisionBoundary = sortedScore[0]/(sortedScore[0] + sortedScore[1])

                if self.indexkAnomaly > self.StabilizationPeriod:
                    # Threshold 1 and Threshold 2 are obtained using chi2inv
                    # (0.99,I) and chi2inv(0.999,I), the data point is regarded as an anomaly if
                    # the condition below is fulfilled. After this condition is
                    # executed, the CACounter is resetted to zero.
                    if ((mahaldist > self.Threshold1 and mahaldist <self.Threshold2) 
                        or decisionBoundary <= 0.55):
                        self.anomalyIndices = torch.cat((self.anomalyIndices,indice),0)
                    else:
                        self.caCounter += cnt

                if (self.caCounter >= self.na):
                    self.ChangePoints.append(self.indexkAnomaly - self.caCounter)
                    self.caCounter = 0

                self.updateCenterCov(x)
    
    def addDataToAnomaly(self,data,label,nHl):
        anomalyData         = torch.index_select(data,  0, self.anomalyIndices)
        anomalyLabel        = torch.index_select(label, 0, self.anomalyIndices)
        self.anomalyData    = torch.cat((self.anomalyData,anomalyData),0)
        self.anomalyLabel   = torch.cat((self.anomalyLabel,anomalyLabel),0)
        self.anomalyIndices = torch.Tensor().long()
        
        if self.anomalyData.shape[0] > 5000*nHl:
            newIndex                 = self.anomalyData.shape[0] - 5000*nHl
            self.anomalyData         = self.anomalyData[newIndex:]
            self.anomalyLabel        = self.anomalyLabel[newIndex:]
            
    def mahalDist(self,x):
        with torch.no_grad():
            mahaldist = torch.matmul(torch.matmul((x-self.center),self.invCov),(x-self.center).transpose(0,1))
            self.mahaldist = mahaldist[0][0].tolist()
        
        return mahaldist
    
def probitFunc(meanIn,stdIn):
    stdIn += 0.0001  # for safety
    out = meanIn/(torch.ones(1) + (np.pi/8)*stdIn**2)**0.5
    
    return out
def generateWeightXavInit(nInput,nNode,nOut,nNewNode):
    copyIn          = createHiddenLayer(nInput,nNode)
    copyOut         = createOutputLayer(nNode,nOut)
    newWeight       = copyIn.linear.weight.data[0:nNewNode]
    newOutputWeight = copyOut.linearOutput.weight.data[:,0:nNewNode]
    
    return newWeight, newOutputWeight

def deleteRowTensor(x,index):
    x = x[torch.arange(x.size(0))!=index] 
    
    return x

def deleteColTensor(x,index):
    x = x.transpose(1,0)
    x = x[torch.arange(x.size(0))!=index]
    x = x.transpose(1,0)
    
    return x

def oneHot(label,nClass):
    nData = label.shape[0]
    
    oneHot = torch.zeros(nData,nClass)
    
    for i, lbl in enumerate(label):
        oneHot[i][lbl] = 1
    
    return oneHot

class hiddenLayer(nn.Module):
    def __init__(self, no_input, no_hidden):
        super(hiddenLayer, self).__init__()
        # hidden layer
        self.linear = nn.Linear(no_input, no_hidden,  bias=True)
        self.activation = nn.Sigmoid()
        nn.init.xavier_uniform_(self.linear.weight)
        self.linear.bias.data.zero_()
        
    def forward(self, x):
        x = self.linear(x)
        x = self.activation(x)
        
        return x
    
def createHiddenLayer(no_input,no_hidden):
    obj = hiddenLayer(no_input,no_hidden)
    
    return obj

class outputLayer(nn.Module):
    def __init__(self, no_hidden, classes):
        super(outputLayer, self).__init__()
        # softmax layer
        self.linearOutput = nn.Linear(no_hidden, classes,  bias=True)
        nn.init.xavier_uniform_(self.linearOutput.weight)
        self.linearOutput.bias.data.zero_()
        
    def forward(self, x):
        x = self.linearOutput(x)
        
        return x
    
def createOutputLayer(no_hidden,classes):
    obj = outputLayer(no_hidden,classes)
    return obj

def nadineFeedforwardTest(netList,x,device):
    # feedforward to all layers
    with torch.no_grad():
        tempVar = x.to(device)
        tempVar = tempVar.type(torch.float)
        
        hList   = []

        for netLen in range(len(netList)):
            currnet = netList[netLen]
            obj     = currnet.eval()
            obj     = obj.to(device)
            tempVar = obj(tempVar)
            if netLen < len(netList) - 1:
                hList = hList + [tempVar.tolist()]

    return tempVar, hList


def nadineFeedforwardBiasVar(netList,x,y,device):
    # feedforward to all layers
    # y in one hot vector form, float, already put in device
    with torch.no_grad():
        classes = netList[-1].linearOutput.weight.shape[0]

        tempVar = x.to(device)
        tempVar = tempVar.type(torch.float)

        for netLen in range(len(netList)):
            currnet           = netList[netLen]
            obj               = currnet.eval()
            obj               = obj.to(device)
            
            if netLen == 0:
                tempVar  = obj(tempVar)
                tempVar2 = (tempVar.clone().detach())**2
            else:
                if netLen == len(netList) - 1:
                    hRlast = tempVar.clone().detach()  # node significance of the last hidden layer
                
                tempVar  = obj(tempVar)
                tempVar2 = obj(tempVar2)
                
        # bias variance
        tempY    = F.softmax(tempVar,dim=1)       # y
        tempY2   = F.softmax(tempVar2,dim=1)      # y2
        bias     = torch.norm((tempY - y)**2)     # bias
        variance = torch.norm(tempY2 - tempY**2)  # variance

    return bias.tolist(), variance.tolist(), hRlast

def nadineFeedforwardTrain(netList,x,device):
    
    tempVar = x.to(device)
    tempVar = tempVar.type(torch.float)
    tempVar.requires_grad_()
    
    # feedforward to all layers
    for netLen in range(len(netList)):
        currnet = netList[netLen]
        obj     = currnet.train()
        obj     = obj.to(device)
        tempVar = obj(tempVar)
            
    return tempVar

def nadineTrain(netList,x,y,nClass,anomaly,miuX,miuBias,miuVar,lr,dLr,criterion,device,epoch=1):
    
    # flags
    growNode  = False
    pruneNode = False
    
    # shuffle the data
    nData = x.shape[0]
    shuffled_indices = torch.randperm(nData)
    
    # label for bias var calculation
    y_biasVar = F.one_hot(y).float()
    if y_biasVar.shape[1] != nClass:
        y_biasVar = oneHot(y,nClass)
    
    for iData in range(0,nData):
        # load data
        indices                 = shuffled_indices[iData:iData+1]
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
        bias, variance, nodeSignificance = nadineFeedforwardBiasVar(netList,
                                                                     outProbit,minibatch_label_biasVar,device)
        
        # calculate mean of bias
        miuBias.calcMeanStd(bias)
        if miuBias.count < 1 or growNode:
            miuBias.resetMinMeanStd()
        else:
            miuBias.calcMeanStdMin()
        
        # calculate mean of variance
        miuVar.calcMeanStd(variance)
        if miuVar.count < 20 or pruneNode:
            miuVar.resetMinMeanStd()
        else:
            miuVar.calcMeanStdMin()
        
        # growing
        growNode = growNodeIdentification(bias,miuBias.minMean,miuBias.minStd,
                                          miuBias.mean,miuBias.std)
        if growNode and miuBias.count >= 1:
            # grow a node
            netList = nodeGrowing(netList,1)
        
        # pruning
        pruneNode = pruneNodeIdentification(variance,miuVar.minMean,miuVar.minStd,
                                            miuVar.mean,miuVar.std)
        if (pruneNode and growNode == 0 and miuVar.count >= 20 and 
           netList[-2].linear.weight.data.shape[0] > netList[-1].linearOutput.weight.data.shape[0]):
            pruneIdx = findLeastSignificantNode(nodeSignificance)
            
            # prune a node
            netList  = nodePruning(netList,pruneIdx)
        
        # declare parameters to be trained
        for netLen in range(len(netList)):
            netOptim  = []
            netOptim  = netOptim + list(netList[netLen].parameters())
            if netLen == 0:
                optimizer = torch.optim.SGD(netOptim, lr = dLr[netLen], momentum = 0.95) #, weight_decay = 5e-4)
                # optimizer = torch.optim.Adam(netOptim, lr = 0.05, weight_decay = 5e-4)
            elif netLen > 0 and netLen <= len(netList) - 2:
                optimizer.add_param_group({'lr': dLr[netLen],'params': netOptim}) 
            else:
                optimizer.add_param_group({'lr': lr,'params': netOptim})
        
        # feedforward
        scores = nadineFeedforwardTrain(netList,minibatch_x,device)
        
        # calculate loss
        minibatch_label = minibatch_label.long()
        loss            = criterion(scores,minibatch_label)
        
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # detect anomaly data
        anomaly.updateAnomaly(minibatch_x, minibatch_label, indices, miuX.mean, scores.clone().detach())
        
    # add anomaly data to buffer
    nHl = len(netList) - 1
    anomaly.addDataToAnomaly(x,y,nHl)
    
    print('Bias: ',miuBias.mean)
    print('Variance: ',miuVar.mean)
    
    return netList, miuX, miuBias, miuVar, anomaly

def nadineTest(netList,test_data,test_label,criterion,device):
    # load data
    test_data  = test_data.to(device)
    test_label = test_label.to(device)
    test_label = test_label.long()
    
    # testing
    start_test              = time.time()
    scores,hRList           = nadineFeedforwardTest(netList,test_data,device)
    rawPredicted, predicted = torch.max(F.softmax(scores.data,dim=1), 1)
    end_test                = time.time()

    # performance calculation
    loss          = criterion(scores,test_label)
    residualError = torch.tensor([1.0]) - rawPredicted
    correct       = (predicted == test_label).sum().item()
    accuracy      = 100*correct/(predicted == test_label).shape[0]  # 1: correct, 0: wrong
    testing_time  = end_test - start_test
    F_matrix      = (predicted != test_label).int().tolist()  # 1: wrong, 0: correct
        
    print('Testing Accuracy: {}'.format(accuracy))
    print('Testing Loss: {}'.format(loss))
    print('Testing Time: {}'.format(testing_time))
    
    return scores, loss, residualError, accuracy, testing_time, F_matrix, hRList

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

def growNodeIdentification(bias,minMeanBias,minStdBias,meanBias,stdBias):
    growNode = False
    
    dynamicKsigmaGrow = 1.25*np.exp(-bias) + 0.75
    growCondition1    = minMeanBias + dynamicKsigmaGrow*minStdBias
    growCondition2    = meanBias + stdBias
    
    if growCondition2 > growCondition1:
        growNode = True
    
    return growNode
def pruneNodeIdentification(var,minMeanVar,minStdVar,meanVar,stdVar):
    pruneNode = False
    
    dynamicKsigmaPrune = 1.25*np.exp(-var) + 0.75
    pruneCondition1    = minMeanVar + 2*dynamicKsigmaPrune*minStdVar
    pruneCondition2    = meanVar + stdVar
    
    if pruneCondition2 > pruneCondition1:
        pruneNode = True
    
    return pruneNode

def findLeastSignificantNode(nodeSig):
    leastSigIdx = torch.argmin(torch.abs(nodeSig)).tolist()
    
    return leastSigIdx

def calcDynamicLr(hrlist,y,defaultLr):
    # calculate correlation between hidden node and output
    
    hrOutCorrCoeff = []
    nOut           = y.shape[1]
    y              = F.softmax(y,dim=1)
    y              = y.transpose(0,1)
    
    for i in range(len(hrlist)):
        currHr    = torch.FloatTensor(hrlist[i]).transpose(0,1)
        nCurrNode = torch.FloatTensor(hrlist[i]).transpose(0,1).shape[0]
        
        corrEachLayer = []
        
        for j in range(0,nCurrNode):

            corrEachNode = []
            for k in range(0,nOut):
                currCorr = np.abs(np.corrcoef(currHr[j].tolist(),y[k].tolist())[0][1])
                
                if (currCorr != currCorr).any():
                    print('There is NaN in calcDynamicLr')
                    currCorr = 0.0001
                    
                corrEachNode = corrEachNode + [currCorr]
                
            corrEachLayer = corrEachLayer + [np.average(corrEachNode)]
            
        hrOutCorrCoeff = hrOutCorrCoeff + [np.average(corrEachLayer)]
        
    dLr = defaultLr*np.exp(-1.0*(1.0/np.asarray(hrOutCorrCoeff) - 1.0))
    print('adjust learning rate')
    
    return dLr.tolist()

def layerGrowing(netList):
    nInput      = netList[-1].linearOutput.in_features
    nOutput     = netList[-1].linearOutput.out_features
    
    del netList[-1]
    
    netList     = netList + [createHiddenLayer(nInput,nOutput),
                             createOutputLayer(nOutput,nOutput)]
    
    avgBias     = meanStd()
    avgVar      = meanStd()
    print('*** ADD a new LAYER ***')
    
 
    
    return copy.deepcopy(netList), avgBias, avgVar


def nodeGrowing(netList,nNewNode):
    netList      = copy.deepcopy(netList)
        
    nInputWin    = netList[-2].linear.weight.shape[1]
    nNodeWin     = netList[-2].linear.weight.shape[0]
    nOutput      = netList[-1].linearOutput.weight.shape[0]
    nNewNodeCurr = nNodeWin + nNewNode

    # grow node for current layer, output
    newWeight, newOutputWeight      = generateWeightXavInit(nInputWin,nNewNodeCurr,nOutput,nNewNode)
    netList[-2].linear.weight.data  = torch.cat((netList[-2].linear.weight.data,
                                                      newWeight),0)  # grow input weights
    netList[-2].linear.bias.data    = torch.cat((netList[-2].linear.bias.data,
                                                      torch.zeros(nNewNode)),0)  # grow input bias
    netList[-2].linear.out_features = nNewNodeCurr
    del netList[-2].linear.weight.grad
    del netList[-2].linear.bias.grad

    # grow input weight of linearOutput
    netList[-1].linearOutput.weight.data = torch.cat((netList[-1].linearOutput.weight.data,
                                                            newOutputWeight),1)
    netList[-1].linearOutput.in_features = nNewNodeCurr
    del netList[-1].linearOutput.weight.grad
    del netList[-1].linearOutput.bias.grad

    print('+++ GROW a hidden NODE +++')
    
    return copy.deepcopy(netList)

def nodePruning(netList,pruneIdx):
    netList      = copy.deepcopy(netList)
        
    nNodeLastHL  = netList[-2].linear.weight.shape[0]
    nPrunedNode  = 1
    nNewNodeCurr = nNodeLastHL - nPrunedNode  # prune a node

    # prune node for current layer, output
    netList[-2].linear.weight.data  = deleteRowTensor(netList[-2].linear.weight.data,
                                                       pruneIdx)  # prune input weights
    netList[-2].linear.bias.data    = deleteRowTensor(netList[-2].linear.bias.data,
                                                       pruneIdx)  # prune input bias
    netList[-2].linear.out_features = nNewNodeCurr
    del netList[-2].linear.weight.grad
    del netList[-2].linear.bias.grad

    # prune input weight of linearOutput
    netList[-1].linearOutput.weight.data = deleteColTensor(netList[-1].linearOutput.weight.data,pruneIdx)
    netList[-1].linearOutput.in_features = nNewNodeCurr
    del netList[-1].linearOutput.weight.grad
    del netList[-1].linearOutput.bias.grad

    print('--- Hidden NODE No: ',pruneIdx,' is PRUNED ---')
    
    return copy.deepcopy(netList)