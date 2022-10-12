# %% [markdown]
# ### License

# %% [code]
# NANYANG TECHNOLOGICAL UNIVERSITY - NTUITIVE PTE LTD Dual License Agreement
# Non-Commercial Use Only

# This NTUITIVE License Agreement, including all exhibits ("NTUITIVE-LA") is a legal agreement between you 
# and NTUITIVE (or “we”) located at 71 Nanyang Drive, NTU Innovation Centre, #01-109, Singapore 637722, 
# a wholly owned subsidiary of Nanyang Technological University (“NTU”) for the software or data identified above, 
# which may include source code, and any associated materials, text or speech files, associated media and "online" or 
# electronic documentation and any updates we provide in our discretion (together, the "Software").

# By installing, copying, or otherwise using this Software, found at https://github.com/andriash001 or 
# https://www.researchgate.net/publication/335757711_ADL_Code_mFile or https://github.com/ContinualAL/ADL, 
# you agree to be bound by the terms of this NTUITIVE-LA. If you do not agree, do not install copy or 
# use the Software. The Software is protected by copyright and other intellectual property laws and 
# is licensed, not sold. If you wish to obtain a commercial royalty bearing license to this software 
# please contact us at mpratama@ntu.edu.sg or andriash001@e.ntu.edu.sg.

# SCOPE OF RIGHTS:
# You may use, copy, reproduce, and distribute this Software for any non-commercial purpose, subject to the 
# restrictions in this NTUITIVE-LA. Some purposes which can be non-commercial are teaching, academic research, 
# public demonstrations and personal experimentation. You may also distribute this Software with books or other 
# teaching materials, or publish the Software on websites, that are intended to teach the use of the Software for 
# academic or other non-commercial purposes.

# You may not use or distribute this Software or any derivative works in any form for commercial purposes. Examples 
# of commercial purposes would be running business operations, licensing, leasing, or selling the Software, distributing 
# the Software for use with commercial products, using the Software in the creation or use of commercial products or 
# any other activity which purpose is to procure a commercial gain to you or others.

# If the Software includes source code or data, you may create derivative works of such portions of the Software and 
# distribute the modified Software for non-commercial purposes, as provided herein.

# If you distribute the Software or any derivative works of the Software, you will distribute them under the same terms 
# and conditions as in this license, and you will not grant other rights to the Software or derivative works that are 
# different from those provided by this NTUITIVE-LA.

# If you have created derivative works of the Software, and distribute such derivative works, you will cause 
# the modified files to carry prominent notices so that recipients know that they are not receiving the original Software. 
# Such notices must state: (i) that you have changed the Software; and (ii) the date of any changes.

# You may not distribute this Software or any derivative works. In return, we simply require that you agree:

# That you will not remove any copyright or other notices from the Software.
# That if any of the Software is in binary format, you will not attempt to modify such portions of the Software, or 
# to reverse engineer or decompile them, except and only to the extent authorized by applicable law.

# That NTUITIVE is granted back, without any restrictions or limitations, a non-exclusive, perpetual, irrevocable, 
# royalty-free, assignable and sub-licensable license, to reproduce, publicly perform or display, install, use, 
# modify, post, distribute, make and have made, sell and transfer your modifications to and/or derivative works of 
# the Software source code or data, for any purpose.

# That any feedback about the Software provided by you to us is voluntarily given, and NTUITIVE shall be free to use 
# the feedback as it sees fit without obligation or restriction of any kind, even if the feedback is designated by 
# you as confidential.

# THAT THE SOFTWARE COMES "AS IS", WITH NO WARRANTIES. THIS MEANS NO EXPRESS, IMPLIED OR STATUTORY WARRANTY, 
# INCLUDING WITHOUT LIMITATION, WARRANTIES OF MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE, ANY WARRANTY AGAINST 
# INTERFERENCE WITH YOUR ENJOYMENT OF THE SOFTWARE OR ANY WARRANTY OF TITLE OR NON-INFRINGEMENT. THERE IS NO WARRANTY 
# THAT THIS SOFTWARE WILL FULFILL ANY OF YOUR PARTICULAR PURPOSES OR NEEDS. ALSO, YOU MUST PASS THIS DISCLAIMER ON 
# WHENEVER YOU DISTRIBUTE THE SOFTWARE OR DERIVATIVE WORKS.

# THAT NEITHER NTUITIVE NOR NTU NOR ANY CONTRIBUTOR TO THE SOFTWARE WILL BE LIABLE FOR ANY DAMAGES RELATED TO 
# THE SOFTWARE OR THIS NTUITIVE-LA, INCLUDING DIRECT, INDIRECT, SPECIAL, CONSEQUENTIAL OR INCIDENTAL DAMAGES, 
# TO THE MAXIMUM EXTENT THE LAW PERMITS, NO MATTER WHAT LEGAL THEORY IT IS BASED ON. ALSO, YOU MUST PASS 
# THIS LIMITATION OF LIABILITY ON WHENEVER YOU DISTRIBUTE THE SOFTWARE OR DERIVATIVE WORKS.
# That we have no duty of reasonable care or lack of negligence, and we are not obligated to (and will not) 
# provide technical support for the Software.

# That if you breach this NTUITIVE-LA or if you sue anyone over patents that you think may apply to or read on 
# the Software or anyone's use of the Software, this NTUITIVE-LA (and your license and rights obtained herein) 
# terminate automatically. Upon any such termination, you shall destroy all of your copies of the Software immediately. 
# Sections 3, 4, 5, 6, 7, 8, 11 and 12 of this NTUITIVE-LA shall survive any termination of this NTUITIVE-LA.
# That the patent rights, if any, granted to you in this NTUITIVE-LA only apply to the Software, not to any 
# derivative works you make.

# That the Software may be subject to U.S. export jurisdiction at the time it is licensed to you, and it may be 
# subject to additional export or import laws in other places. You agree to comply with all such laws and 
# regulations that may apply to the Software after delivery of the software to you.
# That all rights not expressly granted to you in this NTUITIVE-LA are reserved.

# That this NTUITIVE-LA shall be construed and controlled by the laws of the Republic of Singapore without 
# regard to conflicts of law. If any provision of this NTUITIVE-LA shall be deemed unenforceable or contrary to law, 
# the rest of this NTUITIVE-LA shall remain in full effect and interpreted in an enforceable manner that 
# most nearly captures the intent of the original language.

# Do you accept all of the terms of the preceding NTUITIVE-LA license agreement? 

# If you accept the terms, click “I Agree,” then “Next.” Otherwise click “Cancel.”

# Copyright (c) NTUITIVE. All rights reserved.

# %% [markdown]
# ### Import

# %% [code]
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
import pdb

# %% [markdown]
# ### Utils ADL

# %% [code]
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

# %% [code]
def probitFunc(meanIn,stdIn):
    stdIn += 0.0001  # for safety
    out = meanIn/(torch.ones(1) + (np.pi/8)*stdIn**2)**0.5
    
    return out

# %% [code]
def generateWeightXavInit(nInput,nNode,nOut,nNewNode):
    copyNet         = devdan(nInput,nNode,nOut)
    newWeight       = copyNet.linear.weight.data[0:nNewNode]
    newWeightNext   = copyNet.linear.weight.data[:,0:nNewNode]
    newOutputWeight = copyNet.linearOutput.weight.data[:,0:nNewNode]
    
    return newWeight, newOutputWeight, newWeightNext

# %% [code]
def deleteRowTensor(x,index):
    x = x[torch.arange(x.size(0))!=index] 
    
    return x

# %% [code]
def deleteColTensor(x,index):
    x = x.transpose(1,0)
    x = x[torch.arange(x.size(0))!=index]
    x = x.transpose(1,0)
    
    return x

# %% [code]
def oneHot(label,nClass):
    nData = label.shape[0]
    
    oneHot = torch.zeros(nData,nClass)
    
    for i, lbl in enumerate(label):
        oneHot[i][lbl] = 1
    
    return oneHot

# %% [code]
def maskingNoise(x,noiseIntensity = 0.1):
    # noiseStr: the ammount of masking noise 0~1*100%
    
    nData, nInput = x.shape
    nMask = np.max([int(noiseIntensity*nInput),1])
    for i,_ in enumerate(x):
        maskIdx = np.random.randint(nInput,size = nMask)
        x[i][maskIdx] = 0
    
    return x

# %% [markdown]
# ### Network

# %% [code]
class devdan(nn.Module):
    def __init__(self, no_input, no_hidden, classes):
        super(devdan, self).__init__()
        
        # encoder
        self.linear = nn.Linear(no_input, no_hidden,  bias=True)
        nn.init.xavier_uniform_(self.linear.weight)
        self.linear.bias.data.zero_()
        
        # decoder
        self.biasDecoder = nn.Parameter(torch.zeros(no_input))
        
        # softmax layer
        self.linearOutput = nn.Linear(no_hidden, classes,  bias=True)
        nn.init.xavier_uniform_(self.linearOutput.weight)
        self.linearOutput.bias.data.zero_()
        
        self.activation = nn.Sigmoid()
        
        
    def forward(self, x, x_2):
        if x is not None:
            x  = self.linear(x)
            h  = self.activation(x)
            
            with torch.no_grad():
                a  = x.clone().detach()
                h2 = (h.clone().detach())**2
                x2 = self.linearOutput(h2)
            
            # decoder
            r  = F.linear(h, self.linear.weight.t()) + self.biasDecoder
            r  = self.activation(r)  # reconstructed input

            # classifier
            x  = self.linearOutput(h)
            
        else:
            x  = torch.tensor([[0.0]])
            h  = torch.tensor([[0.0]])
            r  = torch.tensor([[0.0]])
            x2 = torch.tensor([[0.0]])
            a  = torch.tensor([[0.0]])
        
        if x_2 is not None:
            with torch.no_grad():
                r_2 = F.linear(x_2, self.linear.weight.t()) + self.biasDecoder
                r_2 = self.activation(r_2)  # reconstructed input
        else:
            r_2 = torch.tensor([[0.0]])
        
        return x, h.clone().detach(), x2.clone().detach(), r, a, r_2.clone().detach()
    
def createDevdan(no_input,no_hidden,classes):
    obj = devdan(no_input,no_hidden,classes)
    
    return obj

# %% [code]
def devdanFeedforwardTest(netList,x,votingWeight,device):
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
            currnet                  = netList[netLen]
            obj                      = currnet.eval()
            obj                      = obj.to(device)
            tempY, tempVar,_,_,_,_   = obj(tempVar,None)
            hList                    = hList + [tempVar.tolist()]
            y                        = y + tempY*votingWeight[netLen]
            if votingWeight[netLen] == 0:
                yList        = yList + [[]]
            else:
                yList        = yList + [F.softmax(tempY,dim=1).tolist()]

    return y, yList, hList

# %% [code]
def devdanFeedforwardBiasVarDisc(netList,netWinIdx,x,y,device):
    # feedforward from the input to the winning layer
    # y in one hot vector form, float, already put in device
    with torch.no_grad():
        minibatch_data  = x.to(device)
        minibatch_data  = minibatch_data.type(torch.float)
        minibatch_label = y
        
        tempVar = minibatch_data
        for netLen in range(len(netList)):
            currnet                       = netList[netLen]
            obj                           = currnet.eval()
            obj                           = obj.to(device)
            tempY, tempVar,tempY2,_,_,_   = obj(tempVar,None)
            
            if netLen == 0:
                tempVar2                  = (tempVar.clone().detach())**2
            else:
                tempY2,tempVar2,_,_,_,_,_ = obj(tempVar2,None)
                
            if netLen == netWinIdx:
                break
        
        tempY    = F.softmax(tempY,dim=1)
        tempY2   = F.softmax(tempY2,dim=1)
        bias     = torch.norm((tempY - minibatch_label)**2)
        variance = torch.norm(tempY2 - tempY**2)

    return bias.tolist(), variance.tolist(), tempVar

# %% [code]
def devdanFeedforwardBiasVarGen(netList,netWinIdx,x,avgFeature,device):
    # feedforward from the input to the winning layer
    with torch.no_grad():
        minibatch_data = x.to(device)
        minibatch_data = minibatch_data.type(torch.float)
        tempVar = minibatch_data
        
        for netLen in range(len(netList)):
            currnet = netList[netLen]
            obj     = currnet.eval()
            obj     = obj.to(device)
            
            if netLen == 0:
                _, _,_,_,A,_       = obj(tempVar,None)
                avgFeature.calcMeanStd(A)
                
                tempVar            = probitFunc(avgFeature.mean,avgFeature.std)
                tempVar2           = (tempVar.clone().detach())**2
                _,_,_,_,_,tempVar  = obj(None,tempVar)
                _,_,_,_,_,tempVar2 = obj(None,tempVar2)
                
            else:
                _,_,_,tempVar,_,_  = obj(tempVar,None)
                _,_,_,tempVar2,_,_ = obj(tempVar2,None)
                
            if netLen == netWinIdx:
                break
        
        bias     = torch.mean((tempVar - minibatch_data)**2)
        variance = torch.mean(tempVar2 - minibatch_data**2)

    return bias.tolist(), variance.tolist(), tempVar, avgFeature

# %% [code]
def devdanFeedforwardTrain(netWin,xWin,device):
    # feed forward only on winning layer
    minibatch_data = xWin.to(device)
    minibatch_data = minibatch_data.type(torch.float)
    minibatch_data.requires_grad_()
    
    netWin      = netWin.train()
    netWin      = netWin.to(device)
    y,_,_,r,_,_ = netWin(minibatch_data,None)
    
    return y, r

# %% [markdown]
# ### Train and Test

# %% [code]
def devdanGenTrain(netList,netWinIdx,xWin,x,miuFeature,miuBias,miuVar,lr,criterion,device,epoch=1):
    
    print('GENERATIVE Training')
    
    # flags
    growNode  = False
    pruneNode = False
    
    # shuffle the data
    nData = x.shape[0]
    shuffled_indices = torch.randperm(nData)
    
    # masked input
    maskedX = maskingNoise(xWin.clone().detach())   # some of the input feature
#     maskedX = xWin.clone().detach()                   # without masking noise
    
    for iData in range(0,nData):
        # load data
        indices         = shuffled_indices[iData:iData+1]
        
        minibatch_xWin  = maskedX[indices]
        minibatch_xWin  = minibatch_xWin.to(device)
         
        minibatch_label = xWin[indices]
        minibatch_label = minibatch_label.to(device)
         
        minibatch_x     = x[indices]
        minibatch_x     = minibatch_x.to(device)
                
        # get bias and variance generative
        bias, variance, nodeSignificance, miuFeature = devdanFeedforwardBiasVarGen(
            netList,netWinIdx,minibatch_x,miuFeature,device)
        
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
            netList    = nodeGrowing(netList,netWinIdx,1)
            if netWinIdx == 0:
                miuFeature = meanStd()
        
        # pruning
        pruneNode = pruneNodeIdentification(variance,miuVar[netWinIdx].minMean,miuVar[netWinIdx].minStd,
                                            miuVar[netWinIdx].mean,miuVar[netWinIdx].std)
        if (pruneNode and not growNode and miuVar[netWinIdx].count >= 20 and 
           netList[netWinIdx].linear.weight.data.shape[0] > netList[netWinIdx].linearOutput.weight.data.shape[0]):
            pruneIdx   = findLeastSignificantNode(nodeSignificance)
            
            # prune a node
            netList    = nodePruning(netList,netWinIdx,pruneIdx)
            if netWinIdx == 0:
                miuFeature = meanStd()
            
        # active learning
        # if not growNode and not pruneNode and activeLearn:
            # active learning can be executed if there is no growing and pruning and active learning is triggered
        
        # declare parameters to be trained
        netOptim  = []
        netOptim  = netOptim + list(netList[netWinIdx].parameters())
        optimizer = torch.optim.SGD(netOptim, lr = lr) #, weight_decay = 5e-4)
        # optimizer = torch.optim.Adam(netOptim, lr = 0.05, weight_decay = 5e-4)
        
        # feedforward
        _,scores  = devdanFeedforwardTrain(netList[netWinIdx],minibatch_xWin,device)
        
        # calculate loss
        loss      = 0.5*criterion(scores,minibatch_label)
        
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
#     print('Bias: ',miuBias[netWinIdx].mean)
#     print('Variance: ',miuVar[netWinIdx].mean)
    
    return netList, miuFeature, miuBias, miuVar

# %% [code]
def devdanDiscTrain(netList,netWinIdx,xWin,x,y,nClass,miuX,miuFeature,miuBias,miuVar,lr,criterion,device,epoch=1):
    
    print('DISCRIMINATIVE Training')
    
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
        
        # get bias and variance discriminative
        outProbit = probitFunc(miuX.mean,miuX.std)
        bias, variance, nodeSignificance = devdanFeedforwardBiasVarDisc(netList,netWinIdx,
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
            if netWinIdx == 0:
                miuFeature = meanStd()
        
        # pruning
        pruneNode = pruneNodeIdentification(variance,miuVar[netWinIdx].minMean,miuVar[netWinIdx].minStd,
                                            miuVar[netWinIdx].mean,miuVar[netWinIdx].std)
        if (pruneNode and not growNode and miuVar[netWinIdx].count >= 20 and 
           netList[netWinIdx].linear.weight.data.shape[0] > netList[netWinIdx].linearOutput.weight.data.shape[0]):
            pruneIdx = findLeastSignificantNode(nodeSignificance)
            
            # prune a node
            netList  = nodePruning(netList,netWinIdx,pruneIdx)
            if netWinIdx == 0:
                miuFeature = meanStd()
            
        # active learning
        # if not growNode and not pruneNode and activeLearn:
            # active learning can be executed if there is no growing and pruning and active learning is triggered
        
        # declare parameters to be trained
        netOptim  = []
        netOptim  = netOptim + list(netList[netWinIdx].parameters())
        optimizer = torch.optim.SGD(netOptim, lr = lr, momentum = 0.95) #, weight_decay = 5e-4)
        # optimizer = torch.optim.Adam(netOptim, lr = 0.05, weight_decay = 5e-4)
        
        # feedforward
        scores,_  = devdanFeedforwardTrain(netList[netWinIdx],minibatch_xWin,device)
        
        # calculate loss
        minibatch_label = minibatch_label.long()
        loss            = criterion(scores,minibatch_label)
        
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
#     print('Bias: ',miuBias[netWinIdx].mean)
#     print('Variance: ',miuVar[netWinIdx].mean)
    
    return netList, miuX, miuBias, miuVar, miuFeature

# %% [code]
def devdanTest(netList,votingWeight,test_data,test_label,batch_size,criterion,device):
    # load data
    test_data  = test_data.to(device)
    test_label = test_label.to(device)
    test_label = test_label.long()
    
    # testing
    start_test              = time.time()
    scores,scoresList,_     = devdanFeedforwardTest(netList,test_data,votingWeight,device)
    rawPredicted, predicted = torch.max(F.softmax(scores.data,dim=1), 1)
    end_test                = time.time()

    # performance calculation
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
            _, predicted = torch.max(torch.FloatTensor(scoresList[netLen]).data, 1)
            F_matrixList = F_matrixList + [(predicted != test_label).int().tolist()]  # 1: wrong, 0: correct
            loacalLoss   = criterion(torch.FloatTensor(scoresList[netLen]),test_label)
            lossList     = lossList + [loacalLoss.tolist()]
        
    print('Testing Accuracy: {}'.format(accuracy))
    print('Testing Loss: {}'.format(loss))
    print('Testing Time: {}'.format(testing_time))
    
    return scores, scoresList, loss, lossList, residualError, accuracy, testing_time, F_matrix, F_matrixList

# %% [markdown]
# ### Network Evaluation

# %% [code]
def growNodeIdentification(bias,minMeanBias,minStdBias,meanBias,stdBias):
    growNode = False
    
    dynamicKsigmaGrow = 1.3*np.exp(-bias) + 0.7
    growCondition1    = minMeanBias + dynamicKsigmaGrow*minStdBias
    growCondition2    = meanBias + stdBias
    
    if growCondition2 > growCondition1:
        growNode = True
    
    return growNode

# %% [code]
def pruneNodeIdentification(var,minMeanVar,minStdVar,meanVar,stdVar):
    pruneNode = False
    
    dynamicKsigmaPrune = 1.3*np.exp(-var) + 0.7
    pruneCondition1    = minMeanVar + 2*dynamicKsigmaPrune*minStdVar
    pruneCondition2    = meanVar + stdVar
    
    if pruneCondition2 > pruneCondition1:
        pruneNode = True
    
    return pruneNode

# %% [code]
def findLeastSignificantNode(nodeSig):
    leastSigIdx = torch.argmin(torch.abs(nodeSig)).tolist()
    
    return leastSigIdx

# %% [markdown]
# ### Evolving

# %% [code]
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
            netList[nextIdx].biasDecoder.data   = torch.cat((netList[nextIdx].biasDecoder.data,
                                                             torch.zeros(nNewNode)),0)
            del netList[nextIdx].linear.weight.grad
            del netList[nextIdx].biasDecoder.grad

            # update input features
            netList[nextIdx].linear.in_features = nNewInputNext
            
        print('+++ GROW a hidden NODE +++')
    else:
        raise IndexError
    
    return copy.deepcopy(netList)

# %% [code]
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
            netList[nextIdx].biasDecoder.data   = deleteRowTensor(netList[nextIdx].biasDecoder.data,pruneIdx)
            del netList[nextIdx].linear.weight.grad
            del netList[nextIdx].biasDecoder.grad

            # update input features
            netList[nextIdx].linear.in_features = nNewInputNext
        
        print('--- Hidden NODE No: ',pruneIdx,' is PRUNED ---')
        
    else:
        raise IndexError
    
    return copy.deepcopy(netList)


