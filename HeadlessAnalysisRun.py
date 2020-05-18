#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 15:07:23 2020

@author: emilio
"""

from methods.GeneralMethods import loadDefaultParams

from classes.Dataset import Dataset

from pathlib import Path

import numpy as np

defaultParams = loadDefaultParams(defParamBase = ".")
dataPath = defaultParams['dataPath']

data = []
data.append({'description': 'Earl 2019-03-18 M1 - MGR',
              'path': dataPath / Path('memoryGuidedReach/Earl/2019/03/18/'),
              'delayStartStateName': 'Delay Period',
              'processor': 'Erinn'});
data.append({'description': 'Earl 2019-03-22 M1 - MGR',
              'path': dataPath / Path('memoryGuidedReach/Earl/2019/03/22/'),
              'delayStartStateName': 'Delay Period',
              'processor': 'Erinn'});
data.append({'description': 'Pepe A1 2018-07-14 PFC - MGS',
              'path': dataPath / Path('memoryGuidedSaccade/Pepe/2018/07/14/Array1_PFC/'),
              'delayStartStateName': 'TARG_OFF',
              'processor': 'Emilio'});
data.append({'description': 'Pepe A2 2018-07-14 PFC - MGS',
              'path': dataPath / Path('memoryGuidedSaccade/Pepe/2018/07/14/Array2_PFC/'),
              'delayStartStateName': 'TARG_OFF',
              'processor': 'Emilio'});
data.append({'description': 'Wakko A1 2018-02-11 PFC - MGS',
              'path': dataPath / Path('memoryGuidedSaccade/Wakko/2018/02/11/Array1_PFC/'),
              'delayStartStateName': 'TARG_OFF',
              'processor': 'Emilio'});
data.append({'description': 'Wakko A2 2018-02-11 PFC - MGS',
              'path': dataPath / Path('memoryGuidedSaccade/Wakko/2018/02/11/Array2_PFC/'),
              'delayStartStateName': 'TARG_OFF',
              'processor': 'Emilio'});
data.append({'description': 'Pepe 2016-02-02 V4 - cuedAttn',
              'path': dataPath / Path('cuedAttention/Pepe/2016/02/02/Array1_V4/'),
              'delayStartStateName': 'Blank Before',
              'processor': 'Emilio'});
data.append({'description': 'Pepe 2016-02-02 PFC - cuedAttn',
              'path': dataPath / Path('cuedAttention/Pepe/2016/02/02/Array2_PFC/'),
              'delayStartStateName': 'Blank Before',
              'processor': 'Emilio'});

data = np.asarray(data) # to allow fancy indexing
#%% process data
processedDataMat = 'processedData.mat'

# dataset = [ [] for _ in range(len(data))]
dataUseLogical = np.zeros(len(data), dtype='bool')
dataIndsProcess = np.array([6])#np.array([1,4,6])
dataUseLogical[dataIndsProcess] = True

for dataUse in data[dataUseLogical]:
# for ind, dataUse in enumerate(data):
#    dataUse = data[1]
    # dataUse = data[ind]
    dataMatPath = dataUse['path'] / processedDataMat
    
    print('processing data set ' + dataUse['description'])
    datasetHere = Dataset(dataMatPath, dataUse['processor'], notChan = [31,0])
    dataUse['dataset'] = datasetHere
    
    
#%% get desired time segment
from classes.BinnedSpikeSet import BinnedSpikeSet
from methods.BinnedSpikeSetListMethods import generateBinnedSpikeListsAroundDelay as genBSLAroundDelay
binnedSpikes = []
binnedSpikesAll = []
binnedSpikesOnlyDelay = []
binnedSpikesEnd = []
groupedSpikesTrialAvg = []
groupedSpikesEndTrialAvg = []
grpLabels = []

endDelaysFromStartList = []
startDelaysList = []
binnedSpikesShortStart = []
binnedSpikesShortEnd = []
groupedSpikesTrialShortStartAvg = []

# this'll bleed a little into the start of the new stimulus with the offset,
# but before any neural response can happen
lenSmallestTrl = 301 #ms; 
furthestBack = 300 #ms
furthestForward = 300
binSizeMs = 25 # good for PFC LDA #50 # 

trialType = 'successful'

stateNamesDelayStart = [data[ind]['delayStartStateName'] for ind in dataIndsProcess]

binnedSpikes = genBSLAroundDelay(data, 
                                    dataIndsProcess,
                                    stateNamesDelayStart,
                                    trialType = trialType,
                                    lenSmallestTrl=lenSmallestTrl, 
                                    binSizeMs = binSizeMs, 
                                    furthestTimeBeforeDelay=furthestBack, 
                                    furthestTimeAfterDelay=furthestForward,
                                    setStartToDelayEnd = False,
                                    setEndToDelayStart = True)


binnedSpikesAll = genBSLAroundDelay(data, 
                                    dataIndsProcess,
                                    stateNamesDelayStart,
                                    trialType = trialType,
                                    lenSmallestTrl=lenSmallestTrl, 
                                    binSizeMs = binSizeMs, 
                                    furthestTimeBeforeDelay=furthestBack, 
                                    furthestTimeAfterDelay=furthestForward,
                                    setStartToDelayEnd = False,
                                    setEndToDelayStart = False)

binnedSpikesOnlyDelay = genBSLAroundDelay(data, 
                                    dataIndsProcess,
                                    stateNamesDelayStart,
                                    trialType = trialType,
                                    lenSmallestTrl=lenSmallestTrl, 
                                    binSizeMs = binSizeMs, 
                                    furthestTimeBeforeDelay=0, 
                                    furthestTimeAfterDelay=0,
                                    setStartToDelayEnd = False,
                                    setEndToDelayStart = False)

binnedSpikeEnd = genBSLAroundDelay(data, 
                                    dataIndsProcess,
                                    stateNamesDelayStart,
                                    trialType = trialType,
                                    lenSmallestTrl=lenSmallestTrl, 
                                    binSizeMs = binSizeMs, 
                                    furthestTimeBeforeDelay=furthestBack, 
                                    furthestTimeAfterDelay=furthestForward,
                                    setStartToDelayEnd = True,
                                    setEndToDelayStart = False)

binnedSpikesShortStart = genBSLAroundDelay(data, 
                                    dataIndsProcess,
                                    stateNamesDelayStart,
                                    trialType = trialType,
                                    lenSmallestTrl=lenSmallestTrl, 
                                    binSizeMs = binSizeMs, 
                                    furthestTimeBeforeDelay=0, 
                                    furthestTimeAfterDelay=lenSmallestTrl,
                                    setStartToDelayEnd = False,
                                    setEndToDelayStart = True)

binnedSpikesShortEnd = genBSLAroundDelay(data, 
                                    dataIndsProcess,
                                    stateNamesDelayStart,
                                    trialType = trialType,
                                    lenSmallestTrl=lenSmallestTrl, 
                                    binSizeMs = binSizeMs, 
                                    furthestTimeBeforeDelay=lenSmallestTrl, 
                                    furthestTimeAfterDelay=0,
                                    setStartToDelayEnd = True,
                                    setEndToDelayStart = False)

offshift = 75 #ms
binnedSpikesShortStartOffshift = genBSLAroundDelay(data, 
                                    dataIndsProcess,
                                    stateNamesDelayStart,
                                    trialType = trialType,
                                    lenSmallestTrl=lenSmallestTrl, 
                                    binSizeMs = binSizeMs, 
                                    furthestTimeBeforeDelay=-offshift, # note that this starts it *forwards* from the delay
                                    furthestTimeAfterDelay=lenSmallestTrl+offshift,
                                    setStartToDelayEnd = False,
                                    setEndToDelayStart = True)

#%% run GPFA
from methods.BinnedSpikeSetListMethods import gpfaComputation

# if __name__ == '__main__':
#     import multiprocessing as mp
#     mp.set_start_method('forkserver')

descriptions = [data[idx]['description'] for idx in dataIndsProcess]
paths = [data[idx]['path'] for idx in dataIndsProcess]
plotOutput = True

xDimTest = [8]#[2,5,8,12,15]#[2,5,8,12,15]
firingRateThresh = 1
combineConditions = False
numStimulusConditions = None # because V4 has two...
sqrtSpikes = False

# # ** inputs for delay start ** #
# listBSS = binnedSpikesShortStart
# timeBeforeAndAfterStart = (0, furthestForward)
# timeBeforeAndAfterEnd = None
# baselineSubtract = False
# signalDescriptor = "first%dMsDelayFRThresh%0.2f%sCondNum%d" % (furthestForward,firingRateThresh, "Bsub" if baselineSubtract else "", numStimulusConditions)
# # ** inputs for delay end ** #
# listBSS = binnedSpikesShortEnd
# timeBeforeAndAfterStart = None
# timeBeforeAndAfterEnd = (-furthestForward, 0)
# baselineSubtract = True
# signalDescriptor = "last%dMsDelayFRThresh%0.2f%sCondNum%d" % (furthestForward,firingRateThresh, "Bsub" if baselineSubtract else "", numStimulusConditions)
# ** inputs for delay offshifted ** #
#listBSS = binnedSpikesShortStartOffshift
##listConcatBSS = [np.concatenate(bnSp, axis=1).view(BinnedSpikeSet)[None,:,:] for bnSp in listBSS]
##chansGoodBSS = [bnSp.channelsAboveThresholdFiringRate(firingRateThresh)[1] for bnSp in listConcatBSS]
##listBSS = [(bnSp[:,:,:]-conBS.mean(axis=2)[:,:,None])/conBS.std(axis=2)[:,:,None] for bnSp, conBS in zip(listBSS, listConcatBSS)]
##listBSS = [bnSp[:,chK,:] for bnSp, chK in zip(listBSS, chansGoodBSS)]
## -- for removing bad channels --
listBSS = binnedSpikesShortStartOffshift.copy()
#allTrue = np.ones(listBSS[0].shape[1], dtype='bool')
## 70 for PFC
#allTrue[np.array([13,18])] = 0 # get rid of possibly bad channel 72? 12, 10, 76? 38? 39? 45? 53? 81?
#listBSS[0] = listBSS[0][:,allTrue]
timeBeforeAndAfterStart = (0+offshift, furthestForward+offshift)
timeBeforeAndAfterEnd = None
baselineSubtract = True
signalDescriptor = "fst%dMsDelShft%dMsFR%0.2f%s%sRmSprsIncons" % (furthestForward, offshift,firingRateThresh, "Bsub" if baselineSubtract else "", "Sqrt" if sqrtSpikes else "")
# ** inputs for beginning and end ** #
#listBSS = binnedSpikesAll
listBSS = binnedSpikesAll.copy()
#allTrue = np.ones(listBSS[0].shape[1], dtype='bool')
##allTrue[np.array([70,72,12,10,76])] = 0 # get rid of possibly bad channels? 12, 10, 76? 38? 39? 45? 53? 81?
##listBSS[0] = listBSS[0][:,allTrue]
#timeBeforeAndAfterStart = (-furthestBack, furthestForward)
#timeBeforeAndAfterEnd = (-furthestBack, furthestForward)
#baselineSubtract = False
#signalDescriptor = "delayStart%d-%dMsdelayEnd%d-%dMsFR%0.2f%sSR" % (furthestBack, furthestForward, furthestBack, furthestForward,firingRateThresh, "Bsub" if baselineSubtract else "")


crossvalidateNumFolds = 4


dims, dimsLL, gpfaDimOut, gpfaTestInds, gpfaTrainInds = gpfaComputation(
    listBSS, descriptions, paths, signalDescriptor = signalDescriptor,
                # [listBSS[-1]], [descriptions[-1]], [paths[-1]],
                          timeBeforeAndAfterStart = timeBeforeAndAfterStart, timeBeforeAndAfterEnd = timeBeforeAndAfterEnd,
                          balanceDirs = True, numStimulusConditions = numStimulusConditions, combineConditions=combineConditions, baselineSubtract = baselineSubtract, sqrtSpikes = False,
                          crossvalidateNumFolds = crossvalidateNumFolds, xDimTest = xDimTest, firingRateThresh=firingRateThresh, plotOutput = plotOutput)

shCovPropPopByArea = []
shCovPropNeurAvgByArea = []
shLogDetGenCovPropPopByArea = []
for gpfaArea, dimsUse, dimsGpfaUse in zip(gpfaDimOut, dims, dimsLL):
    CR = [(gpfaCond[int(gpfaDimParamsUse)]['allEstParams'][0]['C'][:,:numDimsUse],gpfaCond[int(gpfaDimParamsUse)]['allEstParams'][0]['R']) for gpfaCond, numDimsUse, gpfaDimParamsUse in zip(gpfaArea, dimsUse, dimsGpfaUse)]
    shCovPropPop = [np.trace(C @ C.T) / (np.trace(C @ C.T) + np.trace(R)) for C, R in CR] 
    shCovPropNeurAvg = [np.mean(np.diag(C @ C.T) / (np.diag(C @ C.T) + np.diag(R))) for C, R in CR] 
    shLogDetGenCovPropPop = [np.linalg.slogdet(C.T @ C)[1] / (np.linalg.slogdet(C.T @ C)[1] + np.linalg.slogdet(R)[1]) for C, R in CR]
    shCovPropPopByArea.append(np.array(shCovPropPop))
    shCovPropNeurAvgByArea.append(np.array(shCovPropNeurAvg))
    shLogDetGenCovPropPopByArea.append(np.array(shLogDetGenCovPropPop))

# for the moment all we want to save is this...
#%% now save all the figures
from methods.GeneralMethods import saveFiguresToPdf

saveFiguresToPdf(pdfname=signalDescriptor + "M1")

# now some more figures from matplotlib import pyplot as plt
from matplotlib import pyplot as plt
plt.close('all')
if numStimulusConditions is None:
    numConds = len(np.unique(listBSS[0].labels['stimulusMainLabel'], axis=0))
else:
    numConds = numStimulusConditions
testingArea = 0
for condUse in range(numConds):
    for cValUse in range(crossvalidateNumFolds):
        xDimBest = np.array(dimsB[testingArea][condUse])
        xDimsTest = np.array(list(gpPrepB[testingArea][condUse].dimOutput.keys()))
        # do some loops to get the key for the best xdim (given that it'll be loewr than the returned xDimBest)
        xDimScoreBestIndPos = np.argmin((xDimTest-xDimBest)[(xDimsTest - xDimBest) >= 0])
        xDimScoreBest = xDimsTest[(xDimsTest-xDimBest)>=0][xDimScoreBestIndPos]
        
        # sequences are not necessarily in time sorted order, so find correct sorting
        timeSortInds = np.argsort(gpPrepB[testingArea][condUse].trainInds[cValUse])
        
        seqTrainNewAll = np.asarray(gpPrepB[testingArea][condUse].dimOutput[xDimScoreBest]['seqsTrainNew'][cValUse])
        seqTrainNewAllSort = seqTrainNewAll[timeSortInds]
        seqTrainOrthAll = [sq['xorth'] for sq in seqTrainNewAllSort]
        seqTrainConcat = np.concatenate(seqTrainOrthAll, axis = 1)
        plt.figure()
        plt.plot(seqTrainConcat.T, alpha=0.65)
        plt.title("GPFA trajectories over all trials in cond %d, cval %d" % (condUse, cValUse))
        plt.xlabel("trial bins across all trials concatenated")
        plt.ylabel("GPFA projection")

saveFiguresToPdf(pdfname="gpfaAll" + signalDescriptor + "M1")
