#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 15:07:23 2020

@author: emilio
"""

from methods.GeneralMethods import loadDefaultParams

from classes.Dataset import Dataset

from pathlib import Path

defaultParams = loadDefaultParams(defParamBase = ".")
dataPath = defaultParams['dataPath']

data = []
data.append({'description': 'Earl 2019-03-18\nM1 - MGR',
              'path': dataPath / Path('memoryGuidedReach/Earl/2019/03/18/'),
              'delayStartStateName': 'Delay Period',
              'processor': 'Erinn'});
data.append({'description': 'Earl 2019-03-22\nM1 - MGR',
              'path': dataPath / Path('memoryGuidedReach/Earl/2019/03/22/'),
              'delayStartStateName': 'Delay Period',
              'processor': 'Erinn'});
data.append({'description': 'Pepe A1 2018-07-14\nPFC - MGS',
              'path': dataPath / Path('memoryGuidedSaccade/Pepe/2018/07/14/Array1/'),
              'delayStartStateName': 'Delay Period',
              'processor': 'Yuyan'});
data.append({'description': 'Pepe A2 2018-07-14\nPFC - MGS',
              'path': dataPath / Path('memoryGuidedSaccade/Pepe/2018/07/14/Array2/'),
              'delayStartStateName': 'Delay Period',
              'processor': 'Yuyan'});
data.append({'description': 'Wakko A1 2018-02-11\nPFC - MGS',
              'path': dataPath / Path('memoryGuidedSaccade/Wakko/2018/02/11/Array1/'),
              'delayStartStateName': 'Delay Period',
              'processor': 'Yuyan'});
data.append({'description': 'Wakko A2 2018-02-11\nPFC - MGS',
              'path': dataPath / Path('memoryGuidedSaccade/Wakko/2018/02/11/Array2/'),
              'delayStartStateName': 'Delay Period',
              'processor': 'Yuyan'});
data.append({'description': 'Pepe 2016-02-02\nPFC - cuedAttn',
              'path': dataPath / Path('cuedAttention/Pepe/2016/02/02/Array1_PFC/'),
              'delayStartStateName': 'Blank Before',
              'processor': 'Emilio'});
data.append({'description': 'Pepe 2016-02-02\nV4 - cuedAttn',
              'path': dataPath / Path('cuedAttention/Pepe/2016/02/02/Array2_V4/'),
              'delayStartStateName': 'Blank Before',
              'processor': 'Emilio'});

#%% process data
processedDataMat = 'processedData.mat'

# dataset = [ [] for _ in range(len(data))]
cosTuningCurves = []
dataIndsProcess = [1,4,7]
datasets = []
for ind in dataIndsProcess:
# for ind, dataUse in enumerate(data):
#    dataUse = data[1]
    dataUse = data[ind]
    dataMatPath = dataUse['path'] / processedDataMat
    
    print('processing data set ' + dataUse['description'])
    datasetHere = Dataset(dataMatPath, dataUse['processor'], notChan = [31,0])
    data[ind]['dataset'] = datasetHere
    datasets.append(datasetHere)
    
    cosTuningCurves.append(datasetHere.computeCosTuningCurves())
    
#%% get desired time segment
from classes.BinnedSpikeSet import BinnedSpikeSet
from methods.BinnedSpikeSetListMethods import generateBinnedSpikeListsAroundDelay as genBSLAroundDelay
from methods.BinnedSpikeSetListMethods import generateBinnedSpikeListsGroupedByLabel as genBSLLabGrp
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

descriptions = [data[idx]['description'] for idx in dataIndsProcess]
paths = [data[idx]['path'] for idx in dataIndsProcess]

xDimTest = [2,5,8,12,15]#[2,5,8,12,15]
firingRateThresh = 0.5
combineConditions = False
numStimulusConditions = False # because V4 has two...

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
listBSS = binnedSpikesShortStartOffshift
# listConcatBSS = [np.concatenate(bnSp, axis=1).view(BinnedSpikeSet)[None,:,:] for bnSp in listBSS]
# chansGoodBSS = [bnSp.channelsAboveThresholdFiringRate(firingRateThresh)[1] for bnSp in listConcatBSS]
# listBSS = [(bnSp[:,:,:]-conBS.mean(axis=2)[:,:,None])/conBS.std(axis=2)[:,:,None] for bnSp, conBS in zip(listBSS, listConcatBSS)]
# listBSS = [bnSp[:,chK,:] for bnSp, chK in zip(listBSS, chansGoodBSS)]
timeBeforeAndAfterStart = (0+offshift, furthestForward+offshift)
timeBeforeAndAfterEnd = None
baselineSubtract = True
signalDescriptor = "first%dMsDelayOffshift%dMsFRThresh%0.2f%s" % (furthestForward, offshift,firingRateThresh, "Bsub" if baselineSubtract else "")




dimsB, gpPrepB = gpfaComputation(
    listBSS, descriptions, paths, signalDescriptor = signalDescriptor,
                # [listBSS[-1]], [descriptions[-1]], [paths[-1]],
                          timeBeforeAndAfterStart = timeBeforeAndAfterStart, timeBeforeAndAfterEnd = timeBeforeAndAfterEnd,
                          balanceDirs = True, numStimulusConditions = numStimulusConditions, baselineSubtract = baselineSubtract, 
                          crossvalidateNumFolds = 4, xDimTest = xDimTest, firingRateThresh=firingRateThresh)
