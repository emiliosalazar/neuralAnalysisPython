#%%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 15:37:02 2020

@author: emilio
"""
# NOTE: Changed /Users/emilio/anaconda3/lib/python3.7/site-packages/matlab/engine/basefuture.py _send_bytes to python 3.8 version
from MatFileMethods import LoadMatFile#, QuashStructure
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import use as mtpltUse
from matplotlib import get_backend as mtpltWhatBackend
#mtpltUse('qt5agg') # nobody knows why, but this is necessary to not plot a million figures and *also* the combo tabs...
mtpltUse('qt5agg') # nobody knows why, but this is necessary to not plot a million figures and *also* the combo tabs...
from matplotlib.backends.backend_pdf import PdfPages
try:
    from IPython import get_ipython
except ModuleNotFoundError as e:
    pass # don't think we actually use this at the moment...
#from plotWindow.plotWindow import plotWindow

from classes.Dataset import Dataset
from methods.GeneralMethods import loadDefaultParams

from pathlib import Path

import os

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
              'path': dataPath / Path('memoryGuidedSaccade/Pepe/2018/07/14/Array1_PFC/'),
              'delayStartStateName': 'TARG_OFF',
              'processor': 'Emilio'});
data.append({'description': 'Pepe A2 2018-07-14\nPFC - MGS',
              'path': dataPath / Path('memoryGuidedSaccade/Pepe/2018/07/14/Array2_PFC/'),
              'delayStartStateName': 'TARG_OFF',
              'processor': 'Emilio'});
data.append({'description': 'Wakko A1 2018-02-11\nPFC - MGS',
              'path': dataPath / Path('memoryGuidedSaccade/Wakko/2018/02/11/Array1_PFC/'),
              'delayStartStateName': 'TARG_OFF',
              'processor': 'Emilio'});
data.append({'description': 'Wakko A2 2018-02-11\nPFC - MGS',
              'path': dataPath / Path('memoryGuidedSaccade/Wakko/2018/02/11/Array2_PFC/'),
              'delayStartStateName': 'TARG_OFF',
              'processor': 'Emilio'});
data.append({'description': 'Pepe 2016-02-02\nV4 - cuedAttn',
              'path': dataPath / Path('cuedAttention/Pepe/2016/02/02/Array1_V4/'),
              'delayStartStateName': 'Blank Before',
              'processor': 'Emilio'});
data.append({'description': 'Pepe 2016-02-02\nPFC - cuedAttn',
              'path': dataPath / Path('cuedAttention/Pepe/2016/02/02/Array2_PFC/'),
              'delayStartStateName': 'Blank Before',
              'processor': 'Emilio'});

data = np.asarray(data) # to allow fancy indexing
#%% process data
processedDataMat = 'processedData.mat'

# dataset = [ [] for _ in range(len(data))]
dataUseLogical = np.zeros(len(data), dtype='bool')
dataIndsProcess = np.array([6])#np.array([4])#np.array([1,4,6])
dataUseLogical[dataIndsProcess] = True

for dataUse in data[dataUseLogical]:
# for ind, dataUse in enumerate(data):
#    dataUse = data[1]
    # dataUse = data[ind]
    dataMatPath = dataUse['path'] / processedDataMat
    
    print('processing data set ' + dataUse['description'])
    datasetHere = Dataset(dataMatPath, dataUse['processor'], notChan = [31,0])
    dataUse['dataset'] = datasetHere
    
#%%
# return binned spikes
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

lenSmallestTrl = 251 #ms
furthestBack = 250 #ms
furthestForward = 250
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

# for ind, dataInfo in enumerate(data):#dataset:#datasetSuccessNoCatch:
#     if ind in dataIndsProcess:
#         dataStInit = dataInfo['dataset'].successfulTrials().trialsWithoutCatch()
#     else:
#         continue
#     startDelay, endDelay = dataStInit.computeDelayStartAndEnd()
#     startDelayArr = np.asarray(startDelay)
#     endTimeArr = np.asarray(endDelay)
#     delayTimeArr = endTimeArr - startDelayArr
    
    
    
    
#     datasets.append(dataStInit.filterTrials(delayTimeArr>lenSmallestTrl))
#     dataSt = datasets[-1]
#     dataSt.computeCosTuningCurves()
#     startDelay, endDelay = dataSt.computeDelayStartAndEnd()
#     startDelayArr = np.asarray(startDelay)
#     startDelaysList.append(startDelayArr)
#     endDelayArr = np.asarray(endDelay)
#     endDelaysFromStartList.append(endDelayArr-startDelayArr)
#     endTimePad = 0
#     endTimeArr = startDelayArr+endTimePad
    
#     binnedSpikesHere = dataSt.binSpikeData(startMs = list(startDelayArr-furthestBack), endMs = list(endTimeArr+furthestForward), binSizeMs=binSizeMs, notChan=[31, 0])
#     binnedSpikes.append(binnedSpikesHere)
    
#     binnedSpikesHereAll = dataSt.binSpikeData(startMs = list(startDelayArr-furthestBack), endMs = list(endDelayArr+furthestForward), binSizeMs=binSizeMs, notChan=[31, 0])
#     binnedSpikesAll.append(binnedSpikesHereAll)
    
#     binnedSpikesHereOnlyDelay = dataSt.binSpikeData(startMs = list(startDelayArr), endMs = list(endDelayArr), binSizeMs=binSizeMs, notChan=[31, 0])
#     binnedSpikesOnlyDelay.append(binnedSpikesHereOnlyDelay)
    
#     binnedSpikesHereEnd = dataSt.binSpikeData(startMs = list(endDelayArr-furthestBack), endMs = list(endDelayArr+furthestForward), binSizeMs=binSizeMs, notChan=[31, 0])
#     binnedSpikesEnd.append(binnedSpikesHereEnd)
    
#     binnedSpikesHereShortStart = dataSt.binSpikeData(startMs = list(startDelayArr), endMs = list(endTimeArr+lenSmallestTrl), binSizeMs=binSizeMs, notChan=[31, 0])
#     binnedSpikesShortStart.append(binnedSpikesHereShortStart)
    
#     binnedSpikesHereShortEnd = dataSt.binSpikeData(startMs = list(endDelayArr-lenSmallestTrl), endMs = list(endDelayArr), binSizeMs=binSizeMs, notChan=[31, 0])
#     binnedSpikesShortEnd.append(binnedSpikesHereShortEnd)
    
#     uniqueTargAngle, trialsPresented = np.unique(dataSt.markerTargAngles, axis=0, return_inverse=True)

#     groupedSpikes = dataSt.groupSpikes(trialsPresented, uniqueTargAngle, binnedSpikes = binnedSpikesHere)
#     groupedSpikesEnd = dataSt.groupSpikes(trialsPresented, uniqueTargAngle, binnedSpikes = binnedSpikesHereEnd)
#     groupedSpikesShortStart = dataSt.groupSpikes(trialsPresented, uniqueTargAngle, binnedSpikes = binnedSpikesHereShortStart)
    
#     targAvgList, targStdList = zip(*[(groupedSpikes[targ].trialAverage(), groupedSpikes[targ].trialStd()) for targ in range(0, len(groupedSpikes))])
#     targTmTrcAvgArr = np.stack(targAvgList).view(BinnedSpikeSet)
#     targTmTrcStdArr = np.stack(targStdList).view(BinnedSpikeSet)
#     targAvgListEnd, targStdListEnd = zip(*[(groupedSpikesEnd[targ].trialAverage(), groupedSpikesEnd[targ].trialStd()) for targ in range(0, len(groupedSpikesEnd))])
#     targTmTrcAvgEndArr = np.stack(targAvgListEnd).view(BinnedSpikeSet)
#     targTmTrcStdEndArr = np.stack(targStdListEnd).view(BinnedSpikeSet)
    
#     targAvgShrtStartList, targStdShrtStartList = zip(*[(groupedSpikesShortStart[targ].trialAverage(), groupedSpikesShortStart[targ].trialStd()) for targ in range(0, len(groupedSpikesShortStart))])
#     targTmTrcShrtStartAvgArr = np.stack(targAvgShrtStartList).view(BinnedSpikeSet)
#     targTmTrcShrtStartStdArr = np.stack(targStdShrtStartList).view(BinnedSpikeSet)
    
#     groupedSpikesTrialShortStartAvg.append([targTmTrcShrtStartAvgArr, targTmTrcShrtStartStdArr])
#     groupedSpikesTrialAvg.append([targTmTrcAvgArr, targTmTrcStdArr])
#     groupedSpikesEndTrialAvg.append([targTmTrcAvgEndArr, targTmTrcStdEndArr])
#     grpLabels.append(uniqueTargAngle)

#%% descriptive population plots
from methods.BinnedSpikeSetListMethods import plotFiringRates

descriptions = [data[idx]['description'] for idx in dataIndsProcess]
listBSS = binnedSpikesShortStartOffshift
plotFiringRates(listBSS, descriptions, supTitle = 'Delay Start Offshift', cumulative = False)

#%% dimensionality calculation
from methods.BinnedSpikeSetListMethods import dimensionalityComputation

descriptions = [data[idx]['description'] for idx in dataIndsProcess]
listBSS = binnedSpikesShortEnd
numDims = dimensionalityComputation(listBSS, descriptions,
                                    labelUse='stimulusMainLabel', 
                                    maxDims = 30,
                                    baselineSubtract = True,
                                    useAllGroupings = False, 
                                    numberNeuronsMatch = 50, # for the moment based on 75% train, 25% test w/ only 70 trials for smallest target num
                                    minWiGroupFR = 0.5, # this is the firing rate of the neurons for the stimulus
                                    plot=True)

# numDims = []
# for idx, bnSp in enumerate(binnedSpikesShortEnd):
#     # grpSpikes = bnSp.groupByLabel(datasets[idx].markerTargAngles)
#     # [bnSpGrp.numberOfDimensions(title = data[idx]['description'], maxDims = 30, baselineSubtract = True) for bnSpGrp in grpSpikes]
#     baselineSubtract = True
#     numDims.append(bnSp.numberOfDimensions(title = data[dataIndsProcess[idx]]['description'], labels=bnSp.labels['stimulusMainLabel'], maxDims = 30, baselineSubtract = False))
#    numDims.append(bnSp.numberOfDimensions(title = data[idx]['description'], labels=None, maxDims = 30))
    
#%% PCA projections
for idx, bnSp in enumerate(binnedSpikesShortEnd):
    bnSp.channelsAboveThresholdFiringRate(firingRateThresh=1)[0].pca(labels = bnSp.labels['stimulusMainLabel'], plot = True)
    plt.suptitle(data[dataIndsProcess[idx]]['description'])
    
#%% LDA projections
for idx, bnSp in enumerate(binnedSpikesShortEnd):
    bnSp.lda(bnSp.labels['stimulusMainLabel'], plot=True)
    plt.suptitle(data[dataIndsProcess[idx]]['description'])
    
#%% Noise correlations
from scipy.stats import binned_statistic
avgSpkCorrOverall = []
geoMeanOverall = []
for idxSpks, bnSp in enumerate(binnedSpikesShortStart):
    idx = dataIndsProcess[idxSpks]
    scatterFig = plt.figure()
    plotResid = True
    separateNoiseCorrForLabels = False
    normalize = True
    residualSpikes, avgSpkCorrAll, geoMeanCntAll = bnSp.channelsAboveThresholdFiringRate(firingRateThresh=0.5)[0].residualCorrelations(bnSp.labels['stimulusMainLabel'], plot=plotResid, separateNoiseCorrForLabels = separateNoiseCorrForLabels, figTitle = data[idx]['description'], normalize = normalize)
    # residualSpikes.pca(labels = datasets[idx].markerTargAngles, plot = True)
    # plt.suptitle(data[idx]['description'] + " residuals PCA")
    # residualSpikes.lda(labels = datasets[idx].markerTargAngles, plot = True)
    # plt.suptitle(data[idx]['description'] + " residuals LDA")
    avgSpkCorrOverall.append(avgSpkCorrAll)
    geoMeanOverall.append(geoMeanCntAll)
    # Only care about single repeats of the non-self pairs...
    upperTriInds = np.triu_indices(avgSpkCorrAll.shape[0], 1)
    avgSpkCorr = avgSpkCorrAll[upperTriInds]
    geoMeanCnt = geoMeanCntAll[upperTriInds]
    
    
    labels=bnSp.labels['stimulusMainLabel']#datasets[idxSpks].markerTargAngles
    uniqueLabel, labelPresented = np.unique(labels, axis=0, return_inverse=True)
    scatterFig.suptitle(data[idx]['description'] + " geo mean FR vs corr var")
    axs = scatterFig.subplots(2,1)
    axs[0].scatter(geoMeanCnt.flatten(), avgSpkCorr.flatten())
    axs[0].set_ylabel("Correlated Variability")
    
    binAvg, binEdges, bN = binned_statistic(geoMeanCnt.flatten(), avgSpkCorr.flatten(), statistic='mean', bins = np.arange(np.max(geoMeanCnt), step=10))
    binStd, _, _ = binned_statistic(geoMeanCnt.flatten(), avgSpkCorr.flatten(), statistic='std', bins = np.arange(np.max(geoMeanCnt), step=10))
    binEdges = np.diff(binEdges)[0]/2+binEdges[:-1]
    axs[1].plot(binEdges, binAvg)
    axs[1].fill_between(binEdges, binAvg-binStd, binAvg+binStd, alpha=0.2)
    axs[1].set_xlabel("Mean Firing Rate")
    axs[1].set_ylabel("Correlated Variability")
    # residualSpikes[labelPresented==0].numberOfDimensions(title=data[idx]['description'] + " residuals", maxDims = 30)
    
    
#%% some descriptive data plots
from methods.BinnedSpikeSetListMethods import generateLabelGroupStatistics as genBSLLabGrp
from methods.BinnedSpikeSetListMethods import plotExampleChannelResponses
from methods.BinnedSpikeSetListMethods import plotStimDistributionHistograms

descriptions = [data[idx]['description'] for idx in dataIndsProcess]
chansAll = np.array([23,-1,-1,-1,23,-1,20,-1], dtype='int16')
# chansAll = np.array([23,-1,-1,-1,70,-1,20,-1], dtype='int16')
chansForPlots = chansAll[dataIndsProcess]

timeBeforeAndAfterStart = (-furthestBack, furthestForward)
timeBeforeAndAfterEnd = (-furthestBack, furthestForward)


listBSS = binnedSpikesShortStartOffshift
listConcatBSS = [np.concatenate(bnSp, axis=1).view(BinnedSpikeSet)[None,:,:] for bnSp in listBSS]
# chansGoodBSS = [bnSp.channelsAboveThresholdFiringRate(firingRateThresh)[1] for bnSp in listConcatBSS]
listBSSFull = binnedSpikes
listBSSEnd = binnedSpikeEnd
listBSSShortStart = binnedSpikesShortStart
groupedSpikesTrialAvg, grpLabels = genBSLLabGrp(listBSSFull, labelUse='stimulusMainLabel')
groupedSpikesEndTrialAvg, grpLabels = genBSLLabGrp(listBSSEnd, labelUse='stimulusMainLabel')
groupedSpikesTrialShortStartAvg, grpLabels = genBSLLabGrp(listBSSShortStart, labelUse='stimulusMainLabel')
plotExampleChannelResponses(groupedSpikesTrialAvg, groupedSpikesTrialShortStartAvg, groupedSpikesEndTrialAvg, 
                                timeBeforeAndAfterStart, timeBeforeAndAfterEnd,
                                grpLabels, descriptions, ylabel = 'Z score', chansForPlots = chansForPlots)

# Z-scored and baseline subbed!
# listBSSFull = [(bnSp[:,:,:]-conBS.mean(axis=2)[:,:,None])/conBS.std(axis=2)[:,:,None] for bnSp, conBS in zip(listBSSFull, listConcatBSS)]
# listBSSEnd = [(bnSp[:,:,:]-conBS.mean(axis=2)[:,:,None])/conBS.std(axis=2)[:,:,None] for bnSp, conBS in zip(listBSSEnd, listConcatBSS)]
# listBSSShortStart = [(bnSp[:,:,:]-conBS.mean(axis=2)[:,:,None])/conBS.std(axis=2)[:,:,None] for bnSp, conBS in zip(listBSSShortStart, listConcatBSS)]
# listBSS = [bnSp[:,chK,:] for bnSp, chK in zip(listBSS, chansGoodBSS)]
# groupedSpikesTrialAvg, grpLabels = genBSLLabGrp([bnSp.baselineSubtract(labels=bnSp.labels['stimulusMainLabel']) for bnSp in listBSSFull], labelUse='stimulusMainLabel')
# groupedSpikesEndTrialAvg, grpLabels = genBSLLabGrp([bnSp.baselineSubtract(labels=bnSp.labels['stimulusMainLabel']) for bnSp in listBSSEnd], labelUse='stimulusMainLabel')
# groupedSpikesTrialShortStartAvg, grpLabels = genBSLLabGrp([bnSp.baselineSubtract(labels=bnSp.labels['stimulusMainLabel']) for bnSp in listBSSShortStart], labelUse='stimulusMainLabel')
# plotExampleChannelResponses(groupedSpikesTrialAvg, groupedSpikesTrialShortStartAvg, groupedSpikesEndTrialAvg, 
#                                 timeBeforeAndAfterStart, timeBeforeAndAfterEnd,
#                                 grpLabels, descriptions, ylabel = 'Z score', chansForPlots = [23,83, 20])



listBSS = binnedSpikesShortStartOffshift
plotStimDistributionHistograms(listBSS, descriptions)
# numCols = 3
# numRows = 3

# ind = 0
# notRndRndChan =  [23,83, 20, 10, 15, 20]# [,23] [23, 79] [5,83]
# for grpSpike, grpSpikeShrtStrt, grpSpikeEnd, grpLabel in zip(groupedSpikesTrialAvg, groupedSpikesTrialShortStartAvg, groupedSpikesEndTrialAvg, grpLabels):
    
#     rndChan = notRndRndChan[ind]#np.random.randint(grpSpike.shape[1])
#     chanRespMean = np.squeeze(grpSpike[0][:,[rndChan], :])
#     chanRespStd = np.squeeze(grpSpike[1][:,[rndChan],:])
#     chanRespEndMean = np.squeeze(grpSpikeEnd[0][:, [rndChan], :])
#     chanRespEndStd = np.squeeze(grpSpikeEnd[1][:, [rndChan], :])
# #    chanResp.timeAverage()
#     chanTmAvg = grpSpikeShrtStrt[0][:,[rndChan],:].timeAverage()
#     chanTmStd = grpSpikeShrtStrt[0][:,[rndChan],:].timeStd()
    
    
#     plt.figure()
#     plt.suptitle(data[ind]['description'] + ': channel ' + str(rndChan))
#     sbpltAngs = np.arange(3*np.pi/4, -5*np.pi/4, -np.pi/4)
#     # Add nan to represent center for tuning polar curve...
#     sbpltAngs = np.concatenate((sbpltAngs[0:3], sbpltAngs[[-1]], np.expand_dims(np.asarray(np.nan),axis=0), sbpltAngs[[3]], np.flip(sbpltAngs[4:-1])), axis=0)
#     axVals = np.empty((0,4))
#     axes = []
#     for idx in range(0, len(grpLabel)):
#         subplotChoose = np.where([np.allclose(grpLabel[idx] % (2*np.pi), sbpltAngs[i] % (2*np.pi)) for i in range(0,len(sbpltAngs))])[0]
#         if not subplotChoose.size:
#             subplotChoose = np.where(sbpltAngs==np.pi*(grpLabel[idx]-2))[0]
#         subplotChooseDelayStart = subplotChoose[0]*2
#         subplotChooseDelayEnd = subplotChoose[0]*2+1
#         axes.append(plt.subplot(numRows, numCols*2, subplotChooseDelayStart+1))
#         tmValStart = np.arange(-furthestBack+binSizeMs/2, furthestForward+binSizeMs/2, binSizeMs)
#         plt.plot(tmValStart, chanRespMean[idx, :])
#         plt.fill_between(tmValStart, chanRespMean[idx, :]-chanRespStd[idx,:], chanRespMean[idx, :]+chanRespStd[idx,:], alpha=0.2)
#         axVals = np.append(axVals, np.array(plt.axis())[None, :], axis=0)
#         axes.append(plt.subplot(numRows, numCols*2, subplotChooseDelayEnd+1))
#         tmValEnd = np.arange(-furthestBack+binSizeMs/2, furthestForward+binSizeMs/2, binSizeMs)
#         plt.plot(tmValEnd, chanRespEndMean[idx, :])
#         plt.fill_between(tmValEnd, chanRespEndMean[idx, :]-chanRespEndStd[idx,:], chanRespEndMean[idx, :]+chanRespEndStd[idx,:], alpha=0.2)
#         axVals = np.append(axVals, np.array(plt.axis())[None, :], axis=0)
#         if subplotChoose[0] == 0:
#             axSt = axes[-2]
#             axSt.set_ylabel('Average Firing Rate (Hz)')
#         elif subplotChoose[0] == (numCols*numRows - np.floor(numCols/2)):
#             axSt = axes[-2]
#             axSt.set_xlabel('Time Around Delay Start or End (ms)')
            
#         if subplotChoose[0] == np.floor(numCols/2):
#             axSt = axes[-2]
#             axSt.set_title('Start')
#             axEnd = axes[-1]
#             axEnd.set_title('End')
        
#         if subplotChoose[0] % numCols:
#             axEn = axes[-1]
#             axSt = axes[-2]
#             axSt.yaxis.set_ticklabels('')
#             axEn.yaxis.set_ticklabels('')
#         else:
#             axEn = axes[-1]
#             axEn.yaxis.set_ticklabels('')
            
#         if subplotChoose[0] < (numCols*numRows - numCols):
#             axEn = axes[-1]
#             axSt = axes[-2]
#             axSt.xaxis.set_ticklabels('')
#             axEn.xaxis.set_ticklabels('')
      
#     ymin = np.min(np.append(0, np.min(axVals, axis=0)[2]))
#     ymax = np.max(axVals, axis=0)[3]
#     for ax in axes:
#         ax.set_ylim(bottom = ymin, top = ymax )
#         plt.axes(ax)
#         plt.axvline(x=0, linestyle='--')
        
        
#     plt.subplot(numRows, numCols, np.where(np.isnan(sbpltAngs))[0][0]+1, projection='polar')
#     ptch = plt.fill(grpLabel, chanTmAvg)
#     ptch[0].set_fill(False)
#     ptch[0].set_edgecolor('k')
# #    plt.fill_between(grpLabel, chanTmAvg-chanTmStd, chanTmAvg+chanTmStd)
#     ind += 1

# ind = 0
# for indUse,dtst in enumerate(datasets):
#     plt.figure();
#     ind = dataIndsProcess[indUse]
#     plt.suptitle(data[ind]['description'])
# #    _, trialsPresented = np.unique(dtst.markerTargAngles, axis=0, return_inverse=True)
#     angsPres = dtst.markerTargAngles
#     angsPres[angsPres>np.pi] = np.pi - (angsPres[angsPres>np.pi] % (2*np.pi))
#     plt.hist(angsPres,bins = np.arange(-7*np.pi/8, 10*np.pi/8, np.pi/4))
#     plt.title('distribution of directions')
#     plt.xlabel('direction angle (rad)')
#     plt.ylabel('count')
#%%
plt.figure()
[plt.boxplot(dts.cosTuningCurveParams['bslnPerChan'], positions=[idx], labels=[data[idx]['description']]) for idx, dts in enumerate(datasets)]
plt.title('Baseline Firing Rate')
plt.ylabel('firing rate (Hz)')
plt.figure()
[plt.boxplot(dts.cosTuningCurveParams['modPerChan']/dts.cosTuningCurveParams['bslnPerChan'], positions=[idx], labels=[data[idx]['description']]) for idx, dts in enumerate(datasets)]
plt.title('Firing Rate Modulation (normalized by baseline)')
plt.ylabel('normalized modulation around baseline')

#%% some GPFA junk...
from methods.BinnedSpikeSetListMethods import gpfaComputation

descriptions = [data[idx]['description'] for idx in dataIndsProcess]
paths = [data[idx]['path'] for idx in dataIndsProcess]

xDimTest = [2,5,8,12,15]#[2,5,8,12,15]
firingRateThresh = 0.5
numStimulusConditions = 1 # because V4 has two...

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
signalDescriptor = "first%dMsDelayOffshift%dMsFRThresh%0.2f%sCondNum%d" % (furthestForward, offshift,firingRateThresh, "Bsub" if baselineSubtract else "", numStimulusConditions)




dimsB, gpPrepB = gpfaComputation(
    listBSS, descriptions, paths, signalDescriptor = signalDescriptor,
                # [listBSS[-1]], [descriptions[-1]], [paths[-1]],
                          timeBeforeAndAfterStart = timeBeforeAndAfterStart, timeBeforeAndAfterEnd = timeBeforeAndAfterEnd,
                          balanceDirs = True, numStimulusConditions = numStimulusConditions, baselineSubtract = baselineSubtract, 
                          crossvalidateNumFolds = 4, xDimTest = xDimTest, firingRateThresh=firingRateThresh)


# BinnedSpikeSet.gpfa(binnedSpikesAll[0],timeBeforeAndAfterStart = None, timeBeforeAndAfterEnd=(-furthestBack, 0))

# binnedSpikesShort = binnedSpikesShortEnd
# dataInd = -2
# # for dataInd in range(len(binnedSpikesAll)):
# for _ in range(1):
#     onlyDelay = True
#     initRandSt = np.random.get_state()
#     np.random.seed(0)
#     if onlyDelay:
#         trlIndsUse = BinnedSpikeSet.balancedTrialInds(binnedSpikesOnlyDelay[dataInd], datasets[dataInd].markerTargAngles)
#         tmValsStart = np.arange(0, 250, binSizeMs)
#         tmValsEnd = np.arange(-250, 0, binSizeMs)       
#         # tmValsStart = np.arange(0, furthestBack if furthestBack else 251, binSizeMs)
#         # tmValsEnd= np.arange(-furthestForward if furthestForward else 251, 0, binSizeMs)
#         binnedSpikesBalanced = [binnedSpikesOnlyDelay[dataInd][trl] for trl in trlIndsUse]
#     else:
#         trlIndsUse = BinnedSpikeSet.balancedTrialInds(binnedSpikesAll[dataInd], datasets[dataInd].markerTargAngles)
#         tmValsStart = np.arange(-furthestBack, furthestBack if furthestBack else 251, binSizeMs)
#         tmValsEnd = np.arange(-(furthestForward if furthestForward else 251)+binSizeMs, furthestForward, binSizeMs)
#         binnedSpikesBalanced = [binnedSpikesAll[dataInd][trl] for trl in trlIndsUse]
        
#     np.random.set_state(initRandSt)
        
#     endDelaysListBal = endDelaysFromStartList[dataInd][trlIndsUse]
#     uniqueTargAngle, trialsPresented = np.unique(datasets[dataInd].markerTargAngles[trlIndsUse], axis=0, return_inverse=True)
#     binSpkAllArr = np.empty(len(binnedSpikesBalanced), dtype=object)
#     for idx, _ in enumerate(binSpkAllArr):
#         binSpkAllArr[idx] = binnedSpikesBalanced[idx]
        
#     groupedBalancedSpikes = Dataset.groupSpikes(_, trialsPresented, uniqueTargAngle, binnedSpikes = binSpkAllArr); trialsPresented = np.sort(trialsPresented) #[binSpkAllArr] #
#     # binnedSpikesBalanced.groupByLabel(datasets[dataInd].markerTargAngles[trlIndsUse])
#     #groupedBalancedEndDelays = Dataset.groupSpikes(_, trialsPresented, uniqueTargAngle, binnedSpikes= endDelaysListBal)
#     groupedBalancedSpikes = [grp.tolist() for grp in groupedBalancedSpikes]
    
#     uniqueTargAngleDeg = uniqueTargAngle*180/np.pi
    
#     colorset = np.array([[127,201,127],[190,174,212],[253,192,134],[200,200,153],[56,108,176],[240,2,127],[191,91,23],[102,102,102]])/255
#     for idx, grpSpks in enumerate(groupedBalancedSpikes):
#         xDim = 8; # default, haven't decided how to be smart about choice yet
#         gpfaPrep = GPFA(grpSpks)
#         estParams, seqTrainNew, seqTestNew = gpfaPrep.runGpfaInMatlab(eng=eng, crossvalidate=True, xDim=xDim)
        
        
#         rowsPlt = 2
#         colsPlt = xDim/rowsPlt*2 # align to start and to end...
#         axesStart = []
#         axesEnd = []
#         axVals = np.empty((0,4))
#         figSep = plt.figure()
#         fig3 = plt.figure()
#         axStart3d = plt.subplot(1,3,1,projection='3d')
#         axEnd3d = plt.subplot(1,3,2,projection='3d')
#         axAll3d = plt.subplot(1,3,3,projection='3d')
#         plt.suptitle(data[dataInd]['description'] + " " + str(int(uniqueTargAngleDeg[idx])) + " deg")
        
#         for sq, trlInd in zip(seqTestNew, gpfaPrep.testInds):
            
#             plt.figure(fig3.number)
#             axStart3d.plot(sq['xorth'][2,:tmValsStart.shape[0]], sq['xorth'][3,:tmValsStart.shape[0]], sq['xorth'][4,:tmValsStart.shape[0]],
#                        color=colorset[trialsPresented[trlInd],:], linewidth=0.4)
#             axStart3d.set_title('Start')
#             axEnd3d.plot(sq['xorth'][2,-tmValsEnd.shape[0]:], sq['xorth'][3,-tmValsEnd.shape[0]:], sq['xorth'][4,-tmValsEnd.shape[0]:],
#                        color=colorset[trialsPresented[trlInd],:], linewidth=0.4)
#             axEnd3d.set_title('End')
#             axAll3d.plot(sq['xorth'][2,:], sq['xorth'][3,:], sq['xorth'][4,:],
#                        color=colorset[trialsPresented[trlInd],:], linewidth=0.4)
#             axAll3d.set_title('All')
            
#             if True:#trialsPresented[trlInd] == 3:
#                 pltNum = 1
#                 plt.figure(figSep.number)
#                 for dim in sq['xorth']:
#                     if len(axesStart) + len(axesEnd) <rowsPlt*colsPlt:
#                         axesHere = figSep.add_subplot(rowsPlt, colsPlt, pltNum)
#                         if pltNum <= colsPlt:
#                             axesHere.set_title("Start")
#                     else:
#                         axesHere = axesStart[int((pltNum-1)/2)]
#                         plt.axes(axesHere)
                    
                    
#                     plt.plot(tmValsStart, dim[:tmValsStart.shape[0]], color=colorset[trialsPresented[trlInd],:], linewidth=0.4)
                    
                    
                    
#                     axVals = np.append(axVals, np.array(axesHere.axis())[None, :], axis=0)
#                     axesStart.append(axesHere)
#                     pltNum += 1
                    
#                     if len(axesStart) + len(axesEnd) <rowsPlt*colsPlt:
#                         axesHere = figSep.add_subplot(rowsPlt, colsPlt, pltNum)
#                         if pltNum <= colsPlt:
#                             axesHere.set_title("End")
#                     else:
#                         axesHere = axesEnd[int(pltNum/2-1)]
#                         plt.axes(axesHere)
                        
#             #        endTm = groupedBalancedEndDelays[0][trlInd]
#                     # endTm = 0 # we align to the end now...
#                     plt.plot(tmValsEnd, dim[-tmValsEnd.shape[0]:], color=colorset[trialsPresented[trlInd],:], linewidth=0.4)
#             #        endInd = endTm/binSizeMs
#             #        endIndFl = int(np.floor(endTm/binSizeMs))
#             #        endIndNxt = endIndFl + 1
#             #        endTmVal = (dim[endIndNxt] - dim[endIndFl])*(endInd-endIndFl) + dim[endIndFl]
#             #        plt.plot(endTm, endTmVal, 'o',color = [1,0,0])
#                     axVals = np.append(axVals, np.array(axesHere.axis())[None, :], axis=0)
#                     axesEnd.append(axesHere)
#                     pltNum += 1
                
#         ymin = np.min(np.append(0, np.min(axVals, axis=0)[2]))
#         ymax = np.max(axVals, axis=0)[3]
#         for ax in axesStart:
#             ax.set_ylim(bottom = ymin, top = ymax )
#             plt.axes(ax)
#             plt.axvline(x=0, linestyle='--')
            
#         for ax in axesEnd:
#             ax.set_ylim(bottom = ymin, top = ymax )
#             plt.axes(ax)
#             plt.axvline(x=0, linestyle='--')
            
        

