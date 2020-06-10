#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 15:07:23 2020

@author: emilio
"""

from methods.GeneralMethods import loadDefaultParams
from matplotlib import pyplot as plt

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
#
#binnedSpikes = genBSLAroundDelay(data, 
#                                    dataIndsProcess,
#                                    stateNamesDelayStart,
#                                    trialType = trialType,
#                                    lenSmallestTrl=lenSmallestTrl, 
#                                    binSizeMs = binSizeMs, 
#                                    furthestTimeBeforeDelay=furthestBack, 
#                                    furthestTimeAfterDelay=furthestForward,
#                                    setStartToDelayEnd = False,
#                                    setEndToDelayStart = True)
#
#
#binnedSpikesAll = genBSLAroundDelay(data, 
#                                    dataIndsProcess,
#                                    stateNamesDelayStart,
#                                    trialType = trialType,
#                                    lenSmallestTrl=lenSmallestTrl, 
#                                    binSizeMs = binSizeMs, 
#                                    furthestTimeBeforeDelay=furthestBack, 
#                                    furthestTimeAfterDelay=furthestForward,
#                                    setStartToDelayEnd = False,
#                                    setEndToDelayStart = False)
#
#binnedSpikesOnlyDelay = genBSLAroundDelay(data, 
#                                    dataIndsProcess,
#                                    stateNamesDelayStart,
#                                    trialType = trialType,
#                                    lenSmallestTrl=lenSmallestTrl, 
#                                    binSizeMs = binSizeMs, 
#                                    furthestTimeBeforeDelay=0, 
#                                    furthestTimeAfterDelay=0,
#                                    setStartToDelayEnd = False,
#                                    setEndToDelayStart = False)
#
#binnedSpikeEnd = genBSLAroundDelay(data, 
#                                    dataIndsProcess,
#                                    stateNamesDelayStart,
#                                    trialType = trialType,
#                                    lenSmallestTrl=lenSmallestTrl, 
#                                    binSizeMs = binSizeMs, 
#                                    furthestTimeBeforeDelay=furthestBack, 
#                                    furthestTimeAfterDelay=furthestForward,
#                                    setStartToDelayEnd = True,
#                                    setEndToDelayStart = False)
#
#binnedSpikesShortStart = genBSLAroundDelay(data, 
#                                    dataIndsProcess,
#                                    stateNamesDelayStart,
#                                    trialType = trialType,
#                                    lenSmallestTrl=lenSmallestTrl, 
#                                    binSizeMs = binSizeMs, 
#                                    furthestTimeBeforeDelay=0, 
#                                    furthestTimeAfterDelay=lenSmallestTrl,
#                                    setStartToDelayEnd = False,
#                                    setEndToDelayStart = True)
#
#binnedSpikesShortEnd = genBSLAroundDelay(data, 
#                                    dataIndsProcess,
#                                    stateNamesDelayStart,
#                                    trialType = trialType,
#                                    lenSmallestTrl=lenSmallestTrl, 
#                                    binSizeMs = binSizeMs, 
#                                    furthestTimeBeforeDelay=lenSmallestTrl, 
#                                    furthestTimeAfterDelay=0,
#                                    setStartToDelayEnd = True,
#                                    setEndToDelayStart = False)

# NOTE: this one is special because it returns *residuals*
offshift = 75 #ms
firingRateThresh = 1
fanoFactorThresh = 4
baselineSubtract = True
binnedResidualsShortStartOffshift, chFanosResidualsShortStartOffshift = genBSLAroundDelay(data, 
                                    dataIndsProcess,
                                    stateNamesDelayStart,
                                    trialType = trialType,
                                    lenSmallestTrl=lenSmallestTrl, 
                                    binSizeMs = binSizeMs, 
                                    furthestTimeBeforeDelay=-offshift, # note that this starts it *forwards* from the delay
                                    furthestTimeAfterDelay=lenSmallestTrl+offshift,
                                    setStartToDelayEnd = False,
                                    setEndToDelayStart = True,
                                    returnResiduals = baselineSubtract,
                                    removeBadChannels = True,
                                    unitsOut = 'count', # this shouldn't affect GPFA... but will affect fano factor cutoffs...
                                    firingRateThresh = firingRateThresh,
                                    fanoFactorThresh = fanoFactorThresh # suggestion of an okay value (not too conservative as with 8, not too lenient as with 1)
                                    )
baselineSubtract = False
furthestTimeBeforeDelay=-offshift # note that this starts it *forwards* from the delay
furthestTimeAfterDelay=lenSmallestTrl+offshift
binnedSpikesShortStartOffshift, _ = genBSLAroundDelay(data, 
                                    dataIndsProcess,
                                    stateNamesDelayStart,
                                    trialType = trialType,
                                    lenSmallestTrl=lenSmallestTrl, 
                                    binSizeMs = binSizeMs, 
                                    furthestTimeBeforeDelay=-offshift, # note that this starts it *forwards* from the delay
                                    furthestTimeAfterDelay=lenSmallestTrl+offshift,
                                    setStartToDelayEnd = False,
                                    setEndToDelayStart = True,
                                    returnResiduals = baselineSubtract,
                                    removeBadChannels = True,
                                    unitsOut = 'count', # this shouldn't affect GPFA... but will affect fano factor cutoffs...
                                    firingRateThresh = firingRateThresh,
                                    fanoFactorThresh = fanoFactorThresh # suggestion of an okay value (not too conservative as with 8, not too lenient as with 1)
                                    )

from methods.BinnedSpikeSetListMethods import subsampleBinnedSpikeSetsToMatchNeuronsAndTrialsPerCondition as subsmpMatchCond
numSubsamples = 1
binnedResidShStOffSubsamples, trlNeurUseSubsamples, minNumTrlPerCond, minNumNeur = subsmpMatchCond(binnedResidualsShortStartOffshift, numSubsamples = numSubsamples)

binnedSpksShortStOffSubsamplesFR = []
binnedSpksShortStOffSubsamplesCnt = []
for areaTrlNeurUse, spkCnts in zip(trlNeurUseSubsamples, binnedSpikesShortStartOffshift):
    binnedSpksShortStOffSubsamplesHereFR = []
    binnedSpksShortStOffSubsamplesHereCnt = []
    for subsmpTrlNeurUse in areaTrlNeurUse:
        valsUse = spkCnts[subsmpTrlNeurUse[0]][:, subsmpTrlNeurUse[1]]
        binnedSpksShortStOffSubsamplesHereFR.append(valsUse.convertUnitsTo('Hz'))
        binnedSpksShortStOffSubsamplesHereCnt.append(valsUse.convertUnitsTo('count'))

    binnedSpksShortStOffSubsamplesFR.append(binnedSpksShortStOffSubsamplesHereFR)
    binnedSpksShortStOffSubsamplesCnt.append(binnedSpksShortStOffSubsamplesHereCnt)

#%% some descriptive data plots
from methods.BinnedSpikeSetListMethods import generateLabelGroupStatistics as genBSLLabGrp
from methods.BinnedSpikeSetListMethods import plotExampleChannelResponses
from methods.BinnedSpikeSetListMethods import plotStimDistributionHistograms
from methods.BinnedSpikeSetListMethods import plotFiringRates

descriptions = [data[idx]['description'] for idx in dataIndsProcess]

listBSS = binnedSpksShortStOffSubsamplesFR
plotStimDistributionHistograms(listBSS, descriptions)
plotFiringRates(listBSS, descriptions, supTitle = 'Delay Start Offshift Firing Rates', cumulative = False)

listBSS = binnedSpksShortStOffSubsamplesCnt
plotStimDistributionHistograms(listBSS, descriptions)
plotFiringRates(listBSS, descriptions, supTitle = 'Delay Start Offshift Spike Counts', cumulative = False)


saveFiguresToPdf(pdfname="genericMetrics")
plt.close('all')

#breakpoint()
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
# now save all the figures
if plotOutput:
    from methods.GeneralMethods import saveFiguresToPdf

    saveFiguresToPdf(pdfname=signalDescriptor)
    plt.close('all')

#%% Residual correlations
from scipy.stats import binned_statistic
avgSpkCorrOverall = []
residCorrPerCondAll = []
geoMeanOverall = []
plotResid = True
for idxSpks, bnSp in enumerate(listBSS):
    idx = dataIndsProcess[idxSpks]
    if plotResid:
        scatterFig = plt.figure()
    separateNoiseCorrForLabels = True
    normalize = True
    residualSpikes, avgSpkCorrAll, residCorrPerCond, geoMeanCntAll = bnSp.channelsAboveThresholdFiringRate(firingRateThresh=firingRateThresh)[0].residualCorrelations(bnSp.labels['stimulusMainLabel'], plot=plotResid, separateNoiseCorrForLabels = separateNoiseCorrForLabels, figTitle = data[idx]['description'], normalize = normalize)
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

    residCorrPerCond = residCorrPerCond[upperTriInds]
    residCorrPerCondAll.append(residCorrPerCond)
    
    if plotResid:
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

breakpoint()

    
if plotResid:
    from methods.GeneralMethods import saveFiguresToPdf

    saveFiguresToPdf(pdfname="residualsZsc3RemFR2")
    plt.close('all')

mnCorrPerCond = [residCorr.mean(axis=0) for residCorr in residCorrPerCondAll]
plt.figure()
plt.title('r_{sc} vs sh pop cov')
[plt.scatter(mnCC, shCPP, label=desc) for mnCC, shCPP, desc in zip(mnCorrPerCond, shCovPropPopByArea, descriptions)]
plt.xlabel('mean r_{sc}')
plt.ylabel('shared population covariance')
plt.legend()

plt.figure()
plt.title('r_{sc} vs sh cov neur avg')
[plt.scatter(mnCC, shCPN, label=desc) for mnCC, shCPN, desc in zip(mnCorrPerCond, shCovPropNeurAvgByArea, descriptions)]
plt.xlabel('mean r_{sc}')
plt.ylabel('shared covariance neuron average')
plt.legend()

plt.figure()
plt.title('r_{sc} vs log det (gen cov) pop')
[plt.scatter(mnCC, shGCP, label=desc) for mnCC, shGCP, desc in zip(mnCorrPerCond, shLogDetGenCovPropPopByArea, descriptions)]
plt.xlabel('mean r_{sc}')
plt.ylabel('general covariance (log det) population')
plt.legend()

plt.figure()
plt.title('r_{sc} vs dimensionality')
[plt.scatter(mnCC, nD, label=desc) for mnCC, nD, desc in zip(mnCorrPerCond, dims, descriptions)]
plt.xlabel('mean r_{sc}')
plt.ylabel('num dims')
plt.legend()

plt.figure()
plt.title('shared pop covar vs dimensionality')
[plt.scatter(shCPP, nD, label=desc) for shCPP, nD, desc in zip(shCovPropPopByArea, dims, descriptions)]
plt.xlabel('shared population covariance')
plt.ylabel('num dims')
plt.legend()

plt.figure()
plt.title('shared pop covar vs shared covar neur avg')
[plt.scatter(shCPP, shCPN, label=desc) for shCPP, shCPN, desc in zip(shCovPropPopByArea, shCovPropNeurAvgByArea, descriptions)]
plt.xlabel('shared population covariance')
plt.ylabel('shared covariance neuron average')
plt.legend()

from methods.GeneralMethods import saveFiguresToPdf
saveFiguresToPdf(pdfname='scatterMetricsZsc3RemFR2')
## now some more figures
#from matplotlib import pyplot as plt
#plt.close('all')
#if numStimulusConditions is None:
#    numConds = len(np.unique(listBSS[0].labels['stimulusMainLabel'], axis=0))
#else:
#    numConds = numStimulusConditions
#testingArea = 0
#for condUse in range(numConds):
#    for cValUse in range(crossvalidateNumFolds):
#        xDimBest = np.array(dims[testingArea][condUse])
#        xDimsTest = np.array(list(gpfaDimOut[testingArea][condUse].keys()))
#        # do some loops to get the key for the best xdim (given that it'll be loewr than the returned xDimBest)
#        xDimScoreBestIndPos = np.argmin((xDimTest-xDimBest)[(xDimsTest - xDimBest) >= 0])
#        xDimScoreBest = xDimsTest[(xDimsTest-xDimBest)>=0][xDimScoreBestIndPos]
#        
#        # sequences are not necessarily in time sorted order, so find correct sorting
#        timeSortInds = np.argsort(gpfaTrainInds[testingArea][condUse][cValUse])
#        
#        seqTrainNewAll = np.asarray(gpfaDimOut[testingArea][condUse][xDimScoreBest]['seqsTrainNew'][cValUse])
#        seqTrainNewAllSort = seqTrainNewAll[timeSortInds]
#        seqTrainOrthAll = [sq['xorth'] for sq in seqTrainNewAllSort]
#        seqTrainConcat = np.concatenate(seqTrainOrthAll, axis = 1)
#        plt.figure()
#        plt.plot(seqTrainConcat.T, alpha=0.65)
#        plt.title("GPFA trajectories over all trials in cond %d, cval %d" % (condUse, cValUse))
#        plt.xlabel("trial bins across all trials concatenated")
#        plt.ylabel("GPFA projection")
#
#saveFiguresToPdf(pdfname="gpfaAll" + signalDescriptor + "M1")
