#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 15:07:23 2020

@author: emilio
"""

from methods.GeneralMethods import loadDefaultParams
from methods.GeneralMethods import saveFiguresToPdf
from matplotlib import pyplot as plt

from classes.Dataset import Dataset
from setup.DataJointSetup import DatasetInfo, DatasetGeneralLoadParams
import datajoint as dj


from pathlib import Path

import numpy as np
import re
import dill as pickle # because this is what people seem to do?

defaultParams = loadDefaultParams(defParamBase = ".")
dataPath = defaultParams['dataPath']

data = []
data.append({'description': 'Earl 2019-03-18 M1 - MGR',
              'area' : 'M1',
              'path': Path('memoryGuidedReach/Earl/2019/03/18/'),
              'delayStartStateName': 'Delay Period',
              'alignmentStates': [],
              'processor': 'Erinn'});
data.append({'description': 'Earl 2019-03-22 M1 - MGR',
              'area' : 'M1',
              'path': Path('memoryGuidedReach/Earl/2019/03/22/'),
              'delayStartStateName': 'Delay Period',
              'alignmentStates': [],
              'processor': 'Erinn'});
data.append({'description': 'Pepe A1 2018-07-14 PFC - MGS',
              'area' : 'PFC',
              'path': Path('memoryGuidedSaccade/Pepe/2018/07/14/Array1_PFC/'),
              'delayStartStateName': 'TARG_OFF',
              'alignmentStates': ['SOUND_CHANGE','ALIGN'],
              'processor': 'Emilio'});
data.append({'description': 'Pepe A2 2018-07-14 PFC - MGS',
              'area' : 'PFC',
              'path': Path('memoryGuidedSaccade/Pepe/2018/07/14/Array2_PFC/'),
              'delayStartStateName': 'TARG_OFF',
              'alignmentStates': ['SOUND_CHANGE','ALIGN'],
              'processor': 'Emilio'});
data.append({'description': 'Wakko A1 2018-02-11 PFC - MGS',
              'area' : 'PFC',
              'path': Path('memoryGuidedSaccade/Wakko/2018/02/11/Array1_PFC/'),
              'delayStartStateName': 'TARG_OFF',
              'alignmentStates': ['SOUND_CHANGE','ALIGN'],
              'processor': 'Emilio'});
data.append({'description': 'Wakko A2 2018-02-11 PFC - MGS',
              'area' : 'PFC',
              'path': Path('memoryGuidedSaccade/Wakko/2018/02/11/Array2_PFC/'),
              'delayStartStateName': 'TARG_OFF',
              'alignmentStates': ['SOUND_CHANGE','ALIGN'],
              'processor': 'Emilio'});
data.append({'description': 'Pepe 2016-02-02 V4 - cuedAttn',
              'area' : 'V4',
              'path': Path('cuedAttention/Pepe/2016/02/02/Array1_V4/'),
              'delayStartStateName': 'Blank Before',
              'alignmentStates': ['SOUND_CHANGE','ALIGN'],
              'processor': 'Emilio'});
data.append({'description': 'Pepe 2016-02-02 PFC - cuedAttn',
              'area' : 'PFC',
              'path': Path('cuedAttention/Pepe/2016/02/02/Array2_PFC/'),
              'delayStartStateName': 'Blank Before',
              'alignmentStates': ['SOUND_CHANGE','ALIGN'],
              'processor': 'Emilio'});

data = np.asarray(data) # to allow fancy indexing
#%% process data
processedDataMat = 'processedData.mat'
datasetDill = 'dataset.dill'

# dataset = [ [] for _ in range(len(data))]
dataUseLogical = np.zeros(len(data), dtype='bool')
dataIndsProcess = np.array([1,6])#np.array([1,4,6])
dataUseLogical[dataIndsProcess] = True

removeCoincidentSpikes = True
coincidenceTime = 1 #ms
coincidenceThresh = 0.2 # 20% of spikes
checkNumTrls = 0.1 # use 10% of trials
datasetGeneralLoadParams = {
    'remove_coincident_spikes' : removeCoincidentSpikes,
    'coincidence_time' : coincidenceTime, 
    'coincidence_thresh' : coincidenceThresh, 
    'coincidence_fraction_trial_test' : checkNumTrls
}
dsgl = DatasetGeneralLoadParams()
if len(dsgl & datasetGeneralLoadParams)>1:
    raise Exception('multiple copies of the same parameters have been saved... I thought I avoided that')
elif len(dsgl & datasetGeneralLoadParams)>0:
    genParamId = (dsgl & datasetGeneralLoadParams).fetch1('ds_gen_params_id')
else:
    # tacky, but it doesn't return the new id, syoo...
    currIds = dsgl.fetch('ds_gen_params_id')
    DatasetGeneralLoadParams().insert1(datasetGeneralLoadParams)
    newIds = dsgl.fetch('ds_gen_params_id')
    genParamId = list(set(newIds) - set(currIds))[0]

for dataUse in data[dataUseLogical]:
# for ind, dataUse in enumerate(data):
#    dataUse = data[1]
    # dataUse = data[ind]
    dataMatPath = dataPath / dataUse['path'] / processedDataMat

    dataDillPath = dataPath / dataUse['path'] / datasetDill
    
    if dataUse['processor'] == 'Erinn':
        notChan = np.array([31, 0])
    else:
        notChan = np.array([])

    if dataDillPath.exists():
        print('loading dataset ' + dataUse['description'])
        with dataDillPath.open(mode='rb') as datasetDillFh:
            datasetHere = pickle.load(datasetDillFh)
    else:
        print('processing data set ' + dataUse['description'])
        datasetHere = Dataset(dataMatPath, dataUse['processor'], notChan = notChan, removeCoincidentSpikes=removeCoincidentSpikes, coincidenceTime=coincidenceTime, coincidenceThresh=coincidenceThresh, checkNumTrls=checkNumTrls)
        with dataDillPath.open(mode='wb') as datasetDillFh:
            pickle.dump(datasetHere, datasetDillFh)

    datasetHash = dj.hash.uuid_from_file(dataDillPath).hex
    # do some database insertions here
    datasetHereInfo = {
        'dataset_relative_path' : str(dataUse['path'] / datasetDill),
        'dataset_hash' : datasetHash,
        'dataset_name' : dataUse['description'],
        'ds_gen_params_id' : genParamId,
        'processor_name' : dataUse['processor'],
        'brain_area' : dataUse['area'],
        'date_acquired' : re.search('.*?(\d+-\d+-\d+).*', dataUse['description']).group(1)
    }

    dsi = DatasetInfo()
    breakpoint()
    if len(dsi & datasetHereInfo) > 1:
        raise Exception('multiple copies of same dataset in the table...')
    elif len(dsi & datasetHereInfo) > 0:
        dsId = (dsi & datasetHereInfo).fetch1('dataset_id')
    else:
        dsId = len(dsi) + 1
        datasetHereInfo['dataset_id'] = dsId
        dsi.insert1(datasetHereInfo)

    datasetSpecificLoadParams = {
        'dataset_relative_path' : str(dataUse['path'] / datasetDill),
        'ds_gen_params_id' : genParamId,
        'ignore_channels' : notChan
    }
    dsi.DatasetSpecificLoadParams.insert1(datasetSpecificLoadParams)
    breakpoint()
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
# ** inputs for delay offshifted residuals ** #
#listBSS = binnedResidualsShortStartOffshift.copy()
#firingRateThresh=-1
#timeBeforeAndAfterStart = (0+offshift, furthestForward+offshift)
#timeBeforeAndAfterEnd = None
#baselineSubtract = True
#signalDescriptor = "fst%dMsDelShft%dMsFR%0.2f%s%sPreproc" % (furthestForward, offshift,firingRateThresh, "Bsub" if baselineSubtract else "", "Sqrt" if sqrtSpikes else "")
# ** inputs for beginning and end ** #
#listBSS = binnedSpikesAll
#listBSS = binnedSpikesAll.copy()
#allTrue = np.ones(listBSS[0].shape[1], dtype='bool')
##allTrue[np.array([70,72,12,10,76])] = 0 # get rid of possibly bad channels? 12, 10, 76? 38? 39? 45? 53? 81?
##listBSS[0] = listBSS[0][:,allTrue]
#timeBeforeAndAfterStart = (-furthestBack, furthestForward)
#timeBeforeAndAfterEnd = (-furthestBack, furthestForward)
#baselineSubtract = False
#signalDescriptor = "delayStart%d-%dMsdelayEnd%d-%dMsFR%0.2f%sSR" % (furthestBack, furthestForward, furthestBack, furthestForward,firingRateThresh, "Bsub" if baselineSubtract else "")
# ** inputs for delay offshifted residuals matched for trials/channel-neurons ** #
listBSS = binnedResidShStOffSubsamples
baselineSubtract = True # this is how we get the residuals
plotOutput = True # really, I don't want to plot all these figures...
timeBeforeAndAfterStart = (0+offshift, furthestForward+offshift)
timeBeforeAndAfterEnd = None
signalDescriptor = "fst%dMsDelShft%dMsFR%0.2f%s%sTrlCnd%dNrn%dFnoThresh" % (furthestForward, offshift,firingRateThresh, "Bsub" if baselineSubtract else "", "Sqrt" if sqrtSpikes else "", minNumTrlPerCond, minNumNeur)
# note that with the firingRateThresh, it needs to be -infinity to be lower
# than any mean the new subset might have. Why? Because the average firing rate
# for baseline subtracted channels is zero if *ALL* of them are averaged--since
# we're taking just subsets, it's possible that they might be arbitrarily low.
# Moreover, what does a firing rate of a baseline subtracted trace really mean?
firingRateThresh=-np.inf # we've already thresholded (but change it after the name so that's reflected)
baselineSubtract = False # we've already thresholded (but change it after the name so that's reflected)

crossvalidateNumFolds = 4


dims, dimsLL, gpfaDimOut, gpfaTestInds, gpfaTrainInds = gpfaComputation(
    listBSS, descriptions, paths, signalDescriptor = signalDescriptor,
                # [listBSS[-1]], [descriptions[-1]], [paths[-1]],
                          timeBeforeAndAfterStart = timeBeforeAndAfterStart, timeBeforeAndAfterEnd = timeBeforeAndAfterEnd,
                          balanceDirs = True, numStimulusConditions = numStimulusConditions, combineConditions=combineConditions, baselineSubtract = baselineSubtract, sqrtSpikes = False,
                          crossvalidateNumFolds = crossvalidateNumFolds, xDimTest = xDimTest, firingRateThresh=firingRateThresh, plotOutput = plotOutput)

# for the moment all we want to save is this...
# now save all the figures
if plotOutput:

    saveFiguresToPdf(pdfname=signalDescriptor)
    plt.close('all')

#breakpoint()
    
shCovPropPopByArea = []
shCovPropNeurAvgByArea = []
shLogDetGenCovPropPopByArea = []
for gpfaArea, dimsUse, dimsGpfaUse in zip(gpfaDimOut, dims, dimsLL):
    if type(gpfaArea) is list:
        shCovPropPopByAreaHere = []
        shCovPropNeurAvgByAreaHere = []
        shLogDetGenCovPropPopByAreaHere = []
        for (gpA, dU, dGfU) in zip(gpfaArea, dimsUse, dimsGpfaUse):
            CR = [(gpfaCond[int(gpfaDimParamsUse)]['allEstParams'][0]['C'][:,:numDimsUse],gpfaCond[int(gpfaDimParamsUse)]['allEstParams'][0]['R']) for gpfaCond, numDimsUse, gpfaDimParamsUse in zip(gpA, dU, dGfU)]
            shCovPropPop = [np.trace(C @ C.T) / (np.trace(C @ C.T) + np.trace(R)) for C, R in CR] 
            shCovPropNeurAvg = [np.mean(np.diag(C @ C.T) / (np.diag(C @ C.T) + np.diag(R))) for C, R in CR] 
            shLogDetGenCovPropPop = [np.linalg.slogdet(C.T @ C)[1] / (np.linalg.slogdet(C.T @ C)[1] + np.linalg.slogdet(R)[1]) for C, R in CR]
            shCovPropPopByAreaHere.append(np.array(shCovPropPop))
            shCovPropNeurAvgByAreaHere.append(np.array(shCovPropNeurAvg))
            shLogDetGenCovPropPopByAreaHere.append(np.array(shLogDetGenCovPropPop))
        
        shCovPropPopByArea.append(shCovPropPopByAreaHere)
        shCovPropNeurAvgByArea.append(shCovPropNeurAvgByAreaHere)
        shLogDetGenCovPropPopByArea.append(shLogDetGenCovPropPopByAreaHere)
    else:
        CR = [(gpfaCond[int(gpfaDimParamsUse)]['allEstParams'][0]['C'][:,:numDimsUse],gpfaCond[int(gpfaDimParamsUse)]['allEstParams'][0]['R']) for gpfaCond, numDimsUse, gpfaDimParamsUse in zip(gpfaArea, dimsUse, dimsGpfaUse)]
        shCovPropPop = [np.trace(C @ C.T) / (np.trace(C @ C.T) + np.trace(R)) for C, R in CR] 
        shCovPropNeurAvg = [np.mean(np.diag(C @ C.T) / (np.diag(C @ C.T) + np.diag(R))) for C, R in CR] 
        shLogDetGenCovPropPop = [np.linalg.slogdet(C.T @ C)[1] / (np.linalg.slogdet(C.T @ C)[1] + np.linalg.slogdet(R)[1]) for C, R in CR]
        
        shCovPropPopByArea.append(np.array(shCovPropPop))
        shCovPropNeurAvgByArea.append(np.array(shCovPropNeurAvg))
        shLogDetGenCovPropPopByArea.append(np.array(shLogDetGenCovPropPop))


#%% Residual correlations
from scipy.stats import binned_statistic
residCorrMeanOverCondAll = []
residCorrPerCondAll = []
geoMeanOverall = []

residCorrPerCondOBPTAll = []
residCorrMeanOverCondOBPTAll = []
plotResid = True
for idxSpks, bnSp in enumerate(listBSS):
    idx = dataIndsProcess[idxSpks]
    separateNoiseCorrForLabels = True
    normalize = False #True
    if plotResid:
        scatterFig = plt.figure()
        if type(bnSp) is list:
            scatterFig.suptitle(data[idx]['description'] + " geo mean FR vs corr var multiple subsets")
        else:
            scatterFig.suptitle(data[idx]['description'] + " geo mean FR vs corr var")
    if type(bnSp) is list:
        residCorrMeanOverCondAllHere = []
        residCorrPerCondAllHere = []
        geoMeanOverallHere = []
        
        residCorrPerCondOBPTAllHere = []
        residCorrMeanOverCondOBPTAllHere = []
        for idxSubset, bS in enumerate(bnSp):
            trialLengthMs = furthestTimeBeforeDelay + furthestTimeAfterDelay - 1
            bSOneBinPrTrl = bS.convertUnitsTo('count').increaseBinSize(trialLengthMs)
            residualSpikes, residCorrMeanOverCond, residCorrPerCond, geoMeanCntAll = bS.convertUnitsTo('count').residualCorrelations(bS.labels['stimulusMainLabel'], plot=plotResid, separateNoiseCorrForLabels = separateNoiseCorrForLabels, figTitle = data[idx]['description'] + ("subset %d" % idxSubset), normalize = normalize)
            residualSpikesOBPT, residCorrMeanOverCondOBPT, residCorrPerCondOBPT, geoMeanCntAllOBPT = bSOneBinPrTrl.residualCorrelations(bS.labels['stimulusMainLabel'], plot=plotResid, separateNoiseCorrForLabels = separateNoiseCorrForLabels, figTitle = data[idx]['description'] + ("1 bin/trl subset %d" % idxSubset), normalize = normalize)
            
            
            residCorrMeanOverCondAllHere.append(residCorrMeanOverCond)
            geoMeanOverallHere.append(geoMeanCntAll)
            residCorrPerCondAllHere.append(residCorrPerCond)
            
            residCorrMeanOverCondOBPTAllHere.append(residCorrMeanOverCondOBPT)
            residCorrPerCondOBPTAllHere.append(residCorrPerCondOBPT)
            
            
            
            # Only care about single repeats of the non-self pairs...
            upperTriInds = np.triu_indices(residCorrMeanOverCond.shape[0], 1)
            residCorrMeanOverCond = residCorrMeanOverCond[upperTriInds]
            geoMeanCnt = geoMeanCntAll[upperTriInds]
            residCorrPerCond = residCorrPerCond[upperTriInds]
            
            residCorrPerCondOBPT = residCorrPerCondOBPT[upperTriInds]
            
            
            if False:#plotResid:
                axs = scatterFig.subplots(2,1)
                axs[0].scatter(geoMeanCnt.flatten(), residCorrMeanOverCond.flatten())
                axs[0].set_ylabel("Correlated Variability")
                
                binWidth = 5
                binAvg, binEdges, bN = binned_statistic(geoMeanCnt.flatten(), residCorrMeanOverCond.flatten(), statistic='mean', bins = np.arange(np.max(geoMeanCnt), step=binWidth))
                binStd, _, _ = binned_statistic(geoMeanCnt.flatten(), residCorrMeanOverCond.flatten(), statistic='std', bins = np.arange(np.max(geoMeanCnt), step=binWidth))
                binEdges = np.diff(binEdges)[0]/2+binEdges[:-1]
                axs[1].plot(binEdges, binAvg)
                axs[1].fill_between(binEdges, binAvg-binStd, binAvg+binStd, alpha=0.2)
                axs[1].set_xlabel("Geometric Mean Firing Rate")
                axs[1].set_ylabel("Correlated Variability")
                
        residCorrMeanOverCondAll.append(residCorrMeanOverCondAllHere)
        geoMeanOverall.append(geoMeanOverallHere)
        residCorrPerCondAll.append(residCorrPerCondAllHere)
        
        residCorrPerCondOBPTAll.append(residCorrPerCondOBPTAllHere)
        residCorrMeanOverCondOBPTAll.append(residCorrMeanOverCondOBPTAllHere)
    else:
        residualSpikes, residCorrMeanOverCond, residCorrPerCond, geoMeanCntAll = bS.residualCorrelations(bS.labels['stimulusMainLabel'], plot=plotResid, separateNoiseCorrForLabels = separateNoiseCorrForLabels, figTitle = data[idx]['description'], normalize = normalize)
        
        
        residCorrMeanOverCondAll.append(residCorrMeanOverCond)
        geoMeanOverall.append(geoMeanCntAll)
        residCorrPerCondAll.append(residCorrPerCond)
        
        # Only care about single repeats of the non-self pairs...
        upperTriInds = np.triu_indices(residCorrMeanOverCond.shape[0], 1)
        residCorrMeanOverCond = residCorrMeanOverCond[upperTriInds]
        geoMeanCnt = geoMeanCntAll[upperTriInds]
        residCorrPerCond = residCorrPerCond[upperTriInds]
        
        
        if plotResid:
            axs = scatterFig.subplots(2,1)
            axs[0].scatter(geoMeanCnt.flatten(), residCorrMeanOverCond.flatten())
            axs[0].set_ylabel("Correlated Variability")
            
            binWidth = 5
            binAvg, binEdges, bN = binned_statistic(geoMeanCnt.flatten(), residCorrMeanOverCond.flatten(), statistic='mean', bins = np.arange(np.max(geoMeanCnt), step=binWidth))
            binStd, _, _ = binned_statistic(geoMeanCnt.flatten(), residCorrMeanOverCond.flatten(), statistic='std', bins = np.arange(np.max(geoMeanCnt), step=binWidth))
            binEdges = np.diff(binEdges)[0]/2+binEdges[:-1]
            axs[1].plot(binEdges, binAvg)
            axs[1].fill_between(binEdges, binAvg-binStd, binAvg+binStd, alpha=0.2)
            axs[1].set_xlabel("Geometric Mean Firing Rate")
            axs[1].set_ylabel("Correlated Variability")


    
if plotResid:
    from methods.GeneralMethods import saveFiguresToPdf
    
    saveFiguresToPdf(pdfname="residualsOverSubsets")
    plt.close('all')

breakpoint()


if type(residCorrPerCondAll[0]) is list:
    mnCorrPerCond = []
    stdCorrPerCond = []
    for rsCorr in residCorrPerCondAll:
        mnCorrPerCond.append([residCorr.mean(axis=(0,1)) for residCorr in rsCorr])
        stdCorrPerCond.append([residCorr.std(axis=(0,1)) for residCorr in rsCorr])
    
    mnCorrPerCondOBPT = []
    stdCorrPerCondOBPT = []
    for rsCorr in residCorrPerCondOBPTAll:
        mnCorrPerCondOBPT.append([residCorr.mean(axis=(0,1)) for residCorr in rsCorr])
        stdCorrPerCondOBPT.append([residCorr.std(axis=(0,1)) for residCorr in rsCorr])
    
    fRChMnByArea = []
    fRChStdByArea = []
    fanoFactorChMnByArea = []
    fanoFactorChStdByArea = []
    labelUse = 'stimulusMainLabel'
    for bnSpCnt in binnedSpksShortStOffSubsamplesCnt:
        fRChMnByAreaHere = []
        fRChStdByAreaHere = []
        fanoFactorChMnByAreaHere = []
        fanoFactorChStdByAreaHere = []
        for bSC in bnSpCnt:
            grpSpkCnt, uniqueLabel = bSC.groupByLabel(bSC.labels[labelUse])
            fRChMnByAreaHere.append([np.mean(gSC.avgFiringRateByChannel()) for gSC in grpSpkCnt])
            fRChStdByAreaHere.append([np.std(gSC.avgFiringRateByChannel()) for gSC in grpSpkCnt])
            fanoFactorChMnByAreaHere.append([np.mean(gSC.fanoFactorByChannel()) for gSC in grpSpkCnt])
            fanoFactorChStdByAreaHere.append([np.std(bSC.fanoFactorByChannel()) for gSC in grpSpkCnt])
        fRChMnByArea.append(np.stack(fRChMnByAreaHere))
        fRChStdByArea.append(np.stack(fRChStdByAreaHere))
        fanoFactorChMnByArea.append(np.stack(fanoFactorChMnByAreaHere))
        fanoFactorChStdByArea.append(np.stack(fanoFactorChStdByAreaHere))
    
    mnCorrPerCond = [np.hstack(mn) for mn in mnCorrPerCond]
    shCovPropPopByAreaAllSub = [np.hstack(shProp) for shProp in shCovPropPopByArea]
    shCovPropNeurAvgByAreaAllSub = [np.hstack(shPropN) for shPropN in shCovPropNeurAvgByArea]
    shLogDetGenCovPropPopByAreaAllSub = [np.hstack(shLD) for shLD in shLogDetGenCovPropPopByArea]
    dimsByArea = [np.hstack(dm) for dm in dims]
else:
    # TODO fill this in with the rest of the values defined above...
    mnCorrPerCond = [residCorr.mean(axis=0) for residCorr in residCorrPerCondAll]
    stdCorrPerCond = [residCorr.std(axis=0) for residCorr in residCorrPerCondAll]
    dimsByArea = dims

def pltAllVsAll(descriptions, metricDict):
    for metricNum, (metricName, metricVal) in enumerate(metricDict.items()):
        for metric2Num, (metric2Name, metric2Val) in enumerate(metricDict.items()):
            if metric2Num > metricNum:
                plt.figure()
                plt.title('%s vs %s' % ( metricName, metric2Name ))
                [plt.scatter(m1, m2, label=desc) for m1, m2, desc in zip(metricVal, metric2Val, descriptions)]
                plt.xlabel(metricName)
                plt.ylabel(metric2Name)
                plt.legend()

metricDict = { 
    'mean channel firing rate (Hz)' : fRChMnByArea,
    'std channel firing rate (Hz)' : fRChStdByArea,
    'mean channel fano factor' : fanoFactorChMnByArea,
    'std channel fano factor' : fanoFactorChStdByArea,
    'mean(r_{sc})' : mnCorrPerCond,
    'std(r_{sc})' : stdCorrPerCond,
    'mean(r_{sc} 1 bn/tr)' : mnCorrPerCondOBPT,
    'std(r_{sc} 1 bn/tr)' : stdCorrPerCondOBPT,
    'sh pop cov' : shCovPropPopByAreaAllSub,
    'sh cov neur avg' : shCovPropNeurAvgByAreaAllSub,
    'dimensionality' : dimsByArea,
    'gen cov (logdet) pop' : shLogDetGenCovPropPopByAreaAllSub,
}
pltAllVsAll(descriptions, metricDict)
#
#plt.figure()
#plt.title('r_{sc} vs std(r_{sc})')
#[plt.scatter(mnCC, stdCC, label=desc) for mnCC, stdCC, desc in zip(mnCorrPerCond, stdCorrPerCond, descriptions)]
#plt.xlabel('mean r_{sc}')
#plt.ylabel('std r_{sc}')
#plt.legend()
#
#plt.figure()
#plt.title('r_{sc} vs sh pop cov')
#[plt.scatter(mnCC, shCPP, label=desc) for mnCC, shCPP, desc in zip(mnCorrPerCond, shCovPropPopByAreaAllSub, descriptions)]
#plt.xlabel('mean r_{sc}')
#plt.ylabel('shared population covariance')
#plt.legend()
#
#plt.figure()
#plt.title('r_{sc} vs sh cov neur avg')
#[plt.scatter(mnCC, shCPN, label=desc) for mnCC, shCPN, desc in zip(mnCorrPerCond, shCovPropNeurAvgByAreaAllSub, descriptions)]
#plt.xlabel('mean r_{sc}')
#plt.ylabel('shared covariance neuron average')
#plt.legend()
#
#plt.figure()
#plt.title('r_{sc} vs log det (gen cov) pop')
#[plt.scatter(mnCC, shGCP, label=desc) for mnCC, shGCP, desc in zip(mnCorrPerCond, shLogDetGenCovPropPopByAreaAllSub, descriptions)]
#plt.xlabel('mean r_{sc}')
#plt.ylabel('general covariance (log det) population')
#plt.legend()
#
#plt.figure()
#plt.title('r_{sc} vs dimensionality')
#[plt.scatter(mnCC, nD, label=desc) for mnCC, nD, desc in zip(mnCorrPerCond, dimsByArea, descriptions)]
#plt.xlabel('mean r_{sc}')
#plt.ylabel('num dims')
#plt.legend()
#
#plt.figure()
#plt.title('log det (gen cov) pop vs dimensionality')
#[plt.scatter(shGCP, nD, label=desc) for shGCP, nD, desc in zip(shLogDetGenCovPropPopByAreaAllSub, dimsByArea, descriptions)]
#plt.xlabel('mean r_{sc}')
#plt.ylabel('general covariance (log det) population')
#plt.legend()
#
#plt.figure()
#plt.title('shared pop covar vs dimensionality')
#[plt.scatter(shCPP, nD, label=desc) for shCPP, nD, desc in zip(shCovPropPopByAreaAllSub, dimsByArea, descriptions)]
#plt.xlabel('shared population covariance')
#plt.ylabel('num dims')
#plt.legend()
#
#plt.figure()
#plt.title('shared pop covar vs shared covar neur avg')
#[plt.scatter(shCPP, shCPN, label=desc) for shCPP, shCPN, desc in zip(shCovPropPopByAreaAllSub, shCovPropNeurAvgByAreaAllSub, descriptions)]
#plt.xlabel('shared population covariance')
#plt.ylabel('shared covariance neuron average')
#plt.legend()

saveFiguresToPdf(pdfname='scatterMetricsOverSubsets')

breakpoint()
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
