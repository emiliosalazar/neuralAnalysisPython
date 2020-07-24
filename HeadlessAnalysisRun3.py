
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 15:07:23 2020

@author: emilio
"""

from methods.GeneralMethods import loadDefaultParams
from methods.GeneralMethods import saveFiguresToPdf
from matplotlib import pyplot as plt
# this is necessary to get PDF figures to save with the right font sizes...
# annoying but true
from matplotlib import rc
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 10}
rc('font', **font)

from classes.Dataset import Dataset
from setup.DataJointSetup import DatasetInfo, DatasetGeneralLoadParams, BinnedSpikeSetInfo, BinnedSpikeSetProcessParams, FilterSpikeSetParams
import datajoint as dj


from pathlib import Path

import numpy as np
import re
import dill as pickle # because this is what people seem to do?
import hashlib
import json

defaultParams = loadDefaultParams(defParamBase = ".")
dataPath = defaultParams['dataPath']


removeCoincidentChans = True
coincidenceTime = 1 #ms
coincidenceThresh = 0.2 # 20% of spikes
checkNumTrls = 0.1 # use 10% of trials
datasetGeneralLoadParams = {
    'remove_coincident_chans' : removeCoincidentChans,
    'coincidence_time' : coincidenceTime, 
    'coincidence_thresh' : coincidenceThresh, 
    'coincidence_fraction_trial_test' : checkNumTrls
}
dsgl = DatasetGeneralLoadParams()
dsi = DatasetInfo()
bsi = BinnedSpikeSetInfo()
bsp = BinnedSpikeSetProcessParams()
fsp = FilterSpikeSetParams()

#%% get desired time segment
from classes.BinnedSpikeSet import BinnedSpikeSet
from methods.BinnedSpikeSetListMethods import generateBinnedSpikeListsAroundState as genBSLAroundState
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

#brainAreaOfInterest = 'M1'
#datasets = dsi['brain_area="%s"' % brainAreaOfInterest].grabDatasets()
#
#stateNamesDelayStart = [ds.keyStates['delay'] for ds in datasets]
#
#binnedSpikes, _ = genBSLAroundState(datasets,
#                                    stateNamesDelayStart,
#                                    trialType = trialType,
#                                    lenSmallestTrl=lenSmallestTrl, 
#                                    binSizeMs = binSizeMs, 
#                                    furthestTimeBeforeState=furthestBack, 
#                                    furthestTimeAfterState=furthestForward,
#                                    setStartToDelayEnd = False,
#                                    setEndToDelayStart = True)
#
#
#binnedSpikesAll, _ = genBSLAroundState(datasets,
#                                    stateNamesDelayStart,
#                                    trialType = trialType,
#                                    lenSmallestTrl=lenSmallestTrl, 
#                                    binSizeMs = binSizeMs, 
#                                    furthestTimeBeforeState=furthestBack, 
#                                    furthestTimeAfterState=furthestForward,
#                                    setStartToDelayEnd = False,
#                                    setEndToDelayStart = False)
#
#binnedSpikesOnlyDelay, _ = genBSLAroundState(datasets,
#                                    stateNamesDelayStart,
#                                    trialType = trialType,
#                                    lenSmallestTrl=lenSmallestTrl, 
#                                    binSizeMs = binSizeMs, 
#                                    furthestTimeBeforeState=0, 
#                                    furthestTimeAfterState=0,
#                                    setStartToDelayEnd = False,
#                                    setEndToDelayStart = False)
#
#binnedSpikeEnd, _ = genBSLAroundState(datasets,
#                                    stateNamesDelayStart,
#                                    trialType = trialType,
#                                    lenSmallestTrl=lenSmallestTrl, 
#                                    binSizeMs = binSizeMs, 
#                                    furthestTimeBeforeState=furthestBack, 
#                                    furthestTimeAfterState=furthestForward,
#                                    setStartToDelayEnd = True,
#                                    setEndToDelayStart = False)
#
#binnedSpikesShortStart, _ = genBSLAroundState(datasets,
#                                    stateNamesDelayStart,
#                                    trialType = trialType,
#                                    lenSmallestTrl=lenSmallestTrl, 
#                                    binSizeMs = binSizeMs, 
#                                    furthestTimeBeforeState=0, 
#                                    furthestTimeAfterState=lenSmallestTrl,
#                                    setStartToDelayEnd = False,
#                                    setEndToDelayStart = True)
#
#binnedSpikesShortEnd, _ = genBSLAroundState(datasets,
#                                    stateNamesDelayStart,
#                                    trialType = trialType,
#                                    lenSmallestTrl=lenSmallestTrl, 
#                                    binSizeMs = binSizeMs, 
#                                    furthestTimeBeforeState=lenSmallestTrl, 
#                                    furthestTimeAfterState=0,
#                                    setStartToDelayEnd = True,
#                                    setEndToDelayStart = False)
#
## NOTE: this one is special because it returns *residuals*
#baselineSubtract = True
#binnedResidualsShortStartOffshift, chFanosResidualsShortStartOffshift = genBSLAroundState(
#                                    datasets,
#                                    stateNamesDelayStart,
#                                    trialType = trialType,
#                                    lenSmallestTrl=lenSmallestTrl, 
#                                    binSizeMs = binSizeMs, 
#                                    furthestTimeBeforeState=-offshift, # note that this starts it *forwards* from the delay
#                                    furthestTimeAfterState=lenSmallestTrl+offshift,
#                                    setStartToDelayEnd = False,
#                                    setEndToDelayStart = True,
#                                    returnResiduals = baselineSubtract,
#                                    unitsOut = 'count', # this shouldn't affect GPFA... but will affect fano factor cutoffs...
#                                    firingRateThresh = firingRateThresh,
#                                    fanoFactorThresh = fanoFactorThresh # suggestion of an okay value (not too conservative as with 8, not too lenient as with 1)

print("Computing/loading binned spike sets")
offshift = 75 #ms
firingRateThresh = 1
fanoFactorThresh = 4
baselineSubtract = False
furthestTimeBeforeDelay=-offshift # note that this starts it *forwards* from the delay
furthestTimeAfterDelay=lenSmallestTrl+offshift

singKey = []
singKey.append(dsi['brain_area="M1" AND dataset_name LIKE "%2019-03-22%"'].fetch("KEY")[0]) # M1
#singKey.append(dsi['brain_area="PFC"'].fetch("KEY")[0]) # PFC
#singKey.append(dsi['brain_area="V4"'].fetch("KEY")[0]) # V4
singAreaDs = dsi[singKey]
keyStateName = 'delay'
binnedSpikesShortStartOffshift, bssiKeys = genBSLAroundState(singAreaDs,
                                    keyStateName,
                                    trialType = trialType,
                                    lenSmallestTrl=lenSmallestTrl, 
                                    binSizeMs = binSizeMs, 
                                    furthestTimeBeforeState=-offshift, # note that this starts it *forwards* from the delay
                                    furthestTimeAfterState=lenSmallestTrl+offshift,
                                    setStartToStateEnd = False,
                                    setEndToStateStart = True,
                                    returnResiduals = baselineSubtract,
                                    unitsOut = 'count', # this shouldn't affect GPFA... but will affect fano factor cutoffs...
                                    firingRateThresh = firingRateThresh,
                                    fanoFactorThresh = fanoFactorThresh # suggestion of an okay value (not too conservative as with 8, not too lenient as with 1)
                                    )

print("Subsampling binned spike sets")
# alright some subsampling goodness to do more GPFA
from methods.BinnedSpikeSetListMethods import subsampleBinnedSpikeSetsToMatchNeuronsAndTrialsPerCondition as subsmpMatchCond
numSubsamples = 1

bssExp = bsi[bssiKeys]
# technically already loaded above, but in future we'll want to be able to grab
# the directly from the db call
binnedSpikesList = bssExp.grabBinnedSpikes() 
numValsSweep = 3
minNumNeuron = 30 # np.min([bnSp.shape[1] for bnSp in binnedSpikesList])
labelName = 'stimulusMainLabel'
minNumTrlPerCond = 20

#extraOpts = {
#    dsi[dsgl['task="cuedAttention"']] : {
#        'description' : 'choose first blank',
#        'filter' : (lambda res : res.grabBinnedSpikes()[0].labels['sequencePosition']==2)
#        }
#    }
extraOpts = {
    dsi[dsgl['task="cuedAttention"']] : {
        'description' : 'choose non-choice trials',
        'filter' : (lambda res : res.grabBinnedSpikes()[0].labels['sequencePosition']!=res.grabBinnedSpikes()[0].labels['sequenceLength'])
        }
    }

binnedResidSubsamples = []
subsampleExpressions = []
dsNames = []
brainAreas = []
tasks = []
for bKey in zip(bssiKeys): 
    bsInfo = bsi[bKey]
    bSSHere = bsInfo.grabBinnedSpikes()[0]

    if extraOpts:
        extraFilt = []
        extraDescription = []
        anyMatch = False
        for extraOptExp, filterParams in extraOpts.items():
            match = bsi[extraOptExp][bKey]
            filterLambda = filterParams['filter']
            filterDescription = filterParams['description']
            if len(match):
                anyMatch = True
                extraFilt.append(filterLambda(match))
                extraDescription.append(filterDescription)

        if anyMatch:
            totalFilt = np.concatenate(extraFilt,axis=1)
            totalFilt = np.all(totalFilt,axis=1)
            _, filtBssKeys = bsInfo.filterBSS(filterReason = "other", filterDescription = "; ".join(extraDescription), condLabel = labelName, trialFilter = totalFilt, returnKey = True)
            bSSHere = bsi[filtBssKeys].grabBinnedSpikes()[0]

    maxNumNeuron = bSSHere.shape[1]
    numNeuronTests = np.round(np.geomspace(minNumNeuron, maxNumNeuron, numValsSweep))

    numTrls = np.unique(bSSHere.labels[labelName],axis=0,return_counts=True)[1]
    maxNumTrlPerCond = np.min(numTrls)
    numTrialsPerCondTests = np.round(np.geomspace(minNumTrlPerCond, maxNumTrlPerCond, numValsSweep))

    from itertools import product
    subsampCombos = tuple(product(numNeuronTests, numTrialsPerCondTests, [numSubsamples] if type(numSubsamples) is not list else numSubsamples))

    for combo in subsampCombos:
        numNeur = int(combo[0])
        numTrial = int(combo[1])
        numSubsample = int(combo[2])
        bRSub, subE, dNm, bA, tsk = subsmpMatchCond(bsi[bKey], maxNumTrlPerCond = numTrial, maxNumNeuron = numNeur, labelName = labelName, numSubsamples = numSubsample, extraOpts = extraOpts)

        binnedResidSubsamples += bRSub
        subsampleExpressions += subE
        dsNames += dNm
        brainAreas += bA
        tasks += tsk


combs = np.stack(subsampCombos)
numNeurs = combs[:, 0]
numTrls = combs[:, 1]
#breakpoint()
print("Computing GPFA")
from methods.BinnedSpikeSetListMethods import gpfaComputation


plotOutput = True
# cutting off dim of 15... because honestly reiduals don't seem to even go past 8
xDimTest = [2,5,8,12]#[2,5,8,12,15]
firingRateThresh = 1 if not baselineSubtract else 0
combineConditions = False
numStimulusConditions = None # because V4 has two...
sqrtSpikes = False
crossvalidateNumFolds = 4
computeResiduals = True
balanceConds = True
timeBeforeAndAfterStart = (0+offshift, furthestForward+offshift)
timeBeforeAndAfterEnd = None

dims, dimsLL, gpfaDimOut, gpfaTestInds, gpfaTrainInds = gpfaComputation(
    subsampleExpressions,timeBeforeAndAfterStart = timeBeforeAndAfterStart, timeBeforeAndAfterEnd = timeBeforeAndAfterEnd,
                # [listBSS[-1]], [descriptions[-1]], [paths[-1]],
                          balanceConds = balanceConds, numStimulusConditions = numStimulusConditions, combineConditions=combineConditions, computeResiduals = computeResiduals, sqrtSpikes = sqrtSpikes,
                          crossvalidateNumFolds = crossvalidateNumFolds, xDimTest = xDimTest, overallFiringRateThresh=firingRateThresh, perConditionGroupFiringRateThresh = firingRateThresh, plotOutput = plotOutput)

if plotOutput:
    saveFiguresToPdf(pdfname=("gpfaOverOverNeuronTrialSweep-%s" % (brainAreas[0])))
    plt.close('all')

#breakpoint()

binnedSpksShortStOffSubsamplesFR = []
binnedSpksShortStOffSubsamplesCnt = []
for spkCnts in binnedResidSubsamples:
    binnedSpksShortStOffSubsamplesHereFR = []
    binnedSpksShortStOffSubsamplesHereCnt = []
    for spksUsed in spkCnts:
        binnedSpksShortStOffSubsamplesHereFR.append(spksUsed.convertUnitsTo('Hz'))
        binnedSpksShortStOffSubsamplesHereCnt.append(spksUsed.convertUnitsTo('count'))

    binnedSpksShortStOffSubsamplesFR.append(binnedSpksShortStOffSubsamplesHereFR)
    binnedSpksShortStOffSubsamplesCnt.append(binnedSpksShortStOffSubsamplesHereCnt)

#%% some descriptive data plots
from methods.BinnedSpikeSetListMethods import generateLabelGroupStatistics as genBSLLabGrp
from methods.BinnedSpikeSetListMethods import plotExampleChannelResponses
from methods.BinnedSpikeSetListMethods import plotStimDistributionHistograms
from methods.BinnedSpikeSetListMethods import plotFiringRates

tNcN = fsp[subsampleExpressions].fetch('trial_num_per_cond_min', 'ch_num')
descriptions = ["(%d trl, %d ch) - %s" % (tN, cN, dN) for tN, cN, dN in zip(*tNcN, brainAreas)]#[data[idx]['description'] for idx in dataIndsProcess]

listBSS = binnedSpksShortStOffSubsamplesFR
plotStimDistributionHistograms(listBSS, [dN + " " + desc for dN, desc in zip(dsNames, descriptions)])
plotFiringRates(listBSS, descriptions, supTitle = dsNames[0] + '\nDelay Start Offshift Firing Rates', cumulative = False)

listBSS = binnedSpksShortStOffSubsamplesCnt
plotStimDistributionHistograms(listBSS, [dN + " " + desc for dN, desc in zip(dsNames, descriptions)])
plotFiringRates(listBSS, descriptions, supTitle = dsNames[0] + '\nDelay Start Offshift Firing Rates', cumulative = False)


saveFiguresToPdf(pdfname=("genericMetricsOnNeurTrialSweep-%s" % brainAreas[0]))
plt.close('all')

#breakpoint()
#%% run GPFA
#from methods.BinnedSpikeSetListMethods import gpfaComputation

# if __name__ == '__main__':
#     import multiprocessing as mp
#     mp.set_start_method('forkserver')

#descriptions = [data[idx]['description'] for idx in dataIndsProcess]
#paths = [data[idx]['path'] for idx in dataIndsProcess]
#plotOutput = True
#
#xDimTest = [8]#[2,5,8,12,15]#[2,5,8,12,15]
#firingRateThresh = 1
#combineConditions = False
#numStimulusConditions = None # because V4 has two...
#sqrtSpikes = False
#
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
#listBSS = binnedResidShStOffSubsamples
#baselineSubtract = True # this is how we get the residuals
#plotOutput = True # really, I don't want to plot all these figures...
#timeBeforeAndAfterStart = (0+offshift, furthestForward+offshift)
#timeBeforeAndAfterEnd = None
#signalDescriptor = "fst%dMsDelShft%dMsFR%0.2f%s%sTrlCnd%dNrn%dFnoThresh" % (furthestForward, offshift,firingRateThresh, "Bsub" if baselineSubtract else "", "Sqrt" if sqrtSpikes else "", maxNumTrlPerCond, maxNumNeuron)
## note that with the firingRateThresh, it needs to be -infinity to be lower
## than any mean the new subset might have. Why? Because the average firing rate
## for baseline subtracted channels is zero if *ALL* of them are averaged--since
## we're taking just subsets, it's possible that they might be arbitrarily low.
## Moreover, what does a firing rate of a baseline subtracted trace really mean?
#firingRateThresh=-np.inf # we've already thresholded (but change it after the name so that's reflected)
#baselineSubtract = False # we've already thresholded (but change it after the name so that's reflected)

#crossvalidateNumFolds = 4


#dims, dimsLL, gpfaDimOut, gpfaTestInds, gpfaTrainInds = gpfaComputation(
#    listBSS, descriptions, paths, signalDescriptor = signalDescriptor,
#                # [listBSS[-1]], [descriptions[-1]], [paths[-1]],
#                          timeBeforeAndAfterStart = timeBeforeAndAfterStart, timeBeforeAndAfterEnd = timeBeforeAndAfterEnd,
#                          balanceDirs = True, numStimulusConditions = numStimulusConditions, combineConditions=combineConditions, baselineSubtract = baselineSubtract, sqrtSpikes = False,
#                          crossvalidateNumFolds = crossvalidateNumFolds, xDimTest = xDimTest, firingRateThresh=firingRateThresh, plotOutput = plotOutput)
#
## for the moment all we want to save is this...
## now save all the figures
#if plotOutput:
#
#    saveFiguresToPdf(pdfname=signalDescriptor)
#    plt.close('all')

#breakpoint()

listBSS = binnedResidSubsamples
    
shCovPropPopByArea = []
shCovPropNeurAvgByArea = []
shLogDetGenCovPropPopByArea = []
for gpfaArea, dimsUse, dimsGpfaUse in zip(gpfaDimOut, dims, dimsLL):
    if type(gpfaArea) is list:
        shCovPropPopByAreaHere = []
        shCovPropNeurAvgByAreaHere = []
        shLogDetGenCovPropPopByAreaHere = []
        for (gpA, dU, dGfU) in zip(gpfaArea, dimsUse, dimsGpfaUse):
            CR = [(gpfaCond[int(gpfaDimParamsUse)]['allEstParams'][0]['C'][:,:gpfaDimParamsUse],gpfaCond[int(gpfaDimParamsUse)]['allEstParams'][0]['R']) for gpfaCond, numDimsUse, gpfaDimParamsUse in zip(gpA, dU, dGfU)]
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
        CR = [(gpfaCond[int(gpfaDimParamsUse)]['allEstParams'][0]['C'][:,:gpfaDimParamsUse],gpfaCond[int(gpfaDimParamsUse)]['allEstParams'][0]['R']) for gpfaCond, numDimsUse, gpfaDimParamsUse in zip(gpfaArea, dimsUse, dimsGpfaUse)]
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
for idxSpks, (bnSp, description, dsName) in enumerate(zip(listBSS,descriptions,dsNames)):
#    idx = dataIndsProcess[idxSpks]
    separateNoiseCorrForLabels = True
    normalize = False #True
    if plotResid:
        scatterFig = plt.figure()
        if type(bnSp) is list:
            scatterFig.suptitle(dsName + "\n" + description + " geo mean FR vs corr var multiple subsets")
        else:
            scatterFig.suptitle(dsName + "\n" + description + " geo mean FR vs corr var")
    if type(bnSp) is list:
        residCorrMeanOverCondAllHere = []
        residCorrPerCondAllHere = []
        geoMeanOverallHere = []
        
        residCorrPerCondOBPTAllHere = []
        residCorrMeanOverCondOBPTAllHere = []
        for idxSubset, bS in enumerate(bnSp):
            trialLengthMs = furthestTimeBeforeDelay + furthestTimeAfterDelay - 1
            bSOneBinPrTrl = bS.convertUnitsTo('count').increaseBinSize(trialLengthMs)
            residualSpikes, residCorrMeanOverCond, residCorrPerCond, geoMeanCntAll = bS.convertUnitsTo('count').residualCorrelations(bS.labels['stimulusMainLabel'], plot=plotResid, separateNoiseCorrForLabels = separateNoiseCorrForLabels, figTitle = description + (" subset %d" % idxSubset), normalize = normalize)
            residualSpikesOBPT, residCorrMeanOverCondOBPT, residCorrPerCondOBPT, geoMeanCntAllOBPT = bSOneBinPrTrl.residualCorrelations(bS.labels['stimulusMainLabel'], plot=plotResid, separateNoiseCorrForLabels = separateNoiseCorrForLabels, figTitle = description + (" 1 bin/trl subset %d" % idxSubset), normalize = normalize)
            
            
            # Only care about single repeats of the non-self pairs...
            upperTriInds = np.triu_indices(residCorrMeanOverCond.shape[0], 1)
            residCorrMeanOverCond = residCorrMeanOverCond[upperTriInds]
            geoMeanCntAll = geoMeanCntAll[upperTriInds]
            residCorrPerCond = residCorrPerCond[upperTriInds]
            

            residCorrMeanOverCondAllHere.append(residCorrMeanOverCond)
            geoMeanOverallHere.append(geoMeanCntAll)
            residCorrPerCondAllHere.append(residCorrPerCond)

            # only single repeats here as well
            residCorrPerCondOBPT = residCorrPerCondOBPT[upperTriInds]
            
            residCorrMeanOverCondOBPTAllHere.append(residCorrMeanOverCondOBPT)
            residCorrPerCondOBPTAllHere.append(residCorrPerCondOBPT)
            
            
            
            
            
            if plotResid:
                axs = scatterFig.subplots(2,1)
                axs[0].scatter(geoMeanCntAll.flatten(), residCorrMeanOverCond.flatten())
                axs[0].set_ylabel("Correlated Variability")
                
                binWidth = np.max(geoMeanCntAll)/10
                notNan = np.where(~np.isnan(residCorrMeanOverCond))
                binAvg, binEdges, bN = binned_statistic(geoMeanCntAll[notNan].flatten(), residCorrMeanOverCond[notNan].flatten(), statistic='mean', bins = np.arange(np.max(geoMeanCntAll), step=binWidth))
                binStd, _, _ = binned_statistic(geoMeanCntAll[notNan].flatten(), residCorrMeanOverCond[notNan].flatten(), statistic='std', bins = np.arange(np.max(geoMeanCntAll), step=binWidth))
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
        residualSpikes, residCorrMeanOverCond, residCorrPerCond, geoMeanCntAll = bS.residualCorrelations(bS.labels['stimulusMainLabel'], plot=plotResid, separateNoiseCorrForLabels = separateNoiseCorrForLabels, figTitle = description, normalize = normalize)
        
        
        residCorrMeanOverCondAll.append(residCorrMeanOverCond)
        geoMeanOverall.append(geoMeanCntAll)
        residCorrPerCondAll.append(residCorrPerCond)
        
        # Only care about single repeats of the non-self pairs...
        upperTriInds = np.triu_indices(residCorrMeanOverCond.shape[0], 1)
        residCorrMeanOverCond = residCorrMeanOverCond[upperTriInds]
        geoMeanCntAll = geoMeanCntAll[upperTriInds]
        residCorrPerCond = residCorrPerCond[upperTriInds]
        
        
        if plotResid:
            axs = scatterFig.subplots(2,1)
            axs[0].scatter(geoMeanCntAll.flatten(), residCorrMeanOverCond.flatten())
            axs[0].set_ylabel("Correlated Variability")
            
            binWidth = np.max(geoMeanCntAll)/10
            binAvg, binEdges, bN = binned_statistic(geoMeanCntAll.flatten(), residCorrMeanOverCond.flatten(), statistic='mean', bins = np.arange(np.max(geoMeanCntAll), step=binWidth))
            binStd, _, _ = binned_statistic(geoMeanCntAll.flatten(), residCorrMeanOverCond.flatten(), statistic='std', bins = np.arange(np.max(geoMeanCntAll), step=binWidth))
            binEdges = np.diff(binEdges)[0]/2+binEdges[:-1]
            axs[1].plot(binEdges, binAvg)
            axs[1].fill_between(binEdges, binAvg-binStd, binAvg+binStd, alpha=0.2)
            axs[1].set_xlabel("Geometric Mean Firing Rate")
            axs[1].set_ylabel("Correlated Variability")


    
if plotResid:
    from methods.GeneralMethods import saveFiguresToPdf
    
    saveFiguresToPdf(pdfname=("residualsOverNeuronTrialSweep-%s" % brainAreas[0]))
    plt.close('all')



if type(residCorrPerCondAll[0]) is list:
    mnCorrPerCond = []
    stdCorrPerCond = []
    for rsCorr in residCorrPerCondAll:
        mnCorrPerCond.append([residCorr.mean(axis=0) for residCorr in rsCorr])
        stdCorrPerCond.append([residCorr.std(axis=0) for residCorr in rsCorr])
    
    mnCorrPerCondOBPT = []
    stdCorrPerCondOBPT = []
    for rsCorr in residCorrPerCondOBPTAll:
        mnCorrPerCondOBPT.append([residCorr.mean(axis=0) for residCorr in rsCorr])
        stdCorrPerCondOBPT.append([residCorr.std(axis=0) for residCorr in rsCorr])
    
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

def pltAllVsAll(descriptions, labelForCol, labelForMarker, dsNames, metricDict):
    colorsUse = BinnedSpikeSet.colorset
    from itertools import product,chain
    # this variable will use matplotlib's maker of markers to combine polygon
    # numbers and angles
    polygonSides = range(2,8)
    rotDegBtShapes = 30
    # 0 is to make this a polygon
    polyAng = [product([polySd], [0], np.arange(0, np.lcm(np.int(np.round(360/polySd)), rotDegBtShapes) if np.lcm(np.int(np.round(360/polySd)), rotDegBtShapes) < 360 else 360, rotDegBtShapes)) for polySd in polygonSides]
    markerCombos = np.stack(chain.from_iterable(polyAng))
    # sort by angle--this ends up ensuring that sequential plots have different
    # polygons
#    markerCombos = markerCombos[markerCombos[:,2].argsort()]
    # back in tuple form
#    markerCombos = tuple(tuple(mc) for mc in markerCombos)
#    breakpoint()
    for metricNum, (metricName, metricVal) in enumerate(metricDict.items()):
        for metric2Num, (metric2Name, metric2Val) in enumerate(metricDict.items()):
            if metric2Num > metricNum:
                plt.figure()
                plt.title('%s vs %s' % ( metricName, metric2Name ))
                plt.suptitle(dsName)
                unLabForCol, colNum = np.unique(labelForCol, return_inverse=True)
                unLabForTup, mcNum = np.unique(labelForMarker, return_inverse=True)
                [plt.scatter(m1, m2, label=desc, color=colorsUse[colN,:], marker=markerCombos[mcN]) for m1, m2, desc, colN, mcN in zip(metricVal, metric2Val, descriptions, colNum, mcNum)]
                plt.xlabel(metricName)
                plt.ylabel(metric2Name)
                plt.legend(prop={'size':5})

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
    '% sh var' : shCovPropNeurAvgByAreaAllSub,
    'dimensionality' : dimsByArea,
    'gen cov (logdet) pop' : shLogDetGenCovPropPopByAreaAllSub,
}
pltAllVsAll(descriptions, numNeurs, numTrls, dsNames, metricDict)
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

saveFiguresToPdf(pdfname=('scatterMetricsOverNeuronTrialSweep-%s' % brainAreas[0]))

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
