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

#datasets = dsi.grabDatasets()
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
#offshift = 75 #ms
#firingRateThresh = 1
#fanoFactorThresh = 4
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
#                                    )
#baselineSubtract = False
#furthestTimeBeforeDelay=-offshift # note that this starts it *forwards* from the delay
#furthestTimeAfterDelay=lenSmallestTrl+offshift
#binnedSpikesShortStartOffshift, _ = genBSLAroundState(datasets,
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
#                                    )

print("Computing/loading binned spike sets")
trialType = 'all'
firingRateThresh = 1
fanoFactorThresh = 4
furthestTimeBeforeState=300 # note that this starts it *forwards* from the delay
furthestTimeAfterState=300
binSizeMs = 20
baselineSubtract = False
dsExtract = dsi['brain_area="V4"']#.grabDatasets()
keyStateName = 'stimulus'#[ds.keyStates['stimulus'] for ds in dsExtract]
_, bsiFiltKeys = genBSLAroundState(dsExtract,
                                    keyStateName,
                                    trialType = trialType,
                                    lenSmallestTrl=0, 
                                    binSizeMs = binSizeMs, 
                                    furthestTimeBeforeState=furthestTimeBeforeState, # note that this starts it *forwards* from the delay
                                    furthestTimeAfterState=furthestTimeAfterState,
                                    setStartToDelayEnd = False,
                                    setEndToDelayStart = False,
                                    returnResiduals = baselineSubtract,
                                    unitsOut = 'count', # this shouldn't affect GPFA... but will affect fano factor cutoffs...
                                    firingRateThresh = firingRateThresh,
                                    fanoFactorThresh = fanoFactorThresh # suggestion of an okay value (not too conservative as with 8, not too lenient as with 1)
                                    )


# we want trials without 'Success' state (they're a choice trial)
# also trials that are in general > 1000ms, to make sure we have a full second
#trlLenLog = np.array([bnSp[0].shape[0] == 50 for bnSp in binnedSpikesAroundTarget[0]])
#noChStLog = ~dsExtract[0].successfulTrials().trialsWithState('Success')
#bsH = binnedSpikesAroundTarget[0][noChStLog]


bsiFilt = bsi[bsiFiltKeys]#[dsi['brain_area="V4"']['start_alignment_state="Target"']]
#bsToFilt = bsiFilt.grabBinnedSpikes()
filteredBSS = []
numNeuronsTest = 50

print("Subsampling binned spike sets")

fsp = FilterSpikeSetParams()
filtBssExp = bsiFilt[fsp['filter_description="only non-choice trials, 50 random neurons"']]

#for bsF, bsUse in zip(bsiFilt.fetch('KEY'), bsToFilt):
if len(filtBssExp) == 0:
    for bsF in bsiFilt.fetch('KEY'):
        dtUse = dsi[bsF]
#    noChoiceStLog = ~dtUse.grabDatasets()[0].successfulTrials().trialsWithState('Success')
        bsFExp = bsi[bsF] # since this isn't actually an expression
        bsH = bsFExp.grabBinnedSpikes() # since bsFExp is a primary key restriction, there's only one here...
        assert len(bsH) == 1, "j/k there was more than one or none"
        bsH = bsH[0]
        labels = bsH.labels
        nonChoiceTrials = labels['sequenceLength'] != labels['sequencePosition']
        nonChoiceTrials = nonChoiceTrials.squeeze()
        channelsUse = np.random.permutation(range(bsH.shape[1]))[:numNeuronsTest]
        condLabel = 'stimulusMainLabel'
        filteredBSS.append(bsFExp.filterBSS("other", "only non-choice trials, 50 random neurons", condLabel = condLabel, trialFilter = nonChoiceTrials, channelFilter = channelsUse))
#

filtBssExp = bsiFilt[fsp['filter_description="only non-choice trials, 50 random neurons"']]


# some gpfa

from methods.BinnedSpikeSetListMethods import gpfaComputation

#breakpoint()
plotOutput = True
# max at 50--can't have more dimensions than neurons!
xDimTest = [5,10,20,40,49] # 50 will fail on one of them... buuut! I still need to replace everything I computed >.>
firingRateThresh = 1 if not baselineSubtract else 0
combineConditions = False
numStimulusConditions = None # because V4 has two...
sqrtSpikes = False
crossvalidateNumFolds = 4
computeResiduals = True
balanceConds = True

timeBeforeAndAfterStart = (-150 if -150 > -furthestTimeBeforeState else -furthestTimeBeforeState, 300)
timeBeforeAndAfterEnd = (-300, 150 if 150 < furthestTimeAfterState else furthestTimeAfterState)

dims, dimsLL, gpfaDimOut, gpfaTestInds, gpfaTrainInds = gpfaComputation(
    filtBssExp,timeBeforeAndAfterStart = timeBeforeAndAfterStart, timeBeforeAndAfterEnd = timeBeforeAndAfterEnd,
                # [listBSS[-1]], [descriptions[-1]], [paths[-1]],
                          balanceConds = balanceConds, numStimulusConditions = numStimulusConditions, combineConditions=combineConditions, computeResiduals = computeResiduals, sqrtSpikes = sqrtSpikes,
                          crossvalidateNumFolds = crossvalidateNumFolds, xDimTest = xDimTest, overallFiringRateThresh=firingRateThresh, perConditionGroupFiringRateThresh = firingRateThresh, plotOutput = plotOutput)

# other stuff...

saveFiguresToPdf(pdfname="V4GPFA")
plt.close('all')
breakpoint()

