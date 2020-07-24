"""
Compute the dimensionality of V4 in 1s segment
"""
from matplotlib import pyplot as plt
import numpy as np
import re
import dill as pickle # because this is what people seem to do?
import hashlib
import json
from pathlib import Path

from methods.GeneralMethods import saveFiguresToPdf
# database stuff
from setup.DataJointSetup import DatasetInfo, DatasetGeneralLoadParams, BinnedSpikeSetInfo, FilterSpikeSetParams
import datajoint as dj
# the decorator to save these to a database!
from decorators.AnalysisCallDecorators import saveCallsToDatabase

from classes.Dataset import Dataset
from classes.BinnedSpikeSet import BinnedSpikeSet


# for generating the binned spike sets
from methods.BinnedSpikeSetListMethods import generateBinnedSpikeListsAroundState as genBSLAroundState

# for the gpfa...
from methods.BinnedSpikeSetListMethods import gpfaComputation

@saveCallsToDatabase
def v4Stim1sDimensionality():
    dsi = DatasetInfo()
    bsi = BinnedSpikeSetInfo()

    print("Computing/loading binned spike sets")
    trialType = 'all'
    firingRateThresh = 1
    # suggestion of an okay value (not too conservative as with 8, not too lenient as with 1)
    fanoFactorThresh = 4
    furthestTimeBeforeState=300
    furthestTimeAfterState=300
    binSizeMs = 20
    # we let GPFA compute residuals--this allows it to deal with firing
    # rate thresholds on a condition-by-condition basis
    baselineSubtract = False
    setStartToStateEnd = False
    setEndToStateStart = False
    # this shouldn't affect GPFA... but will affect fano factor cutoffs...
    unitsOut = 'count'
    dsExtract = dsi['brain_area="V4"']#.grabDatasets()
    keyStateName = 'stimulus'#[ds.keyStates['stimulus'] for ds in dsExtract]
    _, bsiFiltKeys = genBSLAroundState(dsExtract,
                                        keyStateName,
                                        trialType = trialType,
                                        lenSmallestTrl=0, 
                                        binSizeMs = binSizeMs, 
                                        furthestTimeBeforeState=furthestTimeBeforeState,
                                        furthestTimeAfterState=furthestTimeAfterState,
                                        setStartToStateEnd = setStartToStateEnd,
                                        setEndToStateStart = setEndToStateStart,
                                        returnResiduals = baselineSubtract,
                                        unitsOut = unitsOut, 
                                        firingRateThresh = firingRateThresh,
                                        fanoFactorThresh = fanoFactorThresh 
                                        )

    bsiFilt = bsi[bsiFiltKeys]#[dsi['brain_area="V4"']['start_alignment_state="Target"']]
    #bsToFilt = bsiFilt.grabBinnedSpikes()

    print("Subsampling binned spike sets")

    filteredBSS = []
    numNeuronsTest = 50
    filterDescription = "only non-choice trials, 50 random neurons"
    fsp = FilterSpikeSetParams()
    filtBssExp = bsiFilt[fsp['filter_description = "%s"' % filterDescription]]

    #for bsF, bsUse in zip(bsiFilt.fetch('KEY'), bsToFilt):
    for bsF in bsiFilt.fetch('KEY'):
        bsFExp = bsi[bsF]
        filtBssExp = bsFExp[fsp['filter_description = "%s"' % filterDescription]]

        if len(filtBssExp) == 0:
            bsFExp = bsi[bsF] # since this isn't actually an expression
            bsH = bsFExp.grabBinnedSpikes() # since bsFExp is a primary key restriction, there's only one here...
            assert len(bsH) == 1, "j/k there was more than one or none"
            bsH = bsH[0]
            labels = bsH.labels
            nonChoiceTrials = labels['sequenceLength'] != labels['sequencePosition']
            nonChoiceTrials = nonChoiceTrials.squeeze()
            channelsUse = np.random.permutation(range(bsH.shape[1]))[:numNeuronsTest]
            condLabel = 'stimulusMainLabel'
            filteredBSS.append(bsFExp.filterBSS("other", filterDescription, condLabel = condLabel, trialFilter = nonChoiceTrials, channelFilter = channelsUse))
#

    filtBssExp = bsiFilt[fsp['filter_description = "%s"' % filterDescription]]

    print('Computing and plotting GPFA')
    plotOutput = True
    # max at 50--can't have more dimensions than neurons!
    xDimTest = [5,10,20,40,49] 
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
                              balanceConds = balanceConds, numStimulusConditions = numStimulusConditions, combineConditions=combineConditions, computeResiduals = computeResiduals, sqrtSpikes = sqrtSpikes,
                              crossvalidateNumFolds = crossvalidateNumFolds, xDimTest = xDimTest, overallFiringRateThresh=firingRateThresh, perConditionGroupFiringRateThresh = firingRateThresh, plotOutput = plotOutput)


    saveFiguresToPdf(pdfname="V4GPFA")
    plt.close('all')

