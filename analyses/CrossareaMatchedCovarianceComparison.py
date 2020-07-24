"""
This function is responsible for the cross-area analyses/scatter matrices etc
Like HeadlessAnalysisRun2-5.py for now...
"""
from matplotlib import pyplot as plt
import numpy as np
import re
import dill as pickle # because this is what people seem to do?
import hashlib
import json
from pathlib import Path
from scipy.stats import binned_statistic

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

# subsampling
from methods.BinnedSpikeSetListMethods import subsampleBinnedSpikeSetsToMatchNeuronsAndTrialsPerCondition as subsmpMatchCond

# for the gpfa...
from methods.BinnedSpikeSetListMethods import gpfaComputation

# for descriptions of the data
from methods.BinnedSpikeSetListMethods import generateLabelGroupStatistics as genBSLLabGrp
from methods.BinnedSpikeSetListMethods import plotExampleChannelResponses
from methods.BinnedSpikeSetListMethods import plotStimDistributionHistograms
from methods.BinnedSpikeSetListMethods import plotFiringRates

# for computing metrics
from methods.BinnedSpikeSetListMethods import rscComputations
from methods.GpfaMethods import computePopulationMetrics

# for plotting the metrics
from methods.plotUtils.UnsortedPlotMethods import pltAllVsAll


@saveCallsToDatabase
def crossareaMatchedCovarianceComparison():

    dsgl = DatasetGeneralLoadParams()
    dsi = DatasetInfo()
    bsi = BinnedSpikeSetInfo()
    fsp = FilterSpikeSetParams()


    keyStateName = 'delay' # we look around the *delay*
    offshift = 75 #ms
    lenSmallestTrl = 301 #ms; 
    binSizeMs = 25 # good for PFC LDA #50 # 
    trialType = 'successful'
    firingRateThresh = 1
    fanoFactorThresh = 4
    baselineSubtract = False
    furthestTimeBeforeState=-offshift # note that this starts it *forwards* from the delay
    # this'll bleed a little into the start of the new stimulus with the
    # offset, but before any neural response can happen
    furthestTimeAfterState=lenSmallestTrl+offshift
    # this means that the data will lie around the delay start
    setStartToStateEnd = False
    setEndToStateStart = True
    unitsOut = 'count' # this shouldn't affect GPFA... but will affect fano factor cutoffs...

    print("Computing/loading binned spike sets")
    _, bssiKeys = genBSLAroundState(dsi,
                                        keyStateName,
                                        trialType = trialType,
                                        lenSmallestTrl=lenSmallestTrl, 
                                        binSizeMs = binSizeMs, 
                                        furthestTimeBeforeState=furthestTimeBeforeState,
                                        furthestTimeAfterState=furthestTimeAfterState,
                                        setStartToStateEnd = setStartToStateEnd,
                                        setEndToStateStart = setEndToStateStart,
                                        returnResiduals = baselineSubtract,
                                        unitsOut = unitsOut,
                                        firingRateThresh = firingRateThresh,
                                        fanoFactorThresh = fanoFactorThresh # suggestion of an okay value (not too conservative as with 8, not too lenient as with 1)
                                        )

    # only doing one subsample for now...
    numSubsamples = 1
    # subsamples will have at least 60 neurons
    minNumNeurons = 60

    bssExp = bsi[bssiKeys][fsp['ch_num>%d' % minNumNeurons]]
    # technically already loaded above, but in future we'll want to be able to
    # grab the directly from the db call
    binnedSpikesList = bssExp.grabBinnedSpikes() 
    maxNumNeuron = np.min([bnSp.shape[1] for bnSp in binnedSpikesList])
    labelName = 'stimulusMainLabel'
    numTrls = [np.unique(bnSp.labels[labelName],axis=0,return_counts=True)[1] for bnSp in binnedSpikesList]
    maxNumTrlPerCond = np.min(np.hstack(numTrls))
    extraOpts = {
        dsi[dsgl['task="cuedAttention"']] : {
            'description' : 'choose first blank',
            'filter' : (lambda res : res.grabBinnedSpikes()[0].labels['sequencePosition']==2)
            }
        }
    binnedResidShStOffSubsamples, subsampleExpressions, dsNames, brainAreas, tasks = subsmpMatchCond(bssExp, maxNumTrlPerCond = maxNumTrlPerCond, maxNumNeuron = maxNumNeuron, labelName = labelName, numSubsamples = numSubsamples, extraOpts = extraOpts)

    print("Computing GPFA")
    """ These are the GPFA parameters """
    plotGpfa = False
    # cutting off dim of 15... because honestly residuals don't seem to even go
    # past 8
    xDimTest = [2,5,8,12]#[2,5,8,12,15]
    firingRateThresh = 1 if not baselineSubtract else 0
    combineConditions = False
    numStimulusConditions = None # uses all stimulus conditions
    sqrtSpikes = False
    crossvalidateNumFolds = 4
    computeResiduals = True
    balanceConds = True
    timeBeforeAndAfterStart = (furthestTimeBeforeState, furthestTimeAfterState)
    timeBeforeAndAfterEnd = None

    dims, dimsLL, gpfaDimOut, gpfaTestInds, gpfaTrainInds = gpfaComputation(
        subsampleExpressions,timeBeforeAndAfterStart = timeBeforeAndAfterStart, timeBeforeAndAfterEnd = timeBeforeAndAfterEnd,
                              balanceConds = balanceConds, numStimulusConditions = numStimulusConditions, combineConditions=combineConditions, computeResiduals = computeResiduals, sqrtSpikes = sqrtSpikes,
                              crossvalidateNumFolds = crossvalidateNumFolds, xDimTest = xDimTest, overallFiringRateThresh=firingRateThresh, perConditionGroupFiringRateThresh = firingRateThresh, plotOutput = plotGpfa)

    if plotGpfa:
        saveFiguresToPdf(pdfname=("GPFAManyAreasNeur%dTrl%d" % (maxNumNeuron, maxNumTrlPerCond)))
        plt.close('all')

    
    plotGenericMetrics = False
    binnedSpksShortStOffSubsamplesFR = []
    binnedSpksShortStOffSubsamplesCnt = []
    for spkCnts in binnedResidShStOffSubsamples:
        binnedSpksShortStOffSubsamplesHereFR = []
        binnedSpksShortStOffSubsamplesHereCnt = []
        for spksUsed in spkCnts:
            binnedSpksShortStOffSubsamplesHereFR.append(spksUsed.convertUnitsTo('Hz'))
            binnedSpksShortStOffSubsamplesHereCnt.append(spksUsed.convertUnitsTo('count'))

        binnedSpksShortStOffSubsamplesFR.append(binnedSpksShortStOffSubsamplesHereFR)
        binnedSpksShortStOffSubsamplesCnt.append(binnedSpksShortStOffSubsamplesHereCnt)

    # some descriptive data plots
    descriptions = dsNames #[data[idx]['description'] for idx in dataIndsProcess]

    if plotGenericMetrics:
        supTitle = "%s %s %s" % (keyStateName, 
                                    "end" if setStartToStateEnd else "start",
                                    "offshift" if offshift != 0 else "")
        plotStimDistributionHistograms(binnedSpksShortStOffSubsamplesFR, descriptions)
        plotFiringRates(binnedSpksShortStOffSubsamplesFR, descriptions, supTitle = supTitle + " firing rates", cumulative = False)

        plotStimDistributionHistograms(binnedSpksShortStOffSubsamplesCnt, descriptions)
        plotFiringRates(binnedSpksShortStOffSubsamplesCnt, descriptions, supTitle = supTitle + " count", cumulative = False)


        saveFiguresToPdf(pdfname="genericMetrics")
        plt.close('all')

        
    # Population metrics
    popMetricsDict = computePopulationMetrics(gpfaDimOut, dimsLL, dims)

    #%% Residual correlations
    plotResid = False
    separateNoiseCorrForLabels = not combineConditions
    normalize = False
    rscMetricDict = rscComputations(binnedResidShStOffSubsamples, descriptions, labelName, separateNoiseCorrForLabels = separateNoiseCorrForLabels, normalize = normalize, plotResid = plotResid)

    if plotResid:
        saveFiguresToPdf(pdfname="residualsOverSubsets")
        plt.close('all')

    metricDict = {}
    metricDict.update(popMetricsDict)
    metricDict.update(rscMetricDict)

    pltAllVsAll(descriptions, metricDict, brainAreas, tasks)

    plotScatterMetrics = True
    if plotScatterMetrics:
        saveFiguresToPdf(pdfname='scatterMetricsOverSubsets')


if __name__=='__main__':
    crossareaMatchedCovarianceComparison()