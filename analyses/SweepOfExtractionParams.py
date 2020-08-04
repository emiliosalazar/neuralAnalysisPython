"""
This analysis function runs factory analysis on the same preprocessing and GPFA
was run, and plots the outputs of the two for comparison
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
from setup.DataJointSetup import DatasetInfo, DatasetGeneralLoadParams, BinnedSpikeSetInfo, BinnedSpikeSetProcessParams, FilterSpikeSetParams
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
from methods.plotUtils.UnsortedPlotMethods import plotAllVsAll, plotMetricVsExtractionParams


@saveCallsToDatabase
def sweepOfExtractionParams():

    dsgl = DatasetGeneralLoadParams()
    dsi = DatasetInfo()
    bsi = BinnedSpikeSetInfo()
    bsp = BinnedSpikeSetProcessParams()
    fsp = FilterSpikeSetParams()


    outputFiguresRelativePath = []
    singKey = []
    singKey.append(dsi['brain_area="M1" AND dataset_name LIKE "%2019-03-22%"'].fetch("KEY")[0]) # M1
#    singKey.append(dsi['brain_area="PFC" AND dataset_name LIKE "%2018-07-14%"'].fetch("KEY")[0]) # PFC
#singKey.append(dsi['brain_area="V4"'].fetch("KEY")[0]) # V4
    singAreaDs = dsi[singKey]

    keyStateName = 'delay' # we look around the *delay*
    offshift = 75 #ms
    lenSmallestTrl = 301 #ms; 
    binSizeMsSweep = [25,300] # good for PFC LDA #50 # 
    trialType = 'successful'
    firingRateThresh = 1
    # suggestion of an okay value (not too conservative as with 8, not too
    # lenient as with 1)
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
    bssiKeys = []
    for binSizeMs in binSizeMsSweep:
        _, bssiKeysThisBinSize = genBSLAroundState(singAreaDs,
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
                                            fanoFactorThresh = fanoFactorThresh 
                                            )
        bssiKeys.append(bssiKeysThisBinSize)

    print("Subsampling binned spike sets")
    # only doing one subsample for now...
    numSubsamples = 1
    numValsSweep = 2
    # this isn't the start of the sweep, but the *end* of the sweep that must
    # at least be reached--really it's just getting rid of any datasets that
    # have fewer than this many neurons
    minNumNeuronsToSweepTo = 60
    bssExp = bsi[bssiKeys][fsp['ch_num>%d' % minNumNeuronsToSweepTo]]

    # low end of the sweeps
    minNumNeuron = 30
    minNumTrlPerCond = 20

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
#extraOpts = {
#    dsi[dsgl['task="cuedAttention"']] : {
#        'description' : 'choose non-choice trials',
#        'filter' : (lambda res : res.grabBinnedSpikes()[0].labels['sequencePosition']!=res.grabBinnedSpikes()[0].labels['sequenceLength'])
#        }
#    }
    binnedResidSubsamples = []
    subsampleExpressions = []
    dsNames = []
    brainAreas = []
    tasks = []
    combs = []
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
        numTrls = np.unique(bSSHere.labels[labelName],axis=0,return_counts=True)[1]
        maxNumTrlPerCond = np.min(numTrls)
        if numValsSweep == 1: # if we're not sweeping values, we'll just use the default
            minNumNeuron = maxNumNeuron
            minNumTrlPerCond = maxNumTrlPerCond
        numNeuronTests = np.round(np.geomspace(minNumNeuron, maxNumNeuron, numValsSweep))

        numTrialsPerCondTests = np.round(np.geomspace(minNumTrlPerCond, maxNumTrlPerCond, numValsSweep))

        from itertools import product
        subsampCombos = tuple(product(numNeuronTests, numTrialsPerCondTests, [numSubsamples] if type(numSubsamples) is not list else numSubsamples))
        combs+=subsampCombos

        for combo in subsampCombos:
            numNeur = int(combo[0])
            numTrial = int(combo[1])
            numSubsample = int(combo[2])
            bRSub, subE, dNm, bA, tsk = subsmpMatchCond(bsi[bKey], maxNumTrlPerCond = numTrial, maxNumNeuron = numNeur, labelName = labelName, numSubsamples = numSubsample, extraOpts = extraOpts)

            binnedResidSubsamples += bRSub
            subsampleExpressions += subE
            dsNames += dNm
            brainAreas += [bA[0] + ' %dms bin' % bsp[bsInfo]['bin_size']]
            tasks += tsk

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
    useFaLog = [False,True]

    
    dims, dimsLL, gpfaDimOut, gpfaTestInds, gpfaTrainInds, brainAreasTechnique, brainAreasNoTechnique = [],[],[],[],[], [], []
    for useFa in useFaLog:
        dimsH, dimsLLH, gpfaDimOutH, gpfaTestIndsH, gpfaTrainIndsH = gpfaComputation(
            subsampleExpressions,timeBeforeAndAfterStart = timeBeforeAndAfterStart, timeBeforeAndAfterEnd = timeBeforeAndAfterEnd,
                        # [listBSS[-1]], [descriptions[-1]], [paths[-1]],
                                  balanceConds = balanceConds, numStimulusConditions = numStimulusConditions, combineConditions=combineConditions, computeResiduals = computeResiduals, sqrtSpikes = sqrtSpikes,
                                  crossvalidateNumFolds = crossvalidateNumFolds, xDimTest = xDimTest, overallFiringRateThresh=firingRateThresh, perConditionGroupFiringRateThresh = firingRateThresh, plotOutput = plotGpfa, useFa = useFa)

        dims += dimsH
        dimsLL += dimsLLH
        gpfaDimOut += gpfaDimOutH
        gpfaTestInds += gpfaTestIndsH
        gpfaTrainInds += gpfaTrainIndsH

        brainAreasTechnique += [bA + ' %s' % ('FA' if useFa else 'GPFA') for bA in brainAreas]
        brainAreasNoTechnique += brainAreas

    brainAreas = brainAreasTechnique
    subsampleExpressions*=len(useFaLog)
    dsNames*=len(useFaLog)
    tasks*=len(useFaLog)
    binnedResidSubsamples*=len(useFaLog)
    combs*=len(useFaLog)

    combs = np.stack(combs)
    numNeurs = combs[:, 0]
    numTrls = combs[:, 1]


    if plotGpfa:
        outputFiguresRelativePath.append(saveFiguresToPdf(pdfname=("gpfaOverOverNeuronTrialSweep-%s and %s" % (brainAreas[0],brainAreas[-1]))))
        plt.close('all')

    
    plotGenericMetrics = False
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

    tNcN = [fsp[sbsmpExp].fetch('trial_num_per_cond_min', 'ch_num') for sbsmpExp in subsampleExpressions]
    tNcN = list(zip(*tNcN))
    descriptions = ["(%d trl, %d ch) - %s" % (tN, cN, dN) for tN, cN, dN in zip(*tNcN, brainAreas)]

    if plotGenericMetrics:
        supTitle = "%s %s %s %s" % (dsNames[0], keyStateName, 
                                    "end" if setStartToStateEnd else "start",
                                    "offshift" if offshift != 0 else "")
        plotStimDistributionHistograms(binnedSpksShortStOffSubsamplesFR, [dN + " " + desc for dN, desc in zip(dsNames, descriptions)])
        plotFiringRates(binnedSpksShortStOffSubsamplesFR, descriptions, supTitle = supTitle + " firing rates", cumulative = False)

        plotStimDistributionHistograms(binnedSpksShortStOffSubsamplesCnt, [dN + " " + desc for dN, desc in zip(dsNames, descriptions)])
        plotFiringRates(binnedSpksShortStOffSubsamplesCnt, descriptions, supTitle = supTitle + " count", cumulative = False)


        outputFiguresRelativePath.append(saveFiguresToPdf(pdfname="genericMetricsOnNeurTrialSweep-%s and %s"))
        plt.close('all')

        
    # Population metrics
    popMetricsDict = computePopulationMetrics(gpfaDimOut, dimsLL, dims)

    #%% Residual correlations
    plotResid = False
    separateNoiseCorrForLabels = not combineConditions
    normalize = False
    rscMetricDict = rscComputations(binnedResidSubsamples, descriptions, labelName, separateNoiseCorrForLabels = separateNoiseCorrForLabels, normalize = normalize, plotResid = plotResid)

    if plotResid:
        outputFiguresRelativePath.append(saveFiguresToPdf(pdfname=("residualsOverNeuronTrialSweep-%s" % brainAreas[0])))
        plt.close('all')

    metricDict = {}
    metricDict.update(popMetricsDict)
    metricDict.update(rscMetricDict)


    plotScatterMetrics = True
    if plotScatterMetrics:

        faOrGpfa = np.repeat(useFaLog, len(subsampleExpressions)/len(useFaLog))
        binSize = np.stack([bsp[sbExp]['bin_size'] for sbExp in subsampleExpressions])
        labelForColor = np.concatenate((binSize, faOrGpfa[:,None]),axis=1)
        labelForMarkers = np.concatenate((numNeurs[:,None], numTrls[:,None]),axis=1)

        plotAllVsAll(descriptions, metricDict, labelForColor, labelForMarkers, dsNames[0])
        outputFiguresRelativePath.append(saveFiguresToPdf(pdfname=('scatterMetricsOverNeuronTrialSweep-%s and %s-col=%s-mrk=%s' % (brainAreas[0],brainAreas[-1],'bin&meth','trlNeur'))))
        plt.close('all')

        labelForColor = faOrGpfa
        plotAllVsAll(descriptions, metricDict, labelForColor, labelForMarkers, dsNames[0])
        outputFiguresRelativePath.append(saveFiguresToPdf(pdfname=('scatterMetricsOverNeuronTrialSweep-%s and %s-col=%s-mrk=%s' % (brainAreas[0],brainAreas[-1],'method','trlNeur'))))
        plt.close('all')

        labelForColor = faOrGpfa
        labelForMarkers = binSize
        plotAllVsAll(descriptions, metricDict, labelForColor, labelForMarkers, dsNames[0])
        outputFiguresRelativePath.append(saveFiguresToPdf(pdfname=('scatterMetricsOverNeuronTrialSweep-%s and %s-col=%s-mrk=%s' % (brainAreas[0],brainAreas[-1],'method','binSz'))))
        plt.close('all')

        labelForColor = np.concatenate((faOrGpfa[:,None],numNeurs[:,None], numTrls[:,None]),axis=1)
        labelForMarkers = binSize
        plotAllVsAll(descriptions, metricDict, labelForColor, labelForMarkers, dsNames[0])
        outputFiguresRelativePath.append(saveFiguresToPdf(pdfname=('scatterMetricsOverNeuronTrialSweep-%s and %s-col=%s-mrk=%s' % (brainAreas[0],brainAreas[-1],'methodTrlNeur','binSz'))))
        plt.close('all')

        labelForColor = numTrls
        labelForMarkers = numNeurs
        plotAllVsAll(descriptions, metricDict, labelForColor, labelForMarkers, dsNames[0])
        outputFiguresRelativePath.append(saveFiguresToPdf(pdfname=('scatterMetricsOverNeuronTrialSweep-%s and %s-col=%s-mrk=%s' % (brainAreas[0],brainAreas[-1],'numTrl','numNeur'))))
        plt.close('all')

        descriptions = ["(%d trl, %d ch) - %s" % (tN, cN, dN) for tN, cN, dN in zip(*tNcN, brainAreasNoTechnique)]
        labelForSplit = faOrGpfa
        labelForMarkers = binSize
        labelForColor = np.concatenate((numNeurs[:,None], numTrls[:,None]),axis=1)
        splitNames = ['GPFA', 'FA']
        plotMetricVsExtractionParams(descriptions, popMetricsDict, splitNames, labelForSplit, labelForColor, labelForMarkers, supTitle="")
        outputFiguresRelativePath.append(saveFiguresToPdf(pdfname=('scatterFaVsGpfaPopMetrics-%s and %s-col=%s-mrk=%s' % (brainAreas[0],brainAreas[-1],'bin','trlNeur'))))
        plt.close('all')

    return outputFiguresRelativePath

if __name__=='__main__':
    sweepOfExtractionParams()
