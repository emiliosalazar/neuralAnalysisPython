"""
This function is responsible for the cross-area analyses/scatter matrices etc
Like HeadlessAnalysisRun2-5.py for now...
"""
from matplotlib import pyplot as plt
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['GOTO_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
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
from methods.plotUtils.UnsortedPlotMethods import plotAllVsAll, plotMetricVsExtractionParams, plotMetricsBySeparation


@saveCallsToDatabase
def crossareaMatchedCovarianceComparison(datasetSqlFilter, binnedSpikeSetGenerationParamsDict, subsampleParams, gpfaParams, correlationParams,plotParams):

    dsgl = DatasetGeneralLoadParams()
    dsi = DatasetInfo()
    bsi = BinnedSpikeSetInfo()
    fsp = FilterSpikeSetParams()

    outputFiguresRelativePath = []

    if len(datasetSqlFilter)>0:
        dsiUse = dsi[datasetSqlFilter]
    else:
        dsiUse = dsi

    print("Computing/loading binned spike sets")
    offshift = binnedSpikeSetGenerationParamsDict.pop('offshift')
    _, bssiKeys = genBSLAroundState(dsiUse,
                                        **binnedSpikeSetGenerationParamsDict
                                        )


    # only doing one subsample for now...
    numSubsamples = subsampleParams['numSubsamples']
    # subsamples will have at least 60 neurons
    minNumNeurons = subsampleParams['minNumNeurons']


    extraOpts = subsampleParams['extraOpts']
    labelName = 'stimulusMainLabel'



    combineConditions = gpfaParams['combineConditions']
    if combineConditions:
        minNumTrlAll = subsampleParams['minNumTrlAll']
        bssKeys = bsi[bssiKeys][fsp['ch_num>=%d AND condition_num*trial_num_per_cond_min>=%d' % (minNumNeurons, minNumTrlAll)]].fetch('KEY')
        fspNumInfo = fsp[bsi[bssKeys]].fetch('ch_num', 'trial_num_per_cond_min', 'trial_num', 'condition_num')
        minTrlPerCond = fspNumInfo[1]
        trialNum = fspNumInfo[2]
        condNum = fspNumInfo[3]

        maxNumTrlPerCond = minNumTrlAll/condNum
        if np.any(minTrlPerCond < maxNumTrlPerCond):
            breakpoint() # should never be reached, really
        else:
            # gotta change this to a list and an int to match what following
            # functions expect...
            maxNumTrlPerCond = list(maxNumTrlPerCond.astype(int))

        if np.min(fspNumInfo[0]) < minNumNeurons:
            breakpoint() # should never be reached, really
        else:
            maxNumNeuron = minNumNeurons # np.min(chTrlNumPerCond[0])
    else:
        minNumTrlPerCond = subsampleParams['minNumTrlPerCond']
        bssKeys = bsi[bssiKeys][fsp['ch_num>=%d AND trial_num_per_cond_min>=%d' % (minNumNeurons, minNumTrlPerCond)]].fetch('KEY')
        fspNumInfo = fsp[bsi[bssKeys]].fetch('ch_num', 'trial_num_per_cond_min', 'trial_num', 'condition_num')
        chTrlNumPerCond = fspNumInfo[:2]


        # now, in order to prevent small changes in what datasets get in making
        # everything get reextracted because of different neuron numbers, I'm
        # taking these as set by the input
        if np.min(chTrlNumPerCond[1]) < minNumTrlPerCond:
            breakpoint() # should never be reached, really
        else:
            maxNumTrlPerCond = minNumTrlPerCond 

        if np.min(chTrlNumPerCond[0]) < minNumNeurons:
            breakpoint() # should never be reached, really
        else:
            maxNumNeuron = minNumNeurons # np.min(chTrlNumPerCond[0])


    binnedResidShStOffSubsamples, subsampleExpressions, dsNames, brainAreas, tasks = subsmpMatchCond(bssKeys, maxNumTrlPerCond = maxNumTrlPerCond, maxNumNeuron = maxNumNeuron, labelName = labelName, numSubsamples = numSubsamples, extraOpts = extraOpts)

    print("Computing GPFA")
    """ These are the GPFA parameters """
    iterateParams = gpfaParams.pop('iterateParams', None)
    if iterateParams:
        from itertools import product
        iterateSeparated = [[{k:vind} for vind in v] for k,v in iterateParams.items()]
        paramIterator = list(product(*iterateSeparated)) # making a list so it has a length
    else:
        paramIterator = [gpfaParams]


    bnSpOut, dims, dimsLL, gpfaDimOut, gpfaTestInds, gpfaTrainInds, brainAreasResiduals, brainAreasNoResiduals = [],[],[],[],[], [], [], []
    for paramSet in paramIterator:
        # we'll note that this basically means the for loop is for show when
        # we're not iterating
        if iterateParams:
            [gpfaParams.update(paramVal) for paramVal in paramSet]
            descStrName = ['{}={}'.format(k,v) for paramVal in paramSet for k,v in paramVal.items() ]
        else:
            descStrName = ""

        dimsH, dimsLLH, gpfaDimOutH, gpfaTestIndsH, gpfaTrainIndsH = gpfaComputation(
            subsampleExpressions, **gpfaParams
        )

        dims += dimsH
        dimsLL += dimsLLH
        gpfaDimOut += gpfaDimOutH
        gpfaTestInds += gpfaTestIndsH
        gpfaTrainInds += gpfaTrainIndsH

        brainAreasResiduals += [bA + ' {}'.format(','.join(descStrName)) for bA in brainAreas]
        brainAreasNoResiduals += brainAreas

        bnSpSubsmp = []
        if iterateParams:
            for indAr, (bnSpArea) in enumerate(binnedResidShStOffSubsamples):
                paramsIterated = {k : v for paramVal in paramSet for k,v in paramVal.items()}
                # only do this step if you're iterating params AND
                # computeResiduals is one of them AND the residuals were
                # actually computed
                if paramsIterated.pop('computeResiduals',None): 
                    bnSpSubsmp.append([bnSp.baselineSubtract(labels=bnSp.labels[gpfaParams['labelUse']])[0] for bnSp in bnSpArea])
                else:
                    bnSpSubsmp.append(bnSpArea)
        else:
            for indAr, (bnSpArea) in enumerate(binnedResidShStOffSubsamples):
                bnSpSubsmp.append(bnSpArea)

        bnSpOut += bnSpSubsmp


    brainAreas = brainAreasResiduals
    subsampleExpressions*=len(paramIterator)
    dsNames*=len(paramIterator)
    tasks*=len(paramIterator)
    binnedResidShStOffSubsamples = bnSpOut

    if plotParams['plotGpfa']:
        plotGpfaParams = plotParams['plotGpfa']
        outputFiguresRelativePath.append(saveFiguresToPdf(pdfname=("{}{}Neur{}Trl{}".format(plotGpfaParams['pdfnameSt'],plotParams['analysisIdentifier'], maxNumNeuron, maxNumTrlPerCond))))
        plt.close('all')

    
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

    if plotParams['plotGenericMetrics']:
        plotGenericMetricParams = plotParams['plotGenericMetrics']
        supTitle = plotGenericMetricParams['supTitle']

        plotStimDistributionHistograms(binnedSpksShortStOffSubsamplesFR, descriptions)
        plotFiringRates(binnedSpksShortStOffSubsamplesFR, descriptions, supTitle = supTitle + " firing rates", cumulative = False)

        plotStimDistributionHistograms(binnedSpksShortStOffSubsamplesCnt, descriptions)
        plotFiringRates(binnedSpksShortStOffSubsamplesCnt, descriptions, supTitle = supTitle + " count", cumulative = False)


        outputFiguresRelativePath.append(saveFiguresToPdf(pdfname="{}{}".format(plotGenericMetricParams['pdfnameSt'],plotParams['analysisIdentifier'])))
        plt.close('all')

    # Population metrics
    popMetricsDict = computePopulationMetrics(gpfaDimOut, dimsLL, dims, binnedSpikeSetGenerationParamsDict['binSizeMs'])

    #%% Residual correlations
    plotResid = correlationParams['plotResid']
    rscMetricDict = rscComputations(binnedResidShStOffSubsamples, descriptions, labelName, **correlationParams)


    if plotParams['plotResid']:
        plotResidParams = plotParams['plotResid']
        outputFiguresRelativePath.append(saveFiguresToPdf(pdfname="{}{}".format(plotResidParams['pdfnameSt'],plotParams['analysisIdentifier'])))
        plt.close('all')

    metricDict = {}
    metricDict.update(popMetricsDict)
    metricDict.update(rscMetricDict)




#    descriptions = dsNames*len(computeResidualsLog)
 #[data[idx]['description'] for idx in dataIndsProcess]
    if plotParams['plotScatterMetrics']:
        plotAllVsAllParams = plotParams['plotScatterMetrics']['plotAllVsAllParams']
        plotMetVsExtPrmParams = plotParams['plotScatterMetrics']['plotMetVsExtPrmParams']
        plotMetBySeparation = plotParams['plotScatterMetrics']['plotMetBySeparation']

        if len(plotAllVsAllParams) > 0:
            labelForColor = plotAllVsAllParams['labelForColor']
            if type(labelForColor) is str:
                labelForColor = np.array(eval(labelForColor))

            labelForMarkers = plotAllVsAllParams['labelForMarkers']
            if type(labelForMarkers) is str:
                labelForMarkers = np.array(eval(labelForMarkers))

            plotAllVsAll(descriptions, metricDict, labelForColor, labelForMarkers)
            outputFiguresRelativePath.append(saveFiguresToPdf(pdfname='{}{}'.format(plotAllVsAllParams['pdfnameSt'],plotParams['analysisIdentifier'])))
            plt.close('all')

        
        if len(plotMetVsExtPrmParams) > 0:
            splitNameDesc = plotMetVsExtPrmParams['splitNameDesc']
            labelForSplit = plotMetVsExtPrmParams['labelForSplit']
            labelForPair = plotMetVsExtPrmParams['labelForPair']
            labelForMarkers = plotMetVsExtPrmParams['labelForMarkers']
            labelForColor = plotMetVsExtPrmParams['labelForColor']

            locCopy = locals().copy()
            locPlusGlob = locCopy.update(globals())
            if type(labelForSplit) is str:
                labelForSplit = np.array(eval(labelForSplit, locCopy))
            else:
                labelForSplit = np.repeat(labelForSplit, len(subsampleExpressions)/len(labelForSplit))
                labelForSplit = labelForSplit.astype(int)

            if type(labelForPair) is str:
                labelForPair = np.array(eval(labelForPair, locCopy))
            else:
                breakpoint()
                # is this *really* what you want to do?
                labelForPair = np.repeat(labelForPair, 2)


            if type(labelForMarkers) is str:
                labelForMarkers = np.array(eval(labelForMarkers))
            else:
                labelForMarkers = np.repeat(labelForMarkers, len(subsampleExpressions)/len(labelForMarkers))

            if type(labelForColor) is str:
                labelForColor = np.array(eval(labelForColor))
            else:
                labelForColor = np.repeat(labelForColor, len(subsampleExpressions)/len(labelForColor))

            plotMetricVsExtractionParams(descriptions, metricDict, splitNameDesc, labelForPair, labelForSplit, labelForColor, labelForMarkers, supTitle="")

            outputFiguresRelativePath.append(saveFiguresToPdf(pdfname='{}{}'.format(plotMetVsExtPrmParams['pdfnameSt'],plotParams['analysisIdentifier'])))
            plt.close('all')

        if len(plotMetBySeparation) > 0:
            separationName = plotMetBySeparation['separationName']
            labelForSeparation = plotMetBySeparation['labelForSeparation']
            labelForMarkers = plotMetBySeparation['labelForMarkers']
            labelForColors = plotMetBySeparation['labelForColors']

            if type(labelForSeparation) is str:
                labelForSeparation = np.array(eval(labelForSeparation))
            else:
                labelForSeparation = np.repeat(labelForSeparation, len(subsampleExpressions)/len(labelForSeparation))
                labelForSeparation = labelForSeparation.astype(int)

            if type(labelForMarkers) is str:
                labelForMarkers = np.array(eval(labelForMarkers))
            else:
                labelForMarkers = np.repeat(labelForMarkers, len(subsampleExpressions)/len(labelForMarkers))

            if type(labelForColors) is str:
                labelForColors = np.array(eval(labelForColors))
            else:
                labelForColors = np.repeat(labelForColors, len(subsampleExpressions)/len(labelForColors))


            plotMetricsBySeparation(metricDict, descriptions, separationName, labelForSeparation, labelForColors, labelForMarkers, supTitle = '')
            outputFiguresRelativePath.append(saveFiguresToPdf(pdfname='{}{}'.format(plotMetBySeparation['pdfnameSt'],plotParams['analysisIdentifier'])))
            plt.close('all')

    outputInfo = {}
    outputInfo.update(dict(outputFiguresRelativePath = outputFiguresRelativePath)) if len(outputFiguresRelativePath)>0 else None

    outputInfo.update(dict(
        allMetrics = metricDict,
        groupingInfo = dict(
            descriptions = descriptions,
            brainAreas = brainAreas,
            tasks = tasks,
        ),
        subsamplesUsed = subsampleExpressions,
        )
    )

    return outputInfo


if __name__=='__main__':
    crossareaMatchedCovarianceComparison()
