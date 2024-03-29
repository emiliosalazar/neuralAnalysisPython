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
from setup.DataJointSetup import * #DatasetInfo, DatasetGeneralLoadParams, BinnedSpikeSetInfo, FilterSpikeSetParams
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
from methods.BinnedSpikeSetListMethods import gpfaComputation, slowDriftComputation

# for descriptions of the data
from methods.BinnedSpikeSetListMethods import generateLabelGroupStatistics as genBSLLabGrp
from methods.BinnedSpikeSetListMethods import plotExampleChannelResponses
from methods.BinnedSpikeSetListMethods import plotStimDistributionHistograms
from methods.BinnedSpikeSetListMethods import plotFiringRates

# for computing metrics
from methods.BinnedSpikeSetListMethods import rscComputations, decodeComputations, informationComputations
from methods.GpfaMethods import computePopulationMetrics, computeProjectionMetrics, computeOverSessionMetrics

# for plotting the metrics
from methods.plotMethods.PopMetricsPlotMethods import plotAllVsAll, plotMetricVsExtractionParams, plotMetricsBySeparation
from methods.plotMethods.UnsortedPlotMethods import plotPointProjections


# @saveCallsToDatabase
def crossareaMatchedCovarianceComparison(datasetSqlFilter, binnedSpikeSetGenerationParamsDict, subsampleParams, gpfaParams, correlationParams,plotParams,binnedSpikeSetGenerationParamsDictBaseline = None):

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
    _, bssiKeys = genBSLAroundState(dsiUse,
                                        **binnedSpikeSetGenerationParamsDict
                                        )


    if binnedSpikeSetGenerationParamsDictBaseline is not None:
        _, bssiKeysBaseline = genBSLAroundState(dsiUse,
                                            **binnedSpikeSetGenerationParamsDictBaseline
                                            )


    # only doing one subsample for now...
    numSubsamples = subsampleParams['numSubsamples']
    # subsamples will have at least 60 neurons
    minNumNeurons = subsampleParams['minNumNeurons']


    extraOpts = subsampleParams['extraOpts']
    labelName = subsampleParams['labelName']



    combineConditions = gpfaParams['combineConditions']
    balanceConds = gpfaParams['balanceConds']
    if combineConditions:
        minNumTrlAll = subsampleParams['minNumTrlAll']
        if balanceConds:
            bssKeys = bsi[bssiKeys][fsp['ch_num>=%d AND condition_num*trial_num_per_cond_min>=%d' % (minNumNeurons, minNumTrlAll)]].fetch('KEY')
        else:
            bssKeys = bsi[bssiKeys][fsp['ch_num>=%d AND trial_num>=%d' % (minNumNeurons, minNumTrlAll)]].fetch('KEY')
        # fspNumInfo = fsp[bsi[bssKeys]].fetch('ch_num', 'trial_num_per_cond_min', 'trial_num', 'condition_num')
        fspNumInfo = list(zip(*[fsp[bsi[bK]].fetch('ch_num', 'trial_num_per_cond_min', 'trial_num', 'condition_num') for bK in bssKeys]))
        minTrlPerCond = np.stack(fspNumInfo[1]).squeeze()
        trialNum = np.stack(fspNumInfo[2]).squeeze()
        condNum = np.stack(fspNumInfo[3]).squeeze()

        maxNumTrlPerCond = minNumTrlAll/condNum
        if np.any(minTrlPerCond < maxNumTrlPerCond):
            if balanceConds:
                # should never be reached if balanceConds is True; otherwise,
                # might indicate that conditions weren't evenly presented, so
                # one was presented less than the average
                breakpoint() 
            else:
                breakpoint() # is this computing things right? Same number of output trials per session?
                # gotta change this to a list and an int to match what following
                # functions expect...
                maxNumTrlPerCond = list(minTrlPerCond.astype(int))
        else:
            # gotta change this to a list and an int to match what following
            # functions expect...
            if maxNumTrlPerCond.ndim == 0:
                maxNumTrlPerCond = maxNumTrlPerCond[None]
            maxNumTrlPerCond = list(maxNumTrlPerCond.astype(int))

        if np.min(fspNumInfo[0]) < minNumNeurons:
            breakpoint() # should never be reached, really
        else:
            maxNumNeuron = minNumNeurons # np.min(chTrlNumPerCond[0])
    else:
        minNumTrlPerCond = subsampleParams['minNumTrlPerCond']
        bssKeys = bsi[bssiKeys][fsp['ch_num>=%d AND trial_num_per_cond_min>=%d' % (minNumNeurons, minNumTrlPerCond)]].fetch('KEY')
        fspNumInfo = list(zip(*[fsp[bsi[bK]].fetch('ch_num', 'trial_num_per_cond_min', 'trial_num', 'condition_num') for bK in bssKeys]))
        # fspNumInfo = fsp[bsi[bssKeys]].fetch('ch_num', 'trial_num_per_cond_min', 'trial_num', 'condition_num')
        chTrlNumPerCond = fspNumInfo[:2]
        chTrlNumPerCond = [np.stack(ctnp).squeeze() for ctnp in chTrlNumPerCond]


        # now, in order to prevent small changes in what datasets get in making
        # everything get reextracted because of different neuron numbers, I'm
        # taking these as set by the input
        if np.min(chTrlNumPerCond[1]) < minNumTrlPerCond:
            breakpoint() # should never be reached, really
        else:
            maxNumTrlPerCond = np.min(chTrlNumPerCond[1])

        if np.min(chTrlNumPerCond[0]) < minNumNeurons:
            breakpoint() # should never be reached, really
        else:
            maxNumNeuron = minNumNeurons # np.min(chTrlNumPerCond[0])


    binnedResidShStOffSubsamples, subsampleExpressions, dsNames, brainAreas, tasks = subsmpMatchCond(bssKeys, maxNumTrlPerCond = maxNumTrlPerCond, maxNumNeuron = maxNumNeuron, labelName = labelName, numSubsamples = numSubsamples, extraOpts = extraOpts)

    if binnedSpikeSetGenerationParamsDict is not None:
        trlFiltAll = []
        chFiltAll = []
        subExpParents = []
        for subExp in subsampleExpressions:
            trlFiltRev = []
            chFiltRev = []
            while fsp[subExp]['parent_bss_relative_path'][0] is not None:
                trlFiltRev.append(fsp[subExp]['trial_filter'][0])
                chFiltRev.append(fsp[subExp]['ch_filter'][0])
                subExp = bsi['bss_relative_path="{}"'.format(fsp[subExp]['parent_bss_relative_path'][0])]
            subExpParents.append(subExp) # wait isn't this just bssi keys from above?
            trlFiltAll.append(trlFiltRev)
            chFiltAll.append(chFiltRev)


    print("Computing GPFA")
    """ These are the GPFA parameters """
    iterateParams = gpfaParams.pop('iterateParams', None)
    if iterateParams:
        from itertools import product
        iterateSeparated = [[{k:vind} for vind in v] for k,v in iterateParams.items()]
        paramIterator = list(product(*iterateSeparated)) # making a list so it has a length
    else:
        paramIterator = [gpfaParams]


    bnSpOut, dims, dimsLL, gpfaDimOut, gpfaTestInds, gpfaTrainInds, gpfaCondLabels, brainAreasResiduals, brainAreasNoResiduals = [],[],[],[],[], [], [], [], []
    bnSpOutBaseline = []
    for paramSet in paramIterator:
        # we'll note that this basically means the for loop is for show when
        # we're not iterating
        if iterateParams:
            [gpfaParams.update(paramVal) for paramVal in paramSet]
            descStrName = ['{}={}'.format(k,v) for paramVal in paramSet for k,v in paramVal.items() ]
        else:
            descStrName = ""

        # ** actually run GPFA **
        dimsH, dimsLLH, gpfaDimOutH, gpfaTestIndsH, gpfaTrainIndsH, *_, gpfaCondLabelsH = gpfaComputation(
            subsampleExpressions, **gpfaParams
        )

        dims += dimsH
        dimsLL += dimsLLH
        gpfaDimOut += gpfaDimOutH
        gpfaTestInds += gpfaTestIndsH
        gpfaTrainInds += gpfaTrainIndsH
        gpfaCondLabels += gpfaCondLabelsH

        brainAreasResiduals += [bA + ' {}'.format(','.join(descStrName)) for bA in brainAreas]
        brainAreasNoResiduals += brainAreas

        bnSpSubsmp = []
        bnSpSubsmpBaseline = []
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
                
                if binnedSpikeSetGenerationParamsDictBaseline is not None:
                    chansUse = fsp[subsampleExpressions[indAr]]['ch_filter'][0]
                    bssiKeyUse = bssiKeysBaseline[int(np.array([bs['dataset_relative_path']==subsampleExpressions[indAr]['dataset_relative_path'] for bs in bssiKeysBaseline]).nonzero()[0])]
                    bnSpSubsmpBaseline.append([bsi[bssiKeyUse].grabBinnedSpikes()[0][:,chansUse]])
        else:
            for indAr, (bnSpArea) in enumerate(binnedResidShStOffSubsamples):
                bnSpSubsmp.append(bnSpArea)

                if binnedSpikeSetGenerationParamsDictBaseline is not None:
                    chansUse = fsp[subsampleExpressions[indAr]]['ch_filter'][0]
                    bssiKeyUse = bssiKeysBaseline[int(np.array([bs['dataset_relative_path']==subsampleExpressions[indAr]['dataset_relative_path'] for bs in bssiKeysBaseline]).nonzero()[0])]
                    bnSpSubsmpBaseline.append([bsi[bssiKeyUse].grabBinnedSpikes()[0][:,chansUse]])

        bnSpOut += bnSpSubsmp
        bnSpOutBaseline += bnSpSubsmpBaseline


    brainAreas = brainAreasResiduals
    subsampleExpressions*=len(paramIterator)
    dsNames*=len(paramIterator)
    tasks*=len(paramIterator)
    binnedResidShStOffSubsamples = bnSpOut
    binnedResidShStOffBaselineSubsamples = bnSpOutBaseline


    if plotParams and plotParams['plotGpfa']:
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

    if plotParams and plotParams['plotGenericMetrics']:
        plotGenericMetricParams = plotParams['plotGenericMetrics']
        supTitle = plotGenericMetricParams['supTitle']

        plotStimDistributionHistograms(binnedSpksShortStOffSubsamplesFR, descriptions)
        plotFiringRates(binnedSpksShortStOffSubsamplesFR, descriptions, supTitle = supTitle + " firing rates", cumulative = False)

#        plotStimDistributionHistograms(binnedSpksShortStOffSubsamplesCnt, descriptions)
        plotFiringRates(binnedSpksShortStOffSubsamplesCnt, descriptions, supTitle = supTitle + " count", cumulative = False)


        outputFiguresRelativePath.append(saveFiguresToPdf(pdfname="{}{}".format(plotGenericMetricParams['pdfnameSt'],plotParams['analysisIdentifier'])))
        plt.close('all')

    # Population metrics
    popMetricsDict = computePopulationMetrics(gpfaDimOut, dimsLL, dims, binnedSpikeSetGenerationParamsDict['binSizeMs'], subsampleExpressions)

    # Over session metrics
    # overSessionMetricsDict = computeOverSessionMetrics(subsampleExpressions, gpfaDimOut, dimsLL, gpfaTestInds, gpfaCondLabelsH, labelNameUse = gpfaParams['labelUse'])

    # Projection metrics
    # projMetricsDict, projPtsForPlotDict = computeProjectionMetrics(binnedResidShStOffSubsamples, binnedResidShStOffBaselineSubsamples, gpfaDimOut, dimsLL, gpfaTestInds, gpfaParams)

    if plotParams and plotParams['plotProjections']:
        plotProjParams = plotParams['plotProjections']
        projOnSignalZscByExParams = projPtsForPlotDict['data into top two signal PCs']
        meanOnSignalByExParams = projPtsForPlotDict['mean into top two signal PCs']
        latentsOnSignalZscByExParams = projPtsForPlotDict['noise latents into signal PCs']
        plotPointProjections(projOnSignalZscByExParams, meanOnSignalByExParams, latentsOnSignalZscByExParams, descriptions)
        outputFiguresRelativePath.append(saveFiguresToPdf(pdfname="{}{}".format(plotProjParams['pdfnameSt'],plotParams['analysisIdentifier'])))
        plt.close('all')



    #%% Residual correlations
    plotResid = correlationParams['plotResid']
    rscMetricDict = rscComputations(binnedResidShStOffSubsamples, descriptions, labelName, **correlationParams)



    if plotParams and plotParams['plotResid']:
        plotResidParams = plotParams['plotResid']
        outputFiguresRelativePath.append(saveFiguresToPdf(pdfname="{}{}".format(plotResidParams['pdfnameSt'],plotParams['analysisIdentifier'])))
        plt.close('all')

    # Decode metrics
    # decodeDict = decodeComputations(binnedResidShStOffSubsamples, descriptions, labelName)

    # Info metrics
    # infoDict = informationComputations(binnedResidShStOffSubsamples, labelName)
    # from numpy.polynomial import Polynomial
    # infoDict.update({
    #     'fisher mn' : [np.mean(iC, axis=0) for shOnMnDff, iC in zip(projMetricsDict['(proj noise from 1st latent)/(cond diff mag)'], infoDict['fisher information'])],
    #     'linear slope b/t fisher vs (sh var on mn diff)' : [np.array(Polynomial.fit(iC.squeeze(),shOnMnDff.squeeze(), 1).coef[1])[None] for shOnMnDff, iC in zip(projMetricsDict['(proj noise from 1st latent)/(cond diff mag)'], infoDict['fisher information'])],
    # })
    # infoDict.update({
    #     'fisher mn / (linear slope b/t fisher vs (sh var on mn diff))' : [rel/inf for inf,rel in zip(infoDict['fisher mn'], infoDict['fisher / (noise-to-mean diff)'])],
    # })

    metricDict = {}
    # metricDict.update(overSessionMetricsDict)
    # metricDict.update(projMetricsDict)
    metricDict.update(popMetricsDict)
    # metricDict.update(decodeDict)
    # metricDict.update(infoDict)
    metricDict.update(rscMetricDict)




#    descriptions = dsNames*len(computeResidualsLog)
 #[data[idx]['description'] for idx in dataIndsProcess]
    if plotParams and plotParams['plotScatterMetrics']:
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
