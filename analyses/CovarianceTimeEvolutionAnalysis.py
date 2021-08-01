"""
This function is responsible for looking at the evolution of metrics over
certain time periods and plotting them!
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

# for computing metrics
from methods.BinnedSpikeSetListMethods import rscComputations, decodeComputations, informationComputations
from methods.GpfaMethods import computePopulationMetrics, computeProjectionMetrics

# for plotting the metrics
from methods.plotUtils.PopMetricsPlotMethods import plotTimeEvolution


@saveCallsToDatabase
def covarianceTimeEvolutionAnalysis(datasetSqlFilter, timeShifts, binnedSpikeSetGenerationParamsDict, subsampleParams, gpfaParams, correlationParams,plotParams):

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
    bssiKeysTimeShift = []
    segmentLengthMs = binnedSpikeSetGenerationParamsDict.pop('segmentLengthMs')
    for timeShift in timeShifts:
        furthestTimeBeforeState = int(-timeShift) # need the minus sign because it's the *positive* time before the state
        furthestTimeAfterState = int(timeShift) + segmentLengthMs + 1 # add 1 to make segment inclusive of last value
        _, bssiKeys = genBSLAroundState(dsiUse,
                                            **binnedSpikeSetGenerationParamsDict,
                                            furthestTimeBeforeState = furthestTimeBeforeState,
                                            furthestTimeAfterState = furthestTimeAfterState
                                            )
        bssiKeysTimeShift.append(bssiKeys)

    # subsamples will have at least this many neurons
    minNumNeurons = subsampleParams['minNumNeurons']

    combineConditions = gpfaParams['combineConditions']
    balanceConds = gpfaParams['balanceConds']
    if combineConditions:
        minNumTrlAll = subsampleParams['minNumTrlAll']
        if balanceConds:
            # bssKeys = bsi[bssiKeysTimeShift][fsp['ch_num>=%d AND condition_num*trial_num_per_cond_min>=%d' % (minNumNeurons, minNumTrlAll)]].fetch('KEY')
            bssKeys = [bsi[bssTSOneKey][fsp['ch_num>=%d AND condition_num*trial_num_per_cond_min>=%d' % (minNumNeurons, minNumTrlAll)]].fetch('KEY')[0] for bssTS in bssiKeysTimeShift for bssTSOneKey in bssTS]
        else:
            # bssKeys = bsi[bssiKeysTimeShift][fsp['ch_num>=%d AND trial_num>=%d' % (minNumNeurons, minNumTrlAll)]].fetch('KEY')
            bssKeys = [bsi[bssTSOneKey][fsp['ch_num>=%d AND trial_num>=%d' % (minNumNeurons, minNumTrlAll)]].fetch('KEY')[0] for bssTS in bssiKeysTimeShift for bssTSOneKey in bssTS]
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
                # gotta change this to a list and an int to match what following
                # functions expect...
                maxNumTrlPerCond = list(minTrlPerCond.astype(int))
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
        bssKeys = [bsi[bssTSOneKey][fsp['ch_num>=%d AND trial_num_per_cond_min>=%d' % (minNumNeurons, minNumTrlPerCond)]].fetch('KEY')[0] for bssTS in bssiKeysTimeShift for bssTSOneKey in bssTS]
        fspNumInfo = list(zip(*[fsp[bsi[bK]].fetch('ch_num', 'trial_num_per_cond_min', 'trial_num', 'condition_num') for bK in bssKeys]))
        chTrlNumPerCond = fspNumInfo[:2]
        chTrlNumPerCond = [np.stack(ctnp).squeeze() for ctnp in chTrlNumPerCond]


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


    breakpoint()
    timeShiftMetricDict = {}

    # only doing one subsample for now...
    numSubsamples = subsampleParams['numSubsamples']
    extraOpts = subsampleParams['extraOpts']
    labelName = subsampleParams['labelName']

    for timeShift, bssiKeys in zip(timeShifts, bssiKeysTimeShift):

        # gotta make sure there are enough channels here or weird things
        # happen...
        if combineConditions:
            if balanceConds:
                bssKeysThisTimeShift = bsi[bssiKeys][fsp['ch_num>=%d AND condition_num*trial_num_per_cond_min>=%d' % (minNumNeurons, minNumTrlAll)]].fetch('KEY')
            else:
                bssKeysThisTimeShift = bsi[bssiKeys][fsp['ch_num>=%d AND trial_num>=%d' % (minNumNeurons, minNumTrlAll)]].fetch('KEY')
        else:
            bssKeysThisTimeShift = bsi[bssiKeys][fsp['ch_num>=%d AND trial_num_per_cond_min>=%d' % (minNumNeurons, minNumTrlPerCond)]].fetch('KEY')

        binnedResidShStOffSubsamples, subsampleExpressions, dsNames, brainAreas, tasks = subsmpMatchCond(bssKeysThisTimeShift, maxNumTrlPerCond = maxNumTrlPerCond, maxNumNeuron = maxNumNeuron, labelName = labelName, numSubsamples = numSubsamples, extraOpts = extraOpts)

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

            if timeShift==0:
                breakpoint()
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

        # Projection metrics
        projMetricsDict, projPtsForPlotDict = computeProjectionMetrics(binnedResidShStOffSubsamples, gpfaDimOut, dimsLL, gpfaTestInds, gpfaParams)
            
        # Population metrics
        # popMetricsDict = computePopulationMetrics(gpfaDimOut, dimsLL, dims, binnedSpikeSetGenerationParamsDict['binSizeMs'])

        #%% Residual correlations
        plotResid = correlationParams['plotResid']
        # rscMetricDict = rscComputations(binnedResidShStOffSubsamples, descriptions, labelName, **correlationParams)

        if plotParams['plotResid']:
            plotResidParams = plotParams['plotResid']
            outputFiguresRelativePath.append(saveFiguresToPdf(pdfname="{}{}".format(plotResidParams['pdfnameSt'],plotParams['analysisIdentifier'])))
            plt.close('all')

        # Decode metrics
        # decodeDict = decodeComputations(binnedResidShStOffSubsamples, descriptions, labelName)

        # Info metrics
        infoDict = informationComputations(binnedResidShStOffSubsamples, labelName)
        from numpy.polynomial import Polynomial
        infoDict.update({
            'fisher mn' : [np.mean(iC) for shOnMnDff, iC in zip(projMetricsDict['log((cond diff mag)/(proj noise from 1st latent))'], infoDict['fisher information'])],
            'fisher / (noise-to-mean diff)' : [Polynomial.fit(iC.squeeze(),shOnMnDff.squeeze(), 1).coef[1] for shOnMnDff, iC in zip(projMetricsDict['log((cond diff mag)/(proj noise from 1st latent))'], infoDict['fisher information'])],
        })
        infoDict.update({
            'fisher mn / (linear slope b/t fisher vs ns-mn)' : [rel/inf for inf,rel in zip(infoDict['fisher mn'], infoDict['fisher / (noise-to-mean diff)'])],
        })

        metricDict = {}
        # metricDict.update(projMetricsDict)
        # metricDict.update(popMetricsDict)
        # metricDict.update(rscMetricDict)
        metricDict.update(infoDict)
        # metricDict.update(decodeDict)

        timeShiftMetricDict[timeShift] = metricDict
        timeShiftMetricDict[timeShift]['timeCenterPoint'] = timeShift+(segmentLengthMs)/2 

#    descriptions = dsNames*len(computeResidualsLog)
     #[data[idx]['description'] for idx in dataIndsProcess]
    if plotParams['plotTimeEvolution']:
        plotTmEvParams = plotParams['plotTimeEvolution']
        if isinstance(plotTmEvParams['labelForMarkers'], str):
            labelForMarkers = eval(plotTmEvParams['labelForMarkers'])
        else:
            labelForMarkers = plotTmEvParams['labelForMarkers']
        
        if isinstance(plotTmEvParams['labelForColors'], str):
            labelForColors = eval(plotTmEvParams['labelForColors'])
        else:
            labelForColors = plotTmEvParams['labelForColors']

        plotTimeEvolution(descriptions, timeShiftMetricDict, labelForMarkers = labelForMarkers, labelForColors = labelForColors)
        outputFiguresRelativePath.append(saveFiguresToPdf(pdfname='{}{}'.format(plotTmEvParams['pdfnameSt'],plotParams['analysisIdentifier'])))
        plt.close('all')

    outputInfo = {}
    outputInfo.update(dict(outputFiguresRelativePath = outputFiguresRelativePath)) if len(outputFiguresRelativePath)>0 else None

    return outputInfo


if __name__=='__main__':
    covarianceTimeEvolutionAnalysis()
