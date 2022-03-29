"""
This function is responsible for the cross-area analyses/scatter matrices etc
Like HeadlessAnalysisRun2-5.py for now...
"""
from matplotlib import pyplot as plt
import plotly as ply
import plotly.express as px 

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
def internalSignalsAnalysis(datasetSqlFilter, binnedSpikeSetGenerationParamsDict, subsampleParams, gpfaParams, correlationParams,plotParams,binnedSpikeSetGenerationParamsDictBaseline = None):

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
            # append the last filters
            trlFiltRev.append(fsp[subExp]['trial_filter'][0])
            chFiltRev.append(fsp[subExp]['ch_filter'][0])

            subExpParents.append(subExp) # wait isn't this just bssi keys from above?
            trlFiltAll.append(trlFiltRev)
            chFiltAll.append(chFiltRev)

        chFiltComplete = []
        trlFiltComplete = []
        for chFilts, trlFilts in zip(chFiltAll, trlFiltAll):
            chFiltFromOrig = np.arange(chFilts[-1].shape[0])
            trlFiltFromOrig = np.arange(trlFilts[-1].shape[0])
            for chFiltSteps in chFilts[::-1]:
                chFiltFromOrig = chFiltFromOrig.squeeze()[chFiltSteps]
                # not sure about that squeeze above...
                # chFiltFromOrig = chFiltFromOrig[chFiltSteps]
            chFiltComplete.append(chFiltFromOrig)
            for trlFiltSteps in trlFilts[::-1]:
                trlFiltFromOrig = trlFiltFromOrig[trlFiltSteps]
            trlFiltComplete.append(trlFiltFromOrig)



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
        bssSpikesRef = [subExp.grabBinnedSpikes()[0] for subExp in subsampleExpressions]
        # dimsHSpks, dimsLLHSpks, gpfaDimOutHSpks, gpfaTestIndsHSpks, gpfaTrainIndsHSpks, *_, gpfaCondLabelsHSpks = gpfaComputation(
        #     [subsampleExpressions[0], bssSpikesRef[0]], relativePathsToSave = [savePths[0],savePths[0]], **gpfaParams
        # )
        # breakpoint()
        # dimsHPars, dimsLLHPars, gpfaDimOutHPars, gpfaTestIndsHPars, gpfaTrainIndsHPars, *_, gpfaCondLabelsHPars = gpfaComputation(
        #     subExpParents, **gpfaParams
        # )


        prepInputKeywords = ['sqrtSpikes', 'labelUse', 'condNums', 'combineConds', 'overallFiringRateThresh', 'perConditionGroupFiringRateThresh', 'balanceConds', 'computeResiduals']
        preprocessInputsFinal = {}
        preprocessInputs = gpfaParams
        [preprocessInputsFinal.update({pIK : preprocessInputs[pIK]}) for pIK in prepInputKeywords if pIK in preprocessInputs]
        preprocessInputsFinal['condNums'] = preprocessInputs['conditionNumbers']
        preprocessInputsFinal['combineConds'] = preprocessInputs['combineConditions']
        bssRefChKeep = [bsR.prepareGpfaOrFa(**preprocessInputsFinal)[-1]['channelsKeep'] for bsR in bssSpikesRef]
        bssSpikesRef = [bsR.prepareGpfaOrFa(**preprocessInputsFinal)[0][0] for bsR in bssSpikesRef]
        # chFiltRefFrom = [subExp.trialAndChannelFilterFromParent(subExpP)[1][chRestriction] for subExp, subExpP, chRestriction in zip(subsampleExpressions, subExpParents, bssRefChKeep)]
        # bssParWithCorrCh = [subExp.grabBinnedSpikes()[0][:,chFilt] for subExp, chFilt in zip(subExpParents, chFiltRefFrom)]
        # bssSpikesRef = bssParWithCorrCh
        # bssSpikesRef = [subExp.grabBinnedSpikes()[0][:,chFilt] for subExp, chFilt in zip(subsampleExpressions, bssRefChKeep)]
        # bssSpikesRef = [bSR.baselineSubtract(labels=bSR.labels['stimulusMainLabel'])[0] for bSR in bssSpikesRef]

        # savePths = [Path('fromBss') / 'bss_{}'.format(hashlib.md5(subExp['bss_relative_path'][0].encode('ascii')).hexdigest()[:5]) / 'matchParNeurs' for subExp in subsampleExpressions]
        # dimsHParsMatchNeur, dimsLLHParsMatchNeur, gpfaDimOutHParsMatchNeur, gpfaTestIndsHParsMatchNeur, gpfaTrainIndsHParsMatchNeur, *_, gpfaCondLabelsHParsMatchNeur = gpfaComputation(
        #             bssParWithCorrCh, relativePathsToSave = savePths, **gpfaParams
        #         )

        numTrlsParents = np.stack([fsp[sE]['trial_num'] for sE in subExpParents])


        # slowDriftDirs, slowDriftDict, slowDriftSpikes, slowDriftRASpikes = slowDriftComputation(subsampleExpressions, preprocessInputs = gpfaParams, dimResultsAll = gpfaDimOutH, testInds = gpfaTestIndsH, plotOutput=False)
        slowDriftDirs, slowDriftDict, slowDriftSpikes, slowDriftRASpikes, spikesSansSlowDrift, spikesSansFirstDimSlowDrift = slowDriftComputation(subsampleExpressions, preprocessInputs = gpfaParams, dimResultsAll = gpfaDimOutH, testInds = gpfaTestIndsH, plotOutput={'title' : 'matched neuron/trial subset'})
        # slowDriftDirsPars, slowDriftDictPars, slowDriftParsSpikes, slowDriftParsRASpikes = slowDriftComputation(subExpParents, preprocessInputs = gpfaParams, dimResultsAll = gpfaDimOutHPars, testInds = gpfaTestIndsHPars, plotOutput={'title' : 'all neurons and trials'})
        # slowDriftDirsParsSameNeurs, slowDriftDictParsSameNeurs, slowDriftParsSameNeursSpikes, slowDriftParsSameNeursRASpikes = slowDriftComputation(subExpParents, preprocessInputs = gpfaParams, dimResultsAll = gpfaDimOutHPars, testInds = gpfaTestIndsHPars, plotOutput=False, chRestrictRefExp = subsampleExpressions)
        # dictListExpAndBss = [dict(subExp=sE,spikes=bss) for sE, bss in zip(subExpParents, bssParWithCorrCh)]
        # slowDriftDirsParsSameNeursFA, slowDriftDictParsSameNeursFA, slowDriftParsSameNeursFASpikes, slowDriftParsSameNeursRASpikes = slowDriftComputation(dictListExpAndBss, preprocessInputs = gpfaParams, dimResultsAll = gpfaDimOutHParsMatchNeur, testInds = gpfaTestIndsHParsMatchNeur, plotOutput={'title' : 'matched neurons, all trials'})
        # slowDriftDirComparison = np.stack([(np.arccos(sDP[cF].T @ sD)/np.pi*180).squeeze() if sDP[cF].size==sD.size else np.nan for sDP, sD, cF in zip(slowDriftDirsPars, slowDriftDirs, chFiltComplete)])
        # slowDriftDirComparisonSameNeurs = np.stack([(np.arccos(sDP.T @ sD)/np.pi*180).squeeze() if sDP.size==sD.size else np.nan for sDP, sD in zip(slowDriftDirsParsSameNeurs, slowDriftDirs)])
        # slowDriftDirComparisonMatchedOrDiffNeurExtract = np.stack([(np.arccos(sDP[cF].T @ sD)/np.pi*180).squeeze() if sDP[cF].size==sD.size else np.nan for sDP, sD, cF in zip(slowDriftDirsPars, slowDriftDirsParsSameNeurs, chFiltComplete)])
        # slowDriftDirComparisonSameNeursFA = np.stack([(np.arccos(sDP.T @ sD)/np.pi*180).squeeze() if sDP.size==sD.size else np.nan for sDP, sD in zip(slowDriftDirsParsSameNeursFA, slowDriftDirs)])

        # dsetTry = -1;
        # sDD = slowDriftDirs[dsetTry]
        # sDS = slowDriftSpikes[dsetTry]
        # sDAS = slowDriftRASpikes[dsetTry]
        # slowDriftProjPar = [(sDAS @ sDD).squeeze() for sDAS, sDD in zip(slowDriftParsSpikes, slowDriftDirsPars)]
        # slowDriftProjRA = [(sDAS @ sDD).squeeze() for sDAS, sDD in zip(slowDriftRASpikes, slowDriftDirs)]
        # meanAtt = [sDP.mean() for sDP in slowDriftProjPar]
        # locGreaterThanMeanAtt = [(sDS @ sDD).squeeze() > mnAtt for sDS, sDD, mnAtt in zip(slowDriftSpikes, slowDriftDirs, meanAtt)]
        # bssSpikesHighAtt = [bSR[lGrtMnAtt] for bSR, lGrtMnAtt in zip(bssSpikesRef, locGreaterThanMeanAtt)]
        # bssSpikesLowAtt = [bSR[~lGrtMnAtt] for bSR, lGrtMnAtt in zip(bssSpikesRef, locGreaterThanMeanAtt)]

        gpfaParamsRemStuff = gpfaParams.copy()
        gpfaParamsRemStuff['overallFiringRateThresh'] = -1000

        savePths = [Path('fromBss') / 'bss_{}'.format(hashlib.md5(subExp['bss_relative_path'][0].encode('ascii')).hexdigest()[:5]) / 'remSd_fromSpikes'  for subExp in subsampleExpressions]
        dimsHNoSdOnSp, dimsLLHNoSdOnSp, gpfaDimOutHNoSdOnSp, gpfaTestIndsHNoSdOnSp, gpfaTrainIndsHNoSdOnSp, *_, gpfaCondLabelsHNoSdOnSp = gpfaComputation(
                    spikesSansSlowDrift, relativePathsToSave = savePths, **gpfaParamsRemStuff
                )

        # NOTE: some paths:
        # for rem 1D slow drift: remOneDimSd_fromSpikes
        # for rem 1D slow drift and double 1s dir: remOneDimSlowDrift_double1sDir
        # for double 1D slow drift: doubleOneDimSd
        savePths = [Path('fromBss') / 'bss_{}'.format(hashlib.md5(subExp['bss_relative_path'][0].encode('ascii')).hexdigest()[:5]) / 'remOneDimSd_fromSpikes'  for subExp in subsampleExpressions]
        gpfaParamsRemStuffTemp = gpfaParamsRemStuff.copy()
        # gpfaParamsRemStuffTemp.update({'forceNewGpfaRun' : True})
        dimsHNo1dSdOnSp, dimsLLHNo1dSdOnSp, gpfaDimOutHNo1dSdOnSp, gpfaTestIndsHNo1dSdOnSp, gpfaTrainIndsHNo1dSdOnSp, *_, gpfaCondLabelsHNo1dSdOnSp = gpfaComputation(
                    spikesSansFirstDimSlowDrift, relativePathsToSave = savePths, **gpfaParamsRemStuffTemp
                )

        # bssSpikesRef = [bSR / bSR.std(axis=0) for bSR in bssSpikesRef]
        bssSpikesNoSd = [(bSR.squeeze() - bSR.squeeze() @ (sD @ sD.T))[:,:,None] for bSR, sD in zip(bssSpikesRef, slowDriftDirs)]
        bssSpikesOnlySd = [( bSR.squeeze() @ (sD @ sD.T))[:,:,None] for bSR, sD in zip(bssSpikesRef, slowDriftDirs)]
        percOrigVarWoSd = [np.cov(bSRRem.squeeze().T,ddof=0).trace()/np.cov(bSR.squeeze().T,ddof=0).trace() for bSR, bSRRem in zip(bssSpikesRef, bssSpikesNoSd)]
        savePths = [Path('fromBss') / 'bss_{}'.format(hashlib.md5(subExp['bss_relative_path'][0].encode('ascii')).hexdigest()[:5]) / 'remSd'  for subExp in subsampleExpressions]
        dimsHNoSd, dimsLLHNoSd, gpfaDimOutHNoSd, gpfaTestIndsHNoSd, gpfaTrainIndsHNoSd, *_, gpfaCondLabelsHNoSd = gpfaComputation(
                    bssSpikesNoSd, relativePathsToSave = savePths, **gpfaParamsRemStuff
                )

        allOnesPerArea = [np.ones(bSR.shape[1])[:,None]/np.sqrt(bSR.shape[1]) for bSR in bssSpikesRef]
        bssSpikesNoAllOnes = [(bSR.squeeze() - bSR.squeeze() @ (aO @ aO.T))[:,:,None] for bSR, aO in zip(bssSpikesRef, allOnesPerArea)]
        bssSpikesOnlyAllOnes = [(bSR.squeeze() @ (aO @ aO.T))[:,:,None] for bSR, aO in zip(bssSpikesRef, allOnesPerArea)]
        percOrigVarWoAllOnes = np.array([np.cov(bSRRem.squeeze().T,ddof=0).trace()/np.cov(bSR.squeeze().T,ddof=0).trace() for bSR, bSRRem in zip(bssSpikesRef, bssSpikesNoAllOnes)])
        percOrigVarOnlyAllOnes = np.array([np.cov(bSRRem.squeeze().T,ddof=0).trace()/np.cov(bSR.squeeze().T,ddof=0).trace() for bSR, bSRRem in zip(bssSpikesRef, bssSpikesOnlyAllOnes)])
        savePths = [Path('fromBss') / 'bss_{}'.format(hashlib.md5(subExp['bss_relative_path'][0].encode('ascii')).hexdigest()[:5]) / 'remAll1'  for subExp in subsampleExpressions]
        dimsHNoAllOnes, dimsLLHNoAllOnes, gpfaDimOutHNoAllOnes, gpfaTestIndsHNoAllOnes, gpfaTrainIndsHNoAllOnes, *_, gpfaCondLabelsHNoAllOnes = gpfaComputation(
                    bssSpikesNoAllOnes, relativePathsToSave = savePths, **gpfaParamsRemStuff
                )

        # savePths = [Path('fromBss') / 'bss_{}'.format(hashlib.md5(subExp['bss_relative_path'][0].encode('ascii')).hexdigest()[:5]) / 'sepAtt' / 'highAtt' for subExp in subsampleExpressions]
        # dimsHHighAttSep, dimsLLHHighAttSep, gpfaDimOutHHighAttSep, gpfaTestIndsHHighAttSep, gpfaTrainIndsHHighAttSep, *_, gpfaCondLabelsHHighAttSep = gpfaComputation(
        #             bssSpikesHighAtt, relativePathsToSave = savePths, **gpfaParams
        #         )
        # savePths = [Path('fromBss') / 'bss_{}'.format(hashlib.md5(subExp['bss_relative_path'][0].encode('ascii')).hexdigest()[:5]) / 'sepAtt' / 'lowAtt' for subExp in subsampleExpressions]
        # dimsHLowAttSep, dimsLLHLowAttSep, gpfaDimOutHLowAttSep, gpfaTestIndsHLowAttSep, gpfaTrainIndsHLowAttSep, *_, gpfaCondLabelsHLowAttSep = gpfaComputation(
        #             bssSpikesLowAtt, relativePathsToSave = savePths, **gpfaParams
        #         )
        # savePths = [Path('fromBss') / 'bss_{}'.format(hashlib.md5(subExp['bss_relative_path'][0].encode('ascii')).hexdigest()[:5]) / 'sepAtt' / 'orig' for subExp in subsampleExpressions]
        # dimsHOrigSep, dimsLLHOrigSep, gpfaDimOutHOrigSep, gpfaTestIndsHOrigSep, gpfaTrainIndsHOrigSep, *_, gpfaCondOrigHLowAttSep = gpfaComputation(
        #             bssSpikesRef, relativePathsToSave = savePths, **gpfaParams
        #         )

        # dmsHighAttFound = [list(gpD[0][0].keys())[0] for gpD in gpfaDimOutHHighAttSep]
        # highAttC = [gpD[0][0][dm]['allEstParams'][0]['C'] for gpD, dm in zip(gpfaDimOutHHighAttSep, dmsHighAttFound)]
        # highAttR = [gpD[0][0][dm]['allEstParams'][0]['R'] for gpD, dm in zip(gpfaDimOutHHighAttSep, dmsHighAttFound)]
        # dmsLowAttFound = [list(gpD[0][0].keys())[0] for gpD in gpfaDimOutHLowAttSep]
        # lowAttC = [gpD[0][0][dm]['allEstParams'][0]['C'] for gpD, dm in zip(gpfaDimOutHLowAttSep, dmsLowAttFound)]
        # lowAttR = [gpD[0][0][dm]['allEstParams'][0]['R'] for gpD, dm in zip(gpfaDimOutHLowAttSep, dmsLowAttFound)]
        dmsOrigFound = [list(gpD[0][0].keys())[0] for gpD in gpfaDimOutH]
        # dmsOrigFound2 = [list(gpD[0][0].keys())[0] for gpD in gpfaDimOutHOrigSep]
        origC = [gpD[0][0][dm]['allEstParams'][0]['C'] for gpD, dm in zip(gpfaDimOutH, dmsOrigFound)]
        origR = [gpD[0][0][dm]['allEstParams'][0]['R'] for gpD, dm in zip(gpfaDimOutH, dmsOrigFound)]

        dmsNoSdOnSpFound = [list(gpD[0][0].keys())[0] for gpD in gpfaDimOutHNoSdOnSp]
        noSdOnSpC = [gpD[0][0][dm]['allEstParams'][0]['C'] for gpD, dm in zip(gpfaDimOutHNoSdOnSp, dmsNoSdOnSpFound)]
        noSdOnSpR = [gpD[0][0][dm]['allEstParams'][0]['R'] for gpD, dm in zip(gpfaDimOutHNoSdOnSp, dmsNoSdOnSpFound)]

        dmsNo1dSdOnSpFound = [list(gpD[0][0].keys())[0] for gpD in gpfaDimOutHNo1dSdOnSp]
        no1dSdOnSpC = [gpD[0][0][dm]['allEstParams'][0]['C'] for gpD, dm in zip(gpfaDimOutHNo1dSdOnSp, dmsNo1dSdOnSpFound)]
        no1dSdOnSpR = [gpD[0][0][dm]['allEstParams'][0]['R'] for gpD, dm in zip(gpfaDimOutHNo1dSdOnSp, dmsNo1dSdOnSpFound)]

        dmsNoSdFound = [list(gpD[0][0].keys())[0] for gpD in gpfaDimOutHNoSd]
        noSdC = [gpD[0][0][dm]['allEstParams'][0]['C'] for gpD, dm in zip(gpfaDimOutHNoSd, dmsNoSdFound)]
        noSdR = [gpD[0][0][dm]['allEstParams'][0]['R'] for gpD, dm in zip(gpfaDimOutHNoSd, dmsNoSdFound)]

        dmsNoAoFound = [list(gpD[0][0].keys())[0] for gpD in gpfaDimOutHNoAllOnes]
        noAoC = [gpD[0][0][dm]['allEstParams'][0]['C'] for gpD, dm in zip(gpfaDimOutHNoAllOnes, dmsNoAoFound)]
        noAoR = [gpD[0][0][dm]['allEstParams'][0]['R'] for gpD, dm in zip(gpfaDimOutHNoAllOnes, dmsNoAoFound)]
                
        # lowAttC = attC[-1] # gpfaDimOutHAttSep[-1][0][0][3]['allEstParams'][0]['C']
        # highAttC = attC[-2] # gpfaDimOutHAttSep[-2][0][0][3]['allEstParams'][0]['C']
        # origC = attC[-3] # gpfaDimOutHAttSep[-3][0][0][3]['allEstParams'][0]['C']
        # lowAttR = attR[-1] # gpfaDimOutHAttSep[-1][0][0][3]['allEstParams'][0]['R']
        # highAttR = attR[-2] # gpfaDimOutHAttSep[-2][0][0][3]['allEstParams'][0]['R']
        # origR = attR[-3] # gpfaDimOutHAttSep[-3][0][0][3]['allEstParams'][0]['R']
        # highAttSv = np.array([(np.diag(C @ C.T)/np.diag(C @ C.T + R)).mean() for C, R in zip(highAttC, highAttR)])
        # lowAttSv = np.array([(np.diag(C @ C.T)/np.diag(C @ C.T + R)).mean() for C, R in zip(lowAttC, lowAttR)])
        origSv = np.array([(np.diag(C @ C.T)/np.diag(C @ C.T + R)).mean() for C, R in zip(origC, origR)])
        noSdSv = np.array([(np.diag(C @ C.T)/np.diag(C @ C.T + R)).mean() for C, R in zip(noSdC, noSdR)])
        noSdOnSpSv = np.array([(np.diag(C @ C.T)/np.diag(C @ C.T + R)).mean() for C, R in zip(noSdOnSpC, noSdOnSpR)])
        no1dSdOnSpSv = np.array([(np.diag(C @ C.T)/np.diag(C @ C.T + R)).mean() for C, R in zip(no1dSdOnSpC, no1dSdOnSpR)])
        noAoSv = np.array([(np.diag(C @ C.T)/np.diag(C @ C.T + R)).mean() for C, R in zip(noAoC, noAoR)])

        orig1Sv =np.array([(np.diag(C[:, [0]] @ C[:, [0]].T)/np.diag(C @ C.T + R)).mean() for C, R in zip(origC, origR)])
        noSd1Sv =np.array([(np.diag(C[:, [0]] @ C[:, [0]].T)/np.diag(C @ C.T + R)).mean() if C.shape[1]>0 else 0 for C, R in zip(noSdC, noSdR)])
        noSdOnSp1Sv =np.array([(np.diag(C[:, [0]] @ C[:, [0]].T)/np.diag(C @ C.T + R)).mean() if C.shape[1]>0 else 0 for C, R in zip(noSdOnSpC, noSdOnSpR)])
        no1dSdOnSp1Sv =np.array([(np.diag(C[:, [0]] @ C[:, [0]].T)/np.diag(C @ C.T + R)).mean() if C.shape[1]>0 else 0 for C, R in zip(no1dSdOnSpC, no1dSdOnSpR)])
        noAo1Sv =np.array([(np.diag(C[:, [0]] @ C[:, [0]].T)/np.diag(C @ C.T + R)).mean() if C.shape[1]>0 else 0 for C, R in zip(noAoC, noAoR)])
        noSdSvOfOrig =np.array([nSD * pOV for nSD, pOV in zip(noSdSv, percOrigVarWoSd)])
        # noSdSvOfOrig =np.array([(np.diag(C @ C.T)/np.diag(Co @ Co.T + Ro)).mean() if C.shape[1]>0 else 0 for C, Co, Ro in zip(noSdC, origC, origR)])
        noSdOnSpSvOfOrig =np.array([nSDoS * (100-pOV[0])/100 for nSDoS, pOV in zip(noSdOnSpSv, slowDriftDict['%overall var of rolling avg' ])])
        # noSdOnSpSvOfOrig =np.array([(np.diag(C @ C.T)/np.diag(Co @ Co.T + Ro)).mean() for C, Co, Ro in zip(noSdOnSpC, origC, origR)])
        no1dSdOnSpSvOfOrig =np.array([nSDoS1 * (100-pOV[0])/100 for nSDoS1, pOV in zip(no1dSdOnSpSv, slowDriftDict['%overall var of slow drift dim rolling avg'])])
        # no1dSdOnSpSvOfOrig =np.array([(np.diag(C @ C.T)/np.diag(Co @ Co.T + Ro)).mean() for C, Co, Ro in zip(no1dSdOnSpC, origC, origR)])
        noAoSvOfOrig =np.array([nAO * pOV for nAO, pOV in zip(noAoSv, percOrigVarWoAllOnes)])
        # noAoSvOfOrig =np.array([(np.diag(C @ C.T)/np.diag(Co @ Co.T + Ro)).mean() for C, Co, Ro in zip(noAoC, origC, origR)])

        brainAreasUnique, brnLoc = np.unique(brainAreas, return_inverse=True)
        availableColors = np.array(px.colors.qualitative.Plotly[:len(brainAreasUnique)])
        # px.line().add_scatter(
        #         y=origSv, name='orig'
        #     ).add_scatter(
        #         y=noSdSv,name='no sd'
        #     ).add_scatter(
        #         y=noAoSv, name='no all 1s'
        #     ).update_layout(title='%sv in datasets when directions removed').show()
        # # px.line().add_scatter(y=orig1Sv, name='orig').add_scatter(y=noSd1Sv,name='no sd').add_scatter(y=noAo1Sv, name='no all 1s').show()
        # px.line().add_scatter(
        #         y=dmsOrigFound, name='orig'
        #     ).add_scatter(
        #         y=dmsNoSdFound,name='no sd'
        #     ).add_scatter(
        #         y=dmsNoAoFound, name='no all 1s'
        #     ).update_layout(title='dimensionality in datasets when directions removed').show()

        # px.line().add_scatter(x=brnLoc, y=dmsOrigFound, mode='markers',
        #             marker=dict(color=availableColors[brnLoc]),
        # ).update_layout(
        #     title='dimensionality of original datasets',
        #     xaxis_title_text='brain area',
        #     xaxis = dict(
        #         tickmode = 'array',
        #         tickvals = [0,1,2],
        #         ticktext = brainAreasUnique,
        #     ),
        # ).show()
#%%
        fgDmsComp = px.line()
        # [fgDmsComp.add_scatter(x=np.array([0, 1, 2, 3, 4]) + (bL - 1)/4, y=np.array([og, nos, nosSp, nos1dSp, noa]), mode='markers',
        [fgDmsComp.add_scatter(x=np.array([0, 1, 2, 3, 4])/10 + bL, y=np.array([og, nos, nosSp, nos1dSp, noa]), mode='markers',
                    marker=dict(color=availableColors[bL]), name=brainAreasUnique[bL],
        ) for og, nos, nosSp, nos1dSp, noa, bL in zip(dmsOrigFound, dmsNoSdFound, dmsNoSdOnSpFound, dmsNo1dSdOnSpFound, dmsNoAoFound, brnLoc)]
        fgDmsComp.update_layout(
            title='dimensionality shift when directions removed',
            xaxis_title_text='direction removed',
            xaxis = dict(
                tickmode = 'array',
                # tickvals = np.array([0,1,2,3,4])/10,
                tickvals = (np.array([0,1,2,3,4])/10 + np.arange(brainAreasUnique.shape[0])[:,None]).flatten(),
                # ticktext = ['original', 'no slow drift', 'no slow drift from spikes', 'no 1st slow drift dim from spikes', 'no all 1s'],
                ticktext = np.tile(['original', 'no slow drift', 'no slow drift from spikes', 'no 1st slow drift dim from spikes', 'no all 1s'], brainAreasUnique.shape[0]),
            )
        ).show()

        fgSvComp = px.line()
        # [fgSvComp.add_scatter(x=np.array([0, 1, 2, 3, 4]) + (bL - 1)/4, y=np.array([og, nos, nosSp, nos1dSp, noa]), mode='markers',
        [fgSvComp.add_scatter(x=np.array([0, 1, 2, 3, 4])/10 + bL, y=np.array([og, nos, nosSp, nos1dSp, noa]), mode='markers',
                    marker=dict(color=availableColors[bL]), name=brainAreasUnique[bL],
        ) for og, nos, nosSp, nos1dSp, noa, bL in zip(origSv, noSdSv, noSdOnSpSv, no1dSdOnSpSv, noAoSv, brnLoc)]
        fgSvComp.update_layout(
            title='%sv when directions removed',
            xaxis_title_text='direction removed',
            xaxis = dict(
                tickmode = 'array',
                # tickvals = np.array([0,1,2,3,4]),
                tickvals = (np.array([0,1,2,3,4])/10 + np.arange(brainAreasUnique.shape[0])[:,None]).flatten(),
                # ticktext = ['original', 'no slow drift', 'no slow drift from spikes', 'no 1st slow drift dim from spikes', 'no all 1s'],
                ticktext = np.tile(['original', 'no slow drift', 'no slow drift from spikes', 'no 1st slow drift dim from spikes', 'no all 1s'], brainAreasUnique.shape[0]),
            )
        ).show()

        fgSv1Comp = px.line()
        # [fgSv1Comp.add_scatter(x=np.array([0, 1, 2, 3, 4]) + (bL - 1)/4, y=np.array([og, nos, nosSp, nos1dSp, noa]), mode='markers',
        [fgSv1Comp.add_scatter(x=np.array([0, 1, 2, 3, 4])/10 + bL, y=np.array([og, nos, nosSp, nos1dSp, noa]), mode='markers',
                    marker=dict(color=availableColors[bL]), name=brainAreasUnique[bL],
        ) for og, nos, nosSp, nos1dSp, noa, bL in zip(orig1Sv, noSd1Sv, noSdOnSp1Sv, no1dSdOnSp1Sv, noAo1Sv, brnLoc)]
        fgSv1Comp.update_layout(
            title='%sv of first latent when directions removed',
            xaxis_title_text='direction removed',
            xaxis = dict(
                tickmode = 'array',
                # tickvals = np.array([0,1,2,3,4]),
                tickvals = (np.array([0,1,2,3,4])/10 + np.arange(brainAreasUnique.shape[0])[:,None]).flatten(),
                # ticktext = ['original', 'no slow drift', 'no slow drift from spikes', 'no 1st slow drift dim from spikes', 'no all 1s'],
                ticktext = np.tile(['original', 'no slow drift', 'no slow drift from spikes', 'no 1st slow drift dim from spikes', 'no all 1s'], brainAreasUnique.shape[0]),
            )
        ).show()

        slowDriftDictSimple = slowDriftDict.copy()
        slowDriftDictSimple.pop('%ev of latent dimension by slow drift dim')
        slowDriftDictMetrics = list(slowDriftDictSimple.keys())
        listOfMetrics = [metric for _, metric in slowDriftDictSimple.items()]
        numMetrics = len(slowDriftDictMetrics)
        sDMetrics = px.line()
        [sDMetrics.add_scatter(x=np.arange(numMetrics)/(2*numMetrics) + bL, y=np.array(metrics).squeeze(), mode='markers',
                    marker=dict(color=availableColors[bL]), name=brainAreasUnique[bL],
        ) for metrics, bL in zip(list(zip(*listOfMetrics)), brnLoc)]
        sDMetrics.update_layout(
            title='miscellanious % variance measures of slow drift dim',
            xaxis_title_text='% variance metric',
            xaxis = dict(
                tickmode = 'array',
                # tickvals = np.array([0,1,2,3,4]),
                tickvals = (np.array([0,1,2,3,4])/10 + np.arange(brainAreasUnique.shape[0])[:,None]).flatten(),
                # ticktext = ['original', 'no slow drift', 'no slow drift from spikes', 'no 1st slow drift dim from spikes', 'no all 1s'],
                ticktext = np.tile(slowDriftDictMetrics, brainAreasUnique.shape[0]),
            )
        ).show()


        fgSvOfOrig = px.line()
        [fgSvOfOrig.add_scatter(x=np.array([0, 1, 2, 3, 4]) + (bL - 1)/4, y=np.array([og, nos, nosSp, nos1dSp, noa]), mode='markers',
                    marker=dict(color=availableColors[bL]), name=brainAreasUnique[bL],
        ) for og, nos, nosSp, nos1dSp, noa, bL in zip(origSv, noSdSvOfOrig, noSdOnSpSvOfOrig, no1dSdOnSpSvOfOrig, noAoSvOfOrig, brnLoc)]
        fgSvOfOrig.update_layout(
            title='% of original variance explained by shared latents when direction removed',
            xaxis_title_text='direction removed',
            xaxis = dict(
                tickmode = 'array',
                tickvals = [0,1,2,3,4],
                ticktext = ['original', 'no slow drift', 'no slow drift from spikes', 'no 1st slow drift dim from spikes', 'no all 1s'],
            )
        ).show()

#%%
        # highAttCorr = [np.cov(spks.squeeze().T,ddof=0)/np.std(spks.squeeze(),axis=0,ddof=0)[:,None]/np.std(spks.squeeze(),axis=0,ddof=0)[None,:] for spks in bssSpikesHighAtt]
        # lowAttCorr = [np.cov(spks.squeeze().T,ddof=0)/np.std(spks.squeeze(),axis=0,ddof=0)[:,None]/np.std(spks.squeeze(),axis=0,ddof=0)[None,:] for spks in bssSpikesLowAtt]
        origCorr = [np.cov(spks.squeeze().T,ddof=0)/np.std(spks.squeeze(),axis=0,ddof=0)[:,None]/np.std(spks.squeeze(),axis=0,ddof=0)[None,:] for spks in bssSpikesRef]
        noSdOnSpCorr = [np.cov(spks.squeeze().T,ddof=0)/np.std(spks.squeeze(),axis=0,ddof=0)[:,None]/np.std(spks.squeeze(),axis=0,ddof=0)[None,:] for spks in spikesSansSlowDrift]
        no1dSdOnSpCorr = [np.cov(spks.squeeze().T,ddof=0)/np.std(spks.squeeze(),axis=0,ddof=0)[:,None]/np.std(spks.squeeze(),axis=0,ddof=0)[None,:] for spks in spikesSansFirstDimSlowDrift]
        noSdCorr = [np.cov(spks.squeeze().T,ddof=0)/np.std(spks.squeeze(),axis=0,ddof=0)[:,None]/np.std(spks.squeeze(),axis=0,ddof=0)[None,:] for spks in bssSpikesNoSd]
        onlySdCorr = [np.cov(spks.squeeze().T,ddof=0)/np.std(spks.squeeze(),axis=0,ddof=0)[:,None]/np.std(spks.squeeze(),axis=0,ddof=0)[None,:] for spks in bssSpikesOnlySd]
        noAoCorr = [np.cov(spks.squeeze().T,ddof=0)/np.std(spks.squeeze(),axis=0,ddof=0)[:,None]/np.std(spks.squeeze(),axis=0,ddof=0)[None,:] for spks in bssSpikesNoAllOnes]
        onlyAoCorr = [np.cov(spks.squeeze().T,ddof=0)/np.std(spks.squeeze(),axis=0,ddof=0)[:,None]/np.std(spks.squeeze(),axis=0,ddof=0)[None,:] for spks in bssSpikesOnlyAllOnes]
        # highAttRsc = np.array([spkCorr[np.triu_indices(spkCorr.shape[0],k=1)].mean() for spkCorr in highAttCorr])
        # lowAttRsc = np.array([spkCorr[np.triu_indices(spkCorr.shape[0],k=1)].mean() for spkCorr in lowAttCorr])
        origRsc = np.array([spkCorr[np.triu_indices(spkCorr.shape[0],k=1)].mean() for spkCorr in origCorr])
        noSdOnSpRsc = np.array([spkCorr[np.triu_indices(spkCorr.shape[0],k=1)].mean() for spkCorr in noSdOnSpCorr])
        no1dSdOnSpRsc = np.array([spkCorr[np.triu_indices(spkCorr.shape[0],k=1)].mean() for spkCorr in no1dSdOnSpCorr])
        noSdRsc = np.array([spkCorr[np.triu_indices(spkCorr.shape[0],k=1)].mean() for spkCorr in noSdCorr])
        onlySdRsc = np.array([spkCorr[np.triu_indices(spkCorr.shape[0],k=1)].mean() for spkCorr in onlySdCorr])
        noAoRsc = np.array([spkCorr[np.triu_indices(spkCorr.shape[0],k=1)].mean() for spkCorr in noAoCorr])
        onlyAoRsc = np.array([spkCorr[np.triu_indices(spkCorr.shape[0],k=1)].mean() for spkCorr in onlyAoCorr])
        # highAttRscStd = np.array([spkCorr[np.triu_indices(spkCorr.shape[0],k=1)].std() for spkCorr in highAttCorr])
        # lowAttRscStd = np.array([spkCorr[np.triu_indices(spkCorr.shape[0],k=1)].std() for spkCorr in lowAttCorr])
        origRscStd = np.array([spkCorr[np.triu_indices(spkCorr.shape[0],k=1)].std() for spkCorr in origCorr])
        noSdOnSpRscStd = np.array([spkCorr[np.triu_indices(spkCorr.shape[0],k=1)].std() for spkCorr in noSdOnSpCorr])
        no1dSdOnSpRscStd = np.array([spkCorr[np.triu_indices(spkCorr.shape[0],k=1)].std() for spkCorr in no1dSdOnSpCorr])
        noSdRscStd = np.array([spkCorr[np.triu_indices(spkCorr.shape[0],k=1)].std() for spkCorr in noSdCorr])
        onlySdRscStd = np.array([spkCorr[np.triu_indices(spkCorr.shape[0],k=1)].std() for spkCorr in onlySdCorr])
        noAoRscStd = np.array([spkCorr[np.triu_indices(spkCorr.shape[0],k=1)].std() for spkCorr in noAoCorr])
        onlyAoRscStd = np.array([spkCorr[np.triu_indices(spkCorr.shape[0],k=1)].std() for spkCorr in onlyAoCorr])
        # px.scatter(x=highAttRsc-lowAttRsc, y=highAttSv-lowAttSv).show()
        # px.scatter(x=highAttRscStd-lowAttRscStd, y=highAttSv-lowAttSv).show()
        # px.scatter(x=np.sqrt(highAttRsc**2 + highAttRscStd**2)-np.sqrt(lowAttRsc**2+lowAttRscStd**2), y=highAttSv-lowAttSv).show()

        origFaRsc = np.array([(C @ C.T)[np.triu_indices(C.shape[0], k=1)].mean() for C in origC])
        noSdFaRsc = [(C @ C.T)[np.triu_indices(C.shape[0], k=1)].mean() for C in noSdC]
        no1dSdFaRsc = [(C @ C.T)[np.triu_indices(C.shape[0], k=1)].mean() for C in no1dSdOnSpC]
        noAoFaRsc = [(C @ C.T)[np.triu_indices(C.shape[0], k=1)].mean() for C in noAoC]
        origFaRscStd = [(C @ C.T)[np.triu_indices(C.shape[0], k=1)].std() for C in origC]
        noSdFaRscStd = [(C @ C.T)[np.triu_indices(C.shape[0], k=1)].std() for C in noSdC]
        no1dSdFaRscStd = [(C @ C.T)[np.triu_indices(C.shape[0], k=1)].std() for C in no1dSdOnSpC]
        noAoFaRscStd = [(C @ C.T)[np.triu_indices(C.shape[0], k=1)].std() for C in noAoC]

        origFaByLatRsc = [[(sv**2 * C[:,None] @ C[:,None].T)[np.triu_indices(C.shape[0], k=1)].mean() for _, sv, C in zip(*np.linalg.svd(Call.T))] for Call in origC]
        noSdFaByLatRsc = [[(sv**2 * C[:,None] @ C[:,None].T)[np.triu_indices(C.shape[0], k=1)].mean() for _, sv, C in zip(*np.linalg.svd(Call.T))] for Call in noSdC]
        no1dSdFaByLatRsc = [[(sv**2 * C[:,None] @ C[:,None].T)[np.triu_indices(C.shape[0], k=1)].mean() for _, sv, C in zip(*np.linalg.svd(Call.T))] for Call in no1dSdOnSpC]
        noAoFaByLatRsc = [[(sv**2 * C[:,None] @ C[:,None].T)[np.triu_indices(C.shape[0], k=1)].mean() for _, sv, C in zip(*np.linalg.svd(Call.T))] for Call in noAoC]
        origFaByLatRscStd = [[(sv**2 * C[:,None] @ C[:,None].T)[np.triu_indices(C.shape[0], k=1)].std() for _, sv, C in zip(*np.linalg.svd(Call.T))] for Call in origC]
        noSdFaByLatRscStd = [[(sv**2 * C[:,None] @ C[:,None].T)[np.triu_indices(C.shape[0], k=1)].std() for _, sv, C in zip(*np.linalg.svd(Call.T))] for Call in noSdC]
        no1dSdFaByLatRscStd = [[(sv**2 * C[:,None] @ C[:,None].T)[np.triu_indices(C.shape[0], k=1)].std() for _, sv, C in zip(*np.linalg.svd(Call.T))] for Call in no1dSdOnSpC]
        noAoFaByLatRscStd = [[(sv**2 * C[:,None] @ C[:,None].T)[np.triu_indices(C.shape[0], k=1)].std() for _, sv, C in zip(*np.linalg.svd(Call.T))] for Call in noAoC]

        brainAreasUnique = np.unique(brainAreas)
        availableColors = np.array(px.colors.qualitative.Plotly[:len(brainAreasUnique)])
        fgSt = px.line()
        brainAreasLegend = {bA : True for bA in brainAreasUnique}
        [fgSt.add_scatter(
            x=[rscMn], y=[rscStd],
            marker_symbol='circle',
            name = bA + ' original data',
            legendgroup = 'original data',
            showlegend = brainAreasLegend.pop(bA, False),
            line=dict(color=availableColors[brainAreasUnique==bA][0]),
            )
            for rscMn, rscStd, bA in zip(origRsc, origRscStd, brainAreas)]
        brainAreasLegend = {bA : True for bA in brainAreasUnique}
        [fgSt.add_scatter(
            x=[rscMn], y=[rscStd],
            marker_symbol='x',
            name = bA + ' no slow drift',
            legendgroup = 'no slow drift',
            showlegend = brainAreasLegend.pop(bA, False),
            line=dict(color=availableColors[brainAreasUnique==bA][0]),
            )
            for rscMn, rscStd, bA in zip(noSdRsc, noSdRscStd, brainAreas)]
        brainAreasLegend = {bA : True for bA in brainAreasUnique}
        [fgSt.add_scatter(
            x=[rscMn], y=[rscStd],
            marker_symbol='triangle-up',
            name = bA + ' no slow drift from spikes',
            legendgroup = 'no slow drift from spikes',
            showlegend = brainAreasLegend.pop(bA, False),
            line=dict(color=availableColors[brainAreasUnique==bA][0]),
            )
            for rscMn, rscStd, bA in zip(noSdOnSpRsc, noSdOnSpRscStd, brainAreas)]
        brainAreasLegend = {bA : True for bA in brainAreasUnique}
        [fgSt.add_scatter(
            x=[rscMn], y=[rscStd],
            marker_symbol='triangle-down',
            name = bA + ' no 1D slow drift from spikes',
            legendgroup = 'no 1D slow drift from spikes',
            showlegend = brainAreasLegend.pop(bA, False),
            line=dict(color=availableColors[brainAreasUnique==bA][0]),
            )
            for rscMn, rscStd, bA in zip(no1dSdOnSpRsc, no1dSdOnSpRscStd, brainAreas)]
        brainAreasLegend = {bA : True for bA in brainAreasUnique}
        [fgSt.add_scatter(
            x=[rscMn], y=[rscStd],
            marker_symbol='cross',
            name = bA + ' no all 1s',
            legendgroup = 'no all 1s',
            showlegend = brainAreasLegend.pop(bA, False),
            line=dict(color=availableColors[brainAreasUnique==bA][0]),
            )
            for rscMn, rscStd, bA in zip(noAoRsc, noAoRscStd, brainAreas)]
        fgSt.update_layout(
            title='rsc without slow drift and without all ones',
            xaxis=dict(range=[np.minimum(np.hstack([origRsc, noSdRsc, noAoRsc]).min(), 0), np.minimum(np.hstack([origRsc, noSdRsc, noAoRsc]).max(), 1)]),
            yaxis=dict(range=[np.minimum(np.hstack([origRsc, noSdRsc, noAoRsc]).min(), 0), np.minimum(np.hstack([origRsc, noSdRsc, noAoRsc]).max(), 1)], scaleanchor = "x", scaleratio = 1)
        );
        fgSt.show()

        fgInd = px.line()
        [fgInd.add_scatter(
            x=[rscMn], y=[rscStd],
            marker_symbol='circle',
            line=dict(color=availableColors[brainAreasUnique==bA][0]),
            )
            for rscMn, rscStd, bA in zip(origRsc, origRscStd, brainAreas)]
        [fgInd.add_scatter(
            x=[rscMn], y=[rscStd],
            marker_symbol='x',
            line=dict(color=availableColors[brainAreasUnique==bA][0]),
            )
            for rscMn, rscStd, bA in zip(onlySdRsc, onlySdRscStd, brainAreas)]
        [fgInd.add_scatter(
            x=[rscMn], y=[rscStd],
            marker_symbol='cross',
            line=dict(color=availableColors[brainAreasUnique==bA][0]),
            )
            for rscMn, rscStd, bA in zip(onlyAoRsc, onlyAoRscStd, brainAreas)]
        fgInd.update_layout(
            title='rsc overall and all ones or slow drift rsc'
        )
        fgInd.show()

        fgFromFaWithAndWithout = px.line()
        brainAreasLegend = {bA : True for bA in brainAreasUnique}
        [fgFromFaWithAndWithout.add_scatter(
            x=[rscMn], y=[rscStd],
            marker_symbol='x',
            name = bA + ' no slow drift',
            legendgroup = bA + ' no slow drift',
            showlegend = brainAreasLegend.pop(bA, False),
            line=dict(color=availableColors[brainAreasUnique==bA][0]),
            ) for rscMn, rscStd, bA in zip(origFaRsc, noSdFaRsc, brainAreas)]
        brainAreasLegend = {bA : True for bA in brainAreasUnique}
        [fgFromFaWithAndWithout.add_scatter(
            x=[rscMn], y=[rscStd],
            marker_symbol='cross',
            name = bA + ' no all 1s',
            legendgroup = bA + ' no all 1s',
            showlegend = brainAreasLegend.pop(bA, False),
            line=dict(color=availableColors[brainAreasUnique==bA][0]),
            ) for rscMn, rscStd, bA in zip(origFaRsc, noAoFaRsc, brainAreas)]
        fgFromFaWithAndWithout.update_layout(
            title='rsc from FA (with vs without slow drift/all ones',
            xaxis_title = 'FA original data r_sc',
            yaxis_title = 'FA modified data r_sc'
        );
        fgFromFaWithAndWithout.show()

        fgFromFa = px.line()
        brainAreasLegend = {bA : True for bA in brainAreasUnique}
        [fgFromFa.add_scatter(
            x=[rscMn], y=[rscStd],
            marker_symbol='circle',
            name = bA + ' original data',
            legendgroup = bA + ' original data',
            showlegend = brainAreasLegend.pop(bA, False),
            line=dict(color=availableColors[brainAreasUnique==bA][0]),
            ) for rscMn, rscStd, bA in zip(origFaRsc, [np.array(k)**2 for k in origFaRscStd], brainAreas)]
        brainAreasLegend = {bA : True for bA in brainAreasUnique}
        [fgFromFa.add_scatter(
            x=[rscMn], y=[rscStd],
            marker_symbol='cross',
            name = bA + ' no slow drift',
            legendgroup = bA + ' no slow drift',
            showlegend = brainAreasLegend.pop(bA, False),
            line=dict(color=availableColors[brainAreasUnique==bA][0]),
            ) for rscMn, rscStd, bA in zip(noSdFaRsc, [np.array(k)**2 for k in noSdFaRscStd], brainAreas)]
        brainAreasLegend = {bA : True for bA in brainAreasUnique}
        [fgFromFa.add_scatter(
            x=[rscMn], y=[rscStd],
            marker_symbol='x',
            name = bA + ' no all 1s',
            legendgroup = bA + ' no all 1s',
            showlegend = brainAreasLegend.pop(bA, False),
            line=dict(color=availableColors[brainAreasUnique==bA][0]),
            ) for rscMn, rscStd, bA in zip(noAoFaRsc, [np.array(k)**2 for k in noAoFaRscStd], brainAreas)]
        fgFromFa.update_layout(
            title='rsc from FA (with vs without slow drift/all ones)',
            xaxis_title = 'r_sc mean',
            yaxis_title = 'r_sc std'
        );
        fgFromFa.show()


        fgEachLat = px.line()
        brainAreasLegend = {bA : True for bA in brainAreasUnique}
        [fgEachLat.add_scatter(
            x=rscMn, y=rscStd,
            marker_symbol='circle',
            name = bA + ' original data',
            legendgroup = bA + ' original data',
            showlegend = brainAreasLegend.pop(bA, False),
            line=dict(color=availableColors[brainAreasUnique==bA][0]),
            ) for rscMn, rscStd, bA in zip(origFaByLatRsc, [np.array(k)**2 for k in origFaByLatRscStd], brainAreas)]
        brainAreasLegend = {bA : True for bA in brainAreasUnique}
        [fgEachLat.add_scatter(
            x=rscMn, y=rscStd,
            marker_symbol='cross',
            name = bA + ' no slow drift',
            legendgroup = bA + ' no slow drift',
            showlegend = brainAreasLegend.pop(bA, False),
            line=dict(color=availableColors[brainAreasUnique==bA][0]),
            ) for rscMn, rscStd, bA in zip(noSdFaByLatRsc, [np.array(k)**2 for k in noSdFaByLatRscStd], brainAreas)]
        brainAreasLegend = {bA : True for bA in brainAreasUnique}
        [fgEachLat.add_scatter(
            x=rscMn, y=rscStd,
            marker_symbol='x',
            name = bA + ' no all 1s',
            legendgroup = bA + ' no all 1s',
            showlegend = brainAreasLegend.pop(bA, False),
            line=dict(color=availableColors[brainAreasUnique==bA][0]),
            ) for rscMn, rscStd, bA in zip(noAoFaByLatRsc, [np.array(k)**2 for k in noAoFaByLatRscStd], brainAreas)]
        fgEachLat.update_layout(
            title='rsc from FA by latent (with vs without slow drift/all ones)',
            xaxis_title = 'r_sc mean',
            yaxis_title = 'r_sc std'
        );
        fgEachLat.show()

        fgSdLs = px.line()
        sdLoadSim = [1-sD.shape[0]*sD.var() for sD in slowDriftDirs]
        brainAreasLegend = {bA : True for bA in brainAreasUnique}
        [fgSdLs.add_scatter(
            x=[rscMn], y=[loadSim],
            marker_symbol='x',
            name = bA + ' no slow drift',
            legendgroup = bA + ' no slow drift',
            showlegend = brainAreasLegend.pop(bA, False),
            line=dict(color=availableColors[brainAreasUnique==bA][0]),
            )
            for rscMn, loadSim, bA in zip(origFaRsc - noSdFaRsc, sdLoadSim, brainAreas)]
        fgSdLs.update_layout(
            title='rsc diff vs loading similarity of slow drift',
        )
        fgSdLs.show()

        breakpoint()


        shuffCorrs = []
        trueCorrs = []
        for m1DsetNum in range(len(bssKeys)):
            m1Dset = dsi[bssKeys[m1DsetNum]].grabDataset()
            # m1DsetCorr,_ = m1Dset.filterOutState('Catch')[0].successfulTrials()
            m1DsetUse, *_ = m1Dset.filterTrials(trlFiltAll[m1DsetNum][-1])

            startState, endState, stateNameAfter = m1DsetUse.computeStateStartAndEnd(stateName = 'Delay Period')
            startStateArr = np.asarray(startState)
            endStateArr = np.asarray(endState)
            delayTime = endStateArr - startStateArr

            spPar = subExpParents[m1DsetNum].grabBinnedSpikes()[0]
            initSt = np.random.get_state()
            np.random.seed(seed=0)
            indsUsed = spPar.balancedTrialInds(labels=spPar.labels['stimulusMainLabel'])
            np.random.set_state(initSt)
            stPres = np.empty(len(m1DsetUse.statesPresented), np.object)
            stPres[:] = m1DsetUse.statesPresented
            stNms = np.empty(len(m1DsetUse.stateNames), np.object)
            stNms[:] = m1DsetUse.stateNames
            reachStartTime = np.array([sP[1,sN[sP[0,:]-1]=='Reach Start'] for sP, sN in zip(stPres[indsUsed], stNms[indsUsed])]).squeeze()
            goCueTime = np.array([sP[1,sN[sP[0,:]-1]=='Go Cue'] for sP, sN in zip(stPres[indsUsed], stNms[indsUsed])]).squeeze()
            reactionTime = reachStartTime - goCueTime
            trueCorr = np.corrcoef(reactionTime, slowDriftProjPar[m1DsetNum])[0,1]
            indsR = np.arange(reactionTime.shape[0])
            shuffCorr = np.array([np.corrcoef(reactionTime[np.random.permutation(indsR)], slowDriftProjPar[m1DsetNum])[0,1] for _ in range(1000)])
            shuffCorrs.append(shuffCorr)
            trueCorrs.append(trueCorr)

        breakpoint()



        fgAng = px.scatter(x=np.ones_like(slowDriftDirComparisonSameNeurs), y=slowDriftDirComparisonSameNeurs)
        fgAng.update_layout(
            title = 'angle between low trial and high trial slow drift axis',
            xaxis = dict(
                tickmode = 'array',
                tickvals = [1],
                ticktext = [brainAreas[0]],
            ),
            xaxis_title_text='brain area',
            yaxis_title_text = 'angle ($^\circ$)',
        )
        fgAng.show()
        # fgAng,ax = plt.subplots(1,1)
        # ax.scatter(np.ones_like(slowDriftDirComparisonSameNeurs), slowDriftDirComparisonSameNeurs)
        # ax.set_xticks([1])
        # ax.set_xticklabels(['PFC data'])
        # ax.set_ylabel('angle ($^\circ$)')

        fgAngVsTrls = px.scatter(x=(numTrlsParents-336).squeeze(), y=slowDriftDirComparisonSameNeurs)
        fgAngVsTrls.update_layout(
            title = 'effect of subsampling amount on angle between full/subsample slow drift dim',
            xaxis_title_text="difference between 'all' and 'subset' trial number",
            yaxis_title_text = 'angle ($^\circ$)',
        )
        fgAngVsTrls.show()

        evSlowDriftInShSubsample = np.stack(slowDriftDict['%ev of slow drift dim by shared space']).squeeze()
        evSlowDriftInShFull = np.stack(slowDriftDictParsSameNeursFA['%ev of slow drift dim by shared space']).squeeze()
        fgEvSdInSh = px.scatter(x=evSlowDriftInShFull, y=evSlowDriftInShSubsample)
        fgEvSdInSh.add_scatter(x=[0,100], y=[0,100], mode='lines', line=dict(dash='dash'))
        fgEvSdInSh.update_layout(
            title = 'effect of subsampling on FA space/slow drift overlap',
            xaxis_title_text = 'all trials: %ev of slow drift dim by shared space',
            yaxis_title_text = 'effect of subsampling on FA space/slow drift overlap',
        )
        fgEvSdInSh.show()

        slowDriftDirComparisonMatchedOrDiffNeurExtract[-1] = np.abs(180-slowDriftDirComparisonMatchedOrDiffNeurExtract[-1])
        fgAngMatchedVsAllNeur = px.scatter(x=(np.ones((slowDriftDirComparisonSameNeurs.shape[0], 2)) + [0,1]).flatten(), y=np.vstack([slowDriftDirComparisonSameNeurs,slowDriftDirComparisonMatchedOrDiffNeurExtract]).T.flatten())
        fgAngMatchedVsAllNeur.update_layout(
            title = 'effect of trial limitation vs neuron limitation on slow drift angle',
            xaxis = dict(
                tickmode = 'array',
                tickvals = [1, 2],
                ticktext = ['subsample neurons:\nall trials vs subsample trials', 'all trials:\nall neurons vs subsample neurons '],
            ),
            xaxis_title_text='comparison',
            yaxis_title_text = 'angle ($^\circ$)',
        )
        fgAngMatchedVsAllNeur.show()

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


    metricDict = {}
    metricDict.update(slowDriftDict)
    metricDict.update(overSessionMetricsDict)
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
