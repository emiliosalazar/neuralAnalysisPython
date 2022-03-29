#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 19:01:45 2020

@author: emilio

Here will be methods to act on lists of BinnedSpikeSets. Unclear whether these
are better set out as a class that contains lists of these... but methinks not
"""

import numpy as np
import scipy as sp
from classes.BinnedSpikeSet import BinnedSpikeSet
from matplotlib import pyplot as plt
from pathlib import Path
from methods.GeneralMethods import loadDefaultParams
# from decorators.ParallelProcessingDecorators import parallelize
from setup.DataJointSetup import BinnedSpikeSetProcessParams, BinnedSpikeSetInfo, DatasetInfo, GpfaAnalysisInfo, GpfaAnalysisParams, FilterSpikeSetParams
import hashlib
import json
import dill as pickle

from methods.GpfaMethods import crunchGpfaResults, computeBestDimensionality
from methods.plotMethods.BinnedSpikeSetPlotMethods import plotGpfaResults, plotSlowDriftVsFaSpace

def generateBinnedSpikeListsAroundState(data, keyStateName, trialType = 'successful', lenSmallestTrl=251, binSizeMs = 25, furthestTimeBeforeState=251, furthestTimeAfterState=251, setStartToStateEnd = False, setEndToStateStart = False, returnResiduals = False,  firingRateThresh = 1, fanoFactorThresh = 4, unitsOut = None, trialFilterLambda = None):

#    if type(data) == DatasetInfo:
#    data = data.grabDatasets()

    defaultParams = loadDefaultParams()
    dataPath = defaultParams['dataPath']


    if setStartToStateEnd:
        startOffsetLocation = 'stateEnd'
    else:
        startOffsetLocation = 'stateStart'
    
    if setEndToStateStart:
        endOffsetLocation = 'stateStart'
    else:
        endOffsetLocation = 'stateEnd'

    bSSProcParams = dict(
        key_state_start = keyStateName, # in this function the key state is the start and end
        key_state_end = keyStateName, # in this function the key state is the start and end
        start_offset = -furthestTimeBeforeState,
        start_offset_from_location = startOffsetLocation,
        end_offset = furthestTimeAfterState, # note I'm ignoring the + binSizeMs/20... hope I don't rue the day...
        end_offset_from_location = endOffsetLocation,
        bin_size = binSizeMs,
        firing_rate_thresh = float(firingRateThresh), # to match database
        fano_factor_thresh = float(fanoFactorThresh), # to match database
        trial_type = trialType,
        len_smallest_trial = lenSmallestTrl,
        residuals = int(returnResiduals)
    )
    bsspp = BinnedSpikeSetProcessParams()

    if trialFilterLambda is None:
        trialFilterLambda = {}
    
    trialFilterLambda.update({'remove trials with catch state' : "lambda ds : ds.filterOutState('Catch')"})
    binnedSpikes, bssiKeys = data.computeBinnedSpikesAroundState(bSSProcParams, keyStateName, trialFilterLambda, units = unitsOut)

    
    return binnedSpikes, bssiKeys

# This function lets you match the number of neurons and trials for different
# sets of data you're comparing, to be fairer in the comparison. Note that what
# is referred to as 'neuron' in the method name could well be a thresholded
# channel with multineuron activity... but this nomencalture reminds us what
# the purpose is...
def subsampleBinnedSpikeSetsToMatchNeuronsAndTrialsPerCondition(bssKeys, maxNumTrlPerCond, maxNumNeuron, labelName, numSubsamples = 1,  extraOpts = None, order=True):
    # I do, unfortunately, have to load them twice here--once up here to
    # compute the neuron/trial nums, and again below to make sure they're well
    # aligned

    dsi = DatasetInfo()
    bsi = BinnedSpikeSetInfo()

#    if order:
#        bssKeys = (bssExp * dsi).fetch('KEY', order_by='brain_area')
#    else:
#        bssKeys = bssExp.fetch('KEY')

    subsmpKeys = []
#    initRandSt = np.random.get_state()
    bnSpSubsamples = []
    trlNeurSubsamples = []
    datasetNames = []
    brainAreas = []
    tasks = []
    subsampleExpressions = []
    fsp = FilterSpikeSetParams()
    bsi = BinnedSpikeSetInfo()
    for bssInd, bssKeyForOne in enumerate(bssKeys):
        # reset the seed for every binned spike so we know that whenever a set
        # of data comes through here it gets split up identically... well...
        # assuming the same minima up there...
        bnSpOrigInfo = bsi[bssKeyForOne]
        if extraOpts:
            extraFilt = []
            extraDescription = []
            anyMatch = False
            for extraOptExp, filterParams in extraOpts.items():
                match = bsi[extraOptExp][bssKeyForOne]
                filterLambda = filterParams['filter']
                filterDescription = filterParams['description']
                if len(match):
                    anyMatch = True
                    extraFilt.append(filterLambda(match))
                    extraDescription.append(filterDescription)

            if anyMatch:
                totalFilt = np.concatenate(extraFilt,axis=1)
                totalFilt = np.all(totalFilt,axis=1)
                _, filtBssKeys = bnSpOrigInfo.filterBSS(filterReason = "other", filterDescription = "; ".join(extraDescription), condLabel=labelName, trialFilter = totalFilt, returnKey = True)
                bnSpOrigInfo = bsi[filtBssKeys]


        bnSpHereSubsample, trlNeurHereSubsamples, datasetName, subsampleKeyHere = bnSpOrigInfo.computeRandomSubsets("match for comparisons", numTrialsPerCond = maxNumTrlPerCond[bssInd] if type(maxNumTrlPerCond) is list else maxNumTrlPerCond, numChannels = maxNumNeuron, labelName = labelName, numSubsets = numSubsamples, returnInfo = True)

        # here we check if any subsamples can be formed given the limitations,
        # and only add on subsamples to the list if they were formed
        if len(bsi[subsampleKeyHere]):
            brainArea = dsi[bnSpOrigInfo].fetch('brain_area')[0] # returns array, grab the value (string)
            task = dsi[bnSpOrigInfo].fetch('task')[0]

            bnSpSubsamples.append(bnSpHereSubsample)
            trlNeurSubsamples.append(trlNeurHereSubsamples)
            subsampleExpressions.append(bsi[subsampleKeyHere])
            subsmpKeys.append(subsampleKeyHere)
            datasetNames.append(datasetName)
            brainAreas.append(brainArea)
            tasks.append(task)






#    np.random.set_state(initRandSt)

    return bnSpSubsamples, subsampleExpressions, datasetNames, brainAreas, tasks
    

#%% Analyses on binned spike sets
def generateLabelGroupStatistics(binnedSpikesListToGroup, labelUse = 'stimulusMainLabel'):
    
    groupedSpikesAll = []
    groupedSpikesTrialAvgAndSem = []
    grpLabels = []
    for binnedSpikes in binnedSpikesListToGroup:
    
        # dataSt = data[dataInd]['dataset'].successfulTrials().trialsWithoutCatch()
        if type(binnedSpikes) is list:
            stBinsUn = np.unique([bnSp.alignmentBins[0] for bnSp in binnedSpikes])
            if stBinsUn.size>1:
                raise(Exception("Ruh roh dunno what we're doing here"))
                pass
            else:
                minBins = np.min([bnSp.shape[1] for bnSp in binnedSpikes])
                # note that all alignment bins are from the start, so if the 
                # start doesn't change we don't need to change anything with them...
                start,end,label,alignmentBins = zip(*[(bnSp.start, bnSp.end, bnSp.labels[labelUse],bnSp.alignmentBins) for bnSp in binnedSpikes])
                binnedSpikes = BinnedSpikeSet(np.stack([bnSp[:, :minBins] for bnSp in binnedSpikes]),
                                              start = list(start),
                                              end = list(end),
                                              binSize = binnedSpikes[0].binSize,
                                              labels = {labelUse:np.stack(label)},
                                              alignmentBins = list(alignmentBins))
            
            

#        uniqueTargAngle = np.unique(binnedSpikes.labels[labelUse].astype('float64'), axis=0)
    
        groupedSpikes, uniqueLabel = binnedSpikes.groupByLabel(binnedSpikes.labels[labelUse].astype('float64'))
        
        alignmentBinsAll = binnedSpikes.alignmentBins[0] if binnedSpikes.alignmentBins is not None else None
        stAll = binnedSpikes.start
        endAll = binnedSpikes.end 
        try:
            targAvgList, targSemList = zip(*[(groupedSpikes[targ].trialAverage(), groupedSpikes[targ].trialSem()) for targ in range(0, len(groupedSpikes))])
    #        breakpoint()

            targTmTrcAvgArr = BinnedSpikeSet(np.stack(targAvgList), start=stAll, end = endAll, binSize = binnedSpikes.binSize, labels = {labelUse: uniqueLabel}, alignmentBins = alignmentBinsAll)
            targTmTrcSemArr = BinnedSpikeSet(np.stack(targSemList), start=stAll, end = endAll, binSize = binnedSpikes.binSize, labels = {labelUse: uniqueLabel}, alignmentBins = alignmentBinsAll)
        except ValueError as err:
            breakpoint()
            groupedSpikesTrialAvgAndSem.append([])
        else:
            groupedSpikesTrialAvgAndSem.append([targTmTrcAvgArr, targTmTrcSemArr])

        groupedSpikesAll.append(groupedSpikes)
        grpLabels.append(uniqueLabel)
    
    return groupedSpikesTrialAvgAndSem, groupedSpikesAll, grpLabels
    

def dimensionalityComputation(listBSS, descriptions, labelUse='stimulusMainLabel', maxDims = 30, baselineSubtract = True, useAllGroupings = False, minWiGroupFR = 0.5, numberNeuronsMatch = None, plot=True):
    numDims = []
    for bnSp, description in zip(listBSS, descriptions):
        # grpSpikes = bnSp.groupByLabel(datasets[idx].markerTargAngles)
        # [bnSpGrp.numberOfDimensions(title = data[idx]['description'], maxDims = 30, baselineSubtract = True) for bnSpGrp in grpSpikes]
        if labelUse == None: #NOTE we shouldn't get here... and not sure what happens if we do...
            labels = None;
        else:
            labels = bnSp.labels[labelUse]
            
        if not useAllGroupings:
            singleGroupSpikes = bnSp.groupByLabel(labels, labelExtract = labels[0]) # we're randomly going to choose trials with label = 0
            bnSp = singleGroupSpikes[0]
            labels = None
            
        if numberNeuronsMatch is not None: # it'll be the number neurons to match then...
            bnSp, _ = bnSp.channelsAboveThresholdFiringRate(minWiGroupFR)
            chansUse = np.random.choice(bnSp.shape[1], numberNeuronsMatch, replace=False)
            bnSp = bnSp[:,chansUse]
            
        try:
            numDimHere = bnSp.numberOfDimensions(title = description, labels=labels, maxDims = maxDims, baselineSubtract = baselineSubtract, plot = plot)
        except Exception as faCheck:
            if faCheck.args[0] != "FA:NumObs":
                raise faCheck
            else:
                print(faCheck.args[1])
                numDimHere = None
                
        numDims.append(numDimHere)
    #    numDims.append(bnSp.numberOfDimensions(title = data[idx]['description'], labels=None, maxDims = 30))
    return numDims

def pcaComputation(listBSS, plot=True):
    pass

def ldaComputation(listBSS, plot=True):
    pass


# note that timeBeforeAndAfterStart asks about what the first and last points of
# the binned spikes represent as it relates to the first and last alignment point
# i.e. the default (0,250) tells us that the first bin is aligned with whatever
# alignment was used to generate this spike set (say, the delay start), and should
# be plotted out to 250ms from that point. The default timeBeforeAndAfterEnd
# of (-250,0) tells us that the *last* bin is aligned at the alignment point
# for the end (whatever that may be--say, the delay end), and should be plotted
# starting -250 ms before
def gpfaComputation(bssExp, timeBeforeAndAfterStart = None, timeBeforeAndAfterEnd = None, labelUse = 'stimulusMainLabel', balanceConds = True, computeResiduals = True, conditionNumbers = 1,combineConditions = False, sqrtSpikes = False, forceNewGpfaRun=False,
                    crossvalidateNumFolds = 4, xDimTest = [2,5,8], overallFiringRateThresh=0.5, perConditionGroupFiringRateThresh = 0.5,plotOutputInfo=None, units='count', useFa = False, relativePathsToSave = None):
    
    # from classes.GPFA import GPFA
    from mpl_toolkits.mplot3d import Axes3D # for 3d plotting
    from methods.GeneralMethods import prepareMatlab
    from multiprocessing import pool
    

    gai = GpfaAnalysisInfo()
    gap = GpfaAnalysisParams()
    bsi = BinnedSpikeSetInfo()
    dsi = DatasetInfo()


    gpfaParams = dict(
        method_used = 'gpfa' if not useFa else 'fa',
        overall_fr_thresh = overallFiringRateThresh,
        balance_conds = balanceConds,
        sqrt_spikes = sqrtSpikes,
        num_conds = 1 if not combineConditions else (0 if conditionNumbers is None else conditionNumbers),
        combine_conditions = 'no' if not combineConditions else ('all' if conditionNumbers is None else 'subset'),
        num_folds_crossvalidate = crossvalidateNumFolds,
        on_residuals = computeResiduals,
        units = units
    )


    if type(bssExp) is not list:
        bssExp = [bssExp]
    
    # Um.... DataJoint can apparently handle bssExp being a list of expressions
    # and that's nuts. Unfortunately for the nuttiness, I *do* need to be able
    # to keep these grouped together.
    gpfaRes = []
    gpfaInfo = []
    gpfaResBest = []
    # lstSzs = []

    dimExp = []
    dimMoreLL = []
    scoreAll = []
    gpfaOutDimAll = []
    gpfaTestIndsOutAll = []
    gpfaTrainIndsOutAll = []
    gpfaBinSize = []
    gpfaCondLabels = []
    gpfaAlignmentBins = []
    gpfaDimsTested = []

    # ** first, compute GPFA using the initial test dimensionalities **
    for expNum, subExp in enumerate(bssExp):
        if type(subExp) is BinnedSpikeSet:
            pthsOut = []
            relativePathToSave = relativePathsToSave[expNum]
        for idxXdim,dim in enumerate(xDimTest):
            gpfaParams.update(dict(dimensionality = dim))
            print("Testing/loading dimensionality %d. Left to test: " % dim + (str(xDimTest[idxXdim+1:]) if idxXdim+1<len(xDimTest) else "none"))
            if type(subExp) is BinnedSpikeSet:
                if relativePathToSave is None:
                    raise('When gpfaComputation is run on a BinnedSpikeSet, relativePathToSave must be set')
                defaultParams = loadDefaultParams()
                dataPath = Path(defaultParams['dataPath'])

                gpParamsInDb = gap[gpfaParams]
                if len(gpParamsInDb) > 0:
                    paramsHash = gpParamsInDb.hash()
                else:
                    gpfaParamsSpecific = gpfaParams.copy()
                    gpfaParamsSpecific.pop('dimensionality')
                    gpParamsAnyForAllDims = gap[gpfaParamsSpecific].fetch(as_dict=True)
                    if len(gpParamsAnyForAllDims) == 0:
                        gpParamsAnyForAllDims = gpfaParamsSpecific
                    else:
                        gpParamsAnyForAllDims = gpParamsAnyForAllDims[0]
                    gpParamsAnyForAllDims.update({'dimensionality': dim})
                    gpParamsAnyForAllDims.pop('gpfa_params_id', None)
                    paramsHash = hashlib.md5(json.dumps(gpParamsAnyForAllDims, sort_keys=True).encode('ascii')).hexdigest()

                outputPathToConditions = dataPath / relativePathToSave / 'gpfa' / 'params_{}'.format(paramsHash[:5]) 
                gpfaPrepInputs = dict(labelUse = labelUse, condNums=conditionNumbers, combineConds = combineConditions, overallFiringRateThresh = overallFiringRateThresh, perConditionGroupFiringRateThresh = perConditionGroupFiringRateThresh,
                    balanceConds = balanceConds, computeResiduals = computeResiduals)
                groupedBalancedSpikes, condDescriptors, condsUse, *_ = subExp.prepareGpfaOrFa(**gpfaPrepInputs)
                if not useFa:
                    retVals = subExp.gpfa(groupedBalancedSpikes, outputPathToConditions, condDescriptors, dim, labelUse, crossvalidateNum = crossvalidateNumFolds, forceNewGpfaRun = forceNewGpfaRun) 
                else:
                    retVals = subExp.fa(groupedBalancedSpikes, outputPathToConditions, condDescriptors, dim, labelUse, crossvalidateNum = crossvalidateNumFolds) 
                pthsOut.append(retVals[-1])
            else:
                if len(gap[gpfaParams]) == 0:
                    gap.insert1(gpfaParams)


                # if dimensionality != 0:#True: #not useFa:
                if not forceNewGpfaRun:
                    bssExpComputed = subExp[gai[gap[gpfaParams]]]
                    bssExpToCompute = subExp - gai[gap[gpfaParams]]
                else:
                    bssExpToCompute = subExp

                # note that this *adds* values to GpfaAnalysisInfo, so we can't
                # just filter gai by bssExpToCompute (nothing will be there!)
                gai.computeGpfaResults(gap[gpfaParams], bssExpToCompute, labelUse=labelUse, conditionNumbersGpfa = conditionNumbers, perCondGroupFiringRateThresh = perConditionGroupFiringRateThresh, useFa=useFa, forceNewGpfaRun=forceNewGpfaRun)


        # this is gonna filter the GpfaAnalysisParams (gap) which will be used to
        # filter the infos
        # note that the dict function apparently does a gpfaParams.copy() first
        gpfaParamsAll = [dict(gpfaParams, dimensionality = d) for d in xDimTest]

        # ** second, grab the results of these GPFA runs **
    # for subExp in bssExp: # merged to loop above
        # lstSzs.append(len(gai[subExp]))
        if type(subExp) is BinnedSpikeSet:
            gpfaResHere = {}
            for idxXdim,(dim,pthsGpfa) in enumerate(zip(xDimTest, pthsOut)):
                for pthGpfa, condUse in zip(pthsGpfa, condsUse):
                    # condStr = str(pthGpfa.parent.relative_to(pthGpfa.parent.parent))
                    # condNum = condStr[4:]
                    # condNumList = '[' + condNum + ']' # just matching what happens on the from-database pathway >.>
                    import re
                    condNumStr = re.sub(' ', ',',str(np.array(condUse)))
                    relPathAndCond = (str(relativePathToSave), str(crossvalidateNumFolds), condNumStr)
                    if relPathAndCond not in gpfaResHere:
                        gpfaResHere[relPathAndCond] = {}
                    gpfaResHere[relPathAndCond]['condition'] = np.array(condUse) # NOTE was int(condNum)
                    with np.load(pthGpfa, allow_pickle=True) as gpfaResSaved:
                        gpfaResLoaded = dict(
                            zip((k for k in gpfaResSaved), (gpfaResSaved[k] for k in gpfaResSaved))
                        )
                    gpfaResHere[relPathAndCond][dim] = gpfaResLoaded
                    gpfaInfoHere = [] # for now... I don't think I use this anymore...
        else:
            gpfaResHere, gpfaInfoHere = gai[subExp][gap[gpfaParamsAll]].grabGpfaResults(returnInfo=True, useFa=useFa)

        gpfaRes.append(gpfaResHere)
        gpfaInfo.append(gpfaInfoHere)

        cvApproach = "logLikelihood"
        shCovThresh = 0.95


    # ** third, find the best dimensionality by looking between the test dimensionalities
    # for gpfaInfoHere, gpfaResHere in zip(gpfaInfo, gpfaRes):
        bestDims = computeBestDimensionality(gpfaResHere, cvApproach = cvApproach, shCovThresh = shCovThresh)
        bssPth = np.unique([pthCvCnd[0] for pthCvCnd in gpfaResHere.keys()])
        assert bssPth.size == 1, "Not sure why there are multiple (or no O.o) paths for one GPFA run..."
        bssNewDimExp = bsi[{'bss_relative_path' : str(bssPth[0])}]

        # Can't say I reeeally like eval here, but I think I gotta do it...
        condNumsTested = [eval(pthAndCond[2]) for pthAndCond in gpfaResHere.keys()]

        # NOTE by starting at -1 that means dimensionalities of 0 will be
        # tested! MAKE SURE THIS WORKS!
        xDimRangeAroundTest = [-1] + xDimTest + [(xDimTest[-1]+3)]
        testDimIndInRange = [int(np.nonzero(xDimRangeAroundTest==dm)[0]) for dm in bestDims]
        newDimsToTestPerCond = [np.arange(xDimRangeAroundTest[dmInd-1]+1, xDimRangeAroundTest[dmInd+1]) for dmInd in testDimIndInRange]
        newDimsToTestPerCond = [nD[nD!=cD].tolist() for nD, cD in zip(newDimsToTestPerCond, bestDims)]

        for newDimTestCond, condNum in zip(newDimsToTestPerCond, condNumsTested):

            for newDimTest in newDimTestCond:
                gpfaParams.update(dict(
                    dimensionality = newDimTest,
                    ))
                if type(subExp) is BinnedSpikeSet:
                    defaultParams = loadDefaultParams()
                    dataPath = Path(defaultParams['dataPath'])

                    gpParamsInDb = gap[gpfaParams]
                    if len(gpParamsInDb) > 0:
                        paramsHash = gpParamsInDb.hash()
                    else:
                        gpfaParamsSpecific = gpfaParams.copy()
                        gpfaParamsSpecific.pop('dimensionality')
                        gpParamsAnyForAllDims = gap[gpfaParamsSpecific].fetch(as_dict=True)
                        if len(gpParamsAnyForAllDims) == 0:
                            gpParamsAnyForAllDims = gpfaParamsSpecific
                        else:
                            gpParamsAnyForAllDims = gpParamsAnyForAllDims[0]
                        gpParamsAnyForAllDims.update({'dimensionality': newDimTest})
                        gpParamsAnyForAllDims.pop('gpfa_params_id', None)
                        paramsHash = hashlib.md5(json.dumps(gpParamsAnyForAllDims, sort_keys=True).encode('ascii')).hexdigest()

                    outputPathToConditions = dataPath / relativePathToSave / 'gpfa' / 'params_{}'.format(paramsHash[:5]) 
                    # I feel like this was already computed above, and redoing
                    # here just leaves room for errors...
                    # gpfaPrepInputs = dict(labelUse = labelUse, condNums=[0], combineConds = combineConditions, overallFiringRateThresh = overallFiringRateThresh, perConditionGroupFiringRateThresh = perConditionGroupFiringRateThresh,
                        # balanceConds = balanceConds, computeResiduals = computeResiduals)
                    # groupedBalancedSpikes, condDescriptors, condsUse, *_ = subExp.prepareGpfaOrFa(**gpfaPrepInputs)
                    grpSpikesRun = np.nonzero(np.array([c==condNum for c in condsUse]))[0][0]
                    if not useFa:
                        retVals = subExp.gpfa(groupedBalancedSpikes[grpSpikesRun:grpSpikesRun+1], outputPathToConditions, condDescriptors[grpSpikesRun:grpSpikesRun+1], int(newDimTest), labelUse, crossvalidateNum = crossvalidateNumFolds, forceNewGpfaRun = forceNewGpfaRun) 
                    else:
                        retVals = subExp.fa(groupedBalancedSpikes[grpSpikesRun:grpSpikesRun+1], outputPathToConditions, condDescriptors[grpSpikesRun:grpSpikesRun+1], int(newDimTest), labelUse, crossvalidateNum = crossvalidateNumFolds)

                    pthNewDim = retVals[-1]

                    gpfaResH = {}
                    for pthND in pthNewDim:
                        # condStr = str(pthND.parent.relative_to(pthND.parent.parent))
                        # condNumStr = condStr[4:]
                        # condNumList = '[' + condNumStr + ']'
                        import re
                        condNumStr = re.sub(' ', ',',str(np.array(condNum)))
                        relPathAndCond = (str(relativePathToSave), str(crossvalidateNumFolds), condNumStr)
                        if relPathAndCond not in gpfaResH:
                            gpfaResH[relPathAndCond] = {}
                        gpfaResH[relPathAndCond]['condition'] = np.array(condNum)
                        with np.load(pthND, allow_pickle=True) as gpfaResSaved:
                            gpfaResLoaded = dict(
                                zip((k for k in gpfaResSaved), (gpfaResSaved[k] for k in gpfaResSaved))
                            )
                        gpfaResH[relPathAndCond][newDimTest] = gpfaResLoaded
                        gpfaInfoHereNew = [] # for now... I don't think I use this anymore...
                        keyRes = relPathAndCond
                else:
                    if len(gap[gpfaParams]) == 0:
                        gap.insert1(gpfaParams)

                    gaiCompleted = gai[bssNewDimExp][gap[gpfaParams]]['gpfa_rel_path_from_bss LIKE "%cond{}{}{}%"'.format('s' if len(condNum)>1 else '', '-'.join([str(cN) for cN in condNum]), 'Grpd' if gpfaParams['combine_conditions'] != 'no' else '')]

                    if len(gaiCompleted) == 0 or forceNewGpfaRun:
                        # DOUBLE NOTE: FA can now run separately from GPFA, so we're
                        # back to setting useFa
                        # NOTE: we set useFa to False here because GPFA *HAS* to be
                        # run to get it input into the database... not the best
                        # work flow, but FA has never been in my code ;_;
                        gai.computeGpfaResults(gap[gpfaParams], bssNewDimExp, labelUse=labelUse, conditionNumbersGpfa = None if gpfaParams['combine_conditions'] == 'all' else condNum, perCondGroupFiringRateThresh = perConditionGroupFiringRateThresh, useFa=useFa, forceNewGpfaRun=forceNewGpfaRun) 

                    gpfaResH, gpfaInfoH = gaiCompleted.grabGpfaResults(returnInfo=True, useFa=useFa)
                    keyRes = list(gpfaResH.keys())
                    assert len(keyRes)==1, "Should have one (and only one) key from the new dim runs..."

                    keyRes = keyRes[0]
                gpfaResHere[keyRes].update(gpfaResH[keyRes])
                if len(gpfaInfoHere)>0:
                    gpfaInfoHereNew = {k:gih1 + gih2 for  (k, gih1), (_, gih2) in zip(gpfaInfoHere.items(), gpfaInfoH.items())}
                    gpfaInfoHere.update(gpfaInfoHereNew)


    # ** fourth, run GPFA on this best dimensionality, but now using all the data (which is what num_folds_crossvalidate = 1 does)
    # for gpfaInfoHere, gpfaResHere in zip(gpfaInfo, gpfaRes):
        gpfaCrunchedResults = crunchGpfaResults(gpfaResHere, cvApproach = cvApproach, shCovThresh = shCovThresh)

        bssPaths = [pthAndCond[0] for pthAndCond in gpfaCrunchedResults.keys()]
        # Can't say I reeeally like eval here, but I think I gotta do it...
        condNumsTested = [eval(pthAndCond[2]) for pthAndCond in gpfaCrunchedResults.keys()]

        dimBestLogLikelihood = [d['xDimScoreBestAll'] for _, d in gpfaCrunchedResults.items()]

        gpfaResBestExp = {}
#        gpfaParams.update(dict(num_folds_crossvalidate = 1))

        for dimHere, bssPath, cond in zip(dimBestLogLikelihood, bssPaths, condNumsTested):
            dimHere = dimHere[0]
            gpfaParams.update(dict(dimensionality=dimHere))
            gpfaAnalysisInfoConds = gai['bss_relative_path="%s"' % bssPath]
            bssExpWithGpfa = bsi['bss_relative_path="%s"' % bssPath]

            gpfaAnalysisInfoConds = gpfaAnalysisInfoConds[gap[gpfaParams]]
            # hsh, condNumsAllDb = gpfaAnalysisInfoConds.fetch('gpfa_hash', 'condition_nums')
            # if len(cond)>1:
                # Mmk, I think this'll work; basically here we're checking
                # about combined conditions, which makes cond have more than
                # one value; to do so, we need to loop through the returned
                # condNumsAllDb and check each array individually (well, all
                # values in the array)
                # assert np.sum([np.all(cnd == cond) for cnd in condNumsAllDb]) == 1, "Too many GPFA analysis infos fit these parameters"
                # hshUseLog = np.array([np.all(cnd == cond) for cnd in condNumsAllDb])
                # hshThisCond = hsh[hshUseLog][0] # haven't quite thought through the why of this [0] index... but I think we'd need it even if multiple conditions were used
            # else:
                # assert len(hsh[condNumsAllDb == cond]) == 1, "Too many GPFA analysis infos fit these parameters"
                # hshThisCond = hsh[condNumsAllDb == cond][0] # haven't quite thought through the why of this [0] index... but I think we'd need it even if multiple conditions were used
            # gpfaAnalysisInfoThisCondDim = gpfaAnalysisInfoConds[dict(gpfa_hash = hshThisCond)]
            # gpParams = gap[gpfaAnalysisInfoThisCondDim]

            gpfaParamsNoCval = gpfaParams.copy()
            # note that 1-fold crossvalidation is effectively no
            # crossvalidation--which is what we want here because we've gotten
            # our optimal dimensionality to fit on!
            num_folds_crossvalidate = 1
            gpfaParamsNoCval.update({'num_folds_crossvalidate' : num_folds_crossvalidate})

            if type(subExp) is BinnedSpikeSet:
                defaultParams = loadDefaultParams()
                dataPath = Path(defaultParams['dataPath'])

                gpParamsInDb = gap[gpfaParamsNoCval]
                if len(gpParamsInDb) > 0:
                    paramsHash = gpParamsInDb.hash()
                else:
                    gpfaParamsSpecific = gpfaParamsNoCval.copy()
                    noCvDim = gpfaParamsSpecific.pop('dimensionality')
                    gpParamsAnyForAllDims = gap[gpfaParamsSpecific].fetch(as_dict=True)
                    if len(gpParamsAnyForAllDims) == 0:
                        gpParamsAnyForAllDims = gpfaParamsSpecific
                    else:
                        gpParamsAnyForAllDims = gpParamsAnyForAllDims[0]
                    gpParamsAnyForAllDims.update({'dimensionality': noCvDim})
                    gpParamsAnyForAllDims.pop('gpfa_params_id', None)
                    paramsHash = hashlib.md5(json.dumps(gpParamsAnyForAllDims, sort_keys=True).encode('ascii')).hexdigest()

                outputPathToConditions = dataPath / relativePathToSave / 'gpfa' / 'params_{}'.format(paramsHash[:5]) 
                # gpfaPrepInputs = dict(labelUse = labelUse, condNums=conditionNumbers, combineConds = combineConditions, overallFiringRateThresh = overallFiringRateThresh, perConditionGroupFiringRateThresh = perConditionGroupFiringRateThresh,
                #     balanceConds = balanceConds, computeResiduals = computeResiduals)
                # groupedBalancedSpikes, condDescriptors, condsUse, *_ = subExp.prepareGpfaOrFa(**gpfaPrepInputs)
                grpSpikesRun = np.nonzero(np.array([c==condNum for c in condsUse]))[0][0]
                if not useFa:
                    retVals = subExp.gpfa(groupedBalancedSpikes[grpSpikesRun:grpSpikesRun+1], outputPathToConditions, condDescriptors[grpSpikesRun:grpSpikesRun+1], dimHere, labelUse, crossvalidateNum = num_folds_crossvalidate, forceNewGpfaRun = forceNewGpfaRun) 
                else:
                    retVals = subExp.fa(groupedBalancedSpikes[grpSpikesRun:grpSpikesRun+1], outputPathToConditions, condDescriptors[grpSpikesRun:grpSpikesRun+1], dimHere, labelUse, crossvalidateNum = num_folds_crossvalidate) 

                pthOneCV = retVals[-1][0] # we know there will only be one condition, so one path, here

                # condStr = str(pthOneCV.parent.relative_to(pthOneCV.parent.parent))
                # condNumStr = condStr[4:]
                # condNumList = '[' + condNumStr + ']'
                import re
                condNumStr = re.sub(' ', ',',str(np.array(cond)))
                relPathAndCond = (str(relativePathToSave), str(num_folds_crossvalidate), condNumStr)
                if relPathAndCond not in gpfaResBestExp:
                    gpfaResBestExp[relPathAndCond] = {}
                gpfaResBestExp[relPathAndCond]['condition'] = np.array(cond)
                with np.load(pthOneCV, allow_pickle=True) as gpfaResSaved:
                    gpfaResLoaded = dict(
                        zip((k for k in gpfaResSaved), (gpfaResSaved[k] for k in gpfaResSaved))
                    )
                gpfaResBestExp[relPathAndCond][dimHere] = gpfaResLoaded
                gpfaInfoHereNew = [] # for now... I don't think I use this anymore...
                keyRes = relPathAndCond
            else:
                if len(gap[gpfaParamsNoCval]) == 0:
                    gap.insert1(gpfaParamsNoCval)

                gaiComputed = gai[bssExpWithGpfa][gap[gpfaParamsNoCval]]
                gpHash, condsComputed = gaiComputed.fetch('gpfa_hash', 'condition_nums')
                if len(cond)>1:
                    # need to dtype to Bool in case there *are* no computed
                    # gpHashes so gpHshUseLog ends up empty--if that happens
                    # it has no Bool to inherit from the list, it remains a
                    # float64, and using it to index gpHash errors
                    gpHshUseLog = np.array([np.all(cnd == cond) for cnd in condsComputed], dtype='Bool')
                    gpHashComp = gpHash[gpHshUseLog]
                else:
                    gpHashComp = gpHash[condsComputed==cond]

                if len(gpHashComp):
                    gpHashComp = gpHashComp[0]

                if not forceNewGpfaRun:
                    bssExpComputed = bssExpWithGpfa[gai[gap[gpfaParamsNoCval]]['gpfa_hash="{hash}"'.format(hash=gpHashComp)]]
                    bssExpToCompute = bssExpWithGpfa - gai[gap[gpfaParamsNoCval]]['gpfa_hash="{hash}"'.format(hash=gpHashComp)]
                else:
                    bssExpToCompute = bssExpWithGpfa

                # note that this *adds* values to GpfaAnalysisInfo, so we can't
                # just filter gai by bssExpToCompute (nothing will be there!)
                perConditionGroupFiringRateThresh = gpfaAnalysisInfoConds['condition_grp_fr_thresh'][0]
                labelUse = gpfaAnalysisInfoConds['label_use'][0]
                # DOUBLE NOTE: FA now can run separately from GPFA, so now we use
                # input useFa
                # NOTE: we set useFa to FALSE here because EVEN IF we're extracting
                # the FA parameters, GPFA still needs to be run (the way that this
                # is constructed...)
                gai.computeGpfaResults(gap[gpfaParamsNoCval], bssExpToCompute, labelUse=labelUse, conditionNumbersGpfa = None if gpfaParamsNoCval['combine_conditions'] == 'all' else cond, perCondGroupFiringRateThresh = perConditionGroupFiringRateThresh, useFa=useFa, forceNewGpfaRun=forceNewGpfaRun)

                # Not the prettiest, but we rerun this to account for the fact
                # that the correct gpfaHash is now different...
                gaiComputed = gai[bssExpWithGpfa][gap[gpfaParamsNoCval]]
                gpHash, condsComputed = gaiComputed.fetch('gpfa_hash', 'condition_nums')
                if len(cond)>1:
                    gpHshUseLog = np.array([np.all(cnd == cond) for cnd in condsComputed])
                    gpHashComp = gpHash[gpHshUseLog]
                else:
                    gpHashComp = gpHash[condsComputed==cond]

                if len(gpHashComp):
                    gpHashComp = gpHashComp[0]

                bssExpComputed = bssExpWithGpfa[gai[gap[gpfaParamsNoCval]]['gpfa_hash="{hash}"'.format(hash=gpHashComp)]]



                gpfaResBestCond, gpfaInfoBestCond = gai[bssExpComputed]['gpfa_hash="{hash}"'.format(hash=gpHashComp)][gap[gpfaParamsNoCval]].grabGpfaResults(returnInfo=True, order=True, useFa=useFa)
                gpfaResBestExp.update(gpfaResBestCond)

        gpfaResBest.append(gpfaResBestExp)

    # ** finally, reformat the results for output and plotting **
    # for gpfaResHere, gpfaInfoHere in zip(gpfaResBest, gpfaInfo):
    #     gpfaCrunchedResults = crunchGpfaResults(gpfaResHere, cvApproach = cvApproach, shCovThresh = shCovThresh)
        gpfaCrunchedResults = crunchGpfaResults(gpfaResBestExp, cvApproach = cvApproach, shCovThresh = shCovThresh)

        bssPaths = [pthAndCond[:2] for pthAndCond in gpfaCrunchedResults.keys()]
        _, smDimGrph = np.unique(bssPaths, return_inverse=True, axis=0)
        dimExp.append([d['xDimBestAll'] for _, d in gpfaCrunchedResults.items()])
        dimMoreLL.append([d['xDimScoreBestAll'] for _, d in gpfaCrunchedResults.items()])
        scoreAll.append([d['normalGpfaScoreAll'] for _, d in gpfaCrunchedResults.items()])
        gpfaOutDimAll.append([d['dimResults'] for _, d in gpfaCrunchedResults.items()])
        gpfaTestIndsOutAll.append([d['testInds'] for _, d in gpfaCrunchedResults.items()])
        gpfaTrainIndsOutAll.append([d['trainInds'] for _, d in gpfaCrunchedResults.items()])
        gpfaBinSize.append([d['binSize'] for _, d in gpfaCrunchedResults.items()])
        gpfaCondLabels.append([d['condLabel'] for _, d in gpfaCrunchedResults.items()])
        gpfaAlignmentBins.append([d['alignmentBins'] for _, d in gpfaCrunchedResults.items()])
        gpfaDimsTested.append([d['dimsTest'] for _, d in gpfaCrunchedResults.items()])

    if plotOutputInfo is not None and plotOutputInfo:
        if type(subExp) is BinnedSpikeSet:
            descriptions = plotOutputInfo['descriptions']
            brainAreas = plotOutputInfo['brainAreas']
            bssPlotExp = None
        else:
            descriptions = [gI['datasetNames'] for gI in gpfaInfo]
            brainAreas = np.unique(dsi[bssExp]['brain_area'])
            bssPlotExp = bssExp
        # plotGpfaResults(bssExp, gpfaRes, useFa, crossvalidateNumFolds, timeBeforeAndAfterStart, timeBeforeAndAfterEnd)
        plotGpfaResults(gpfaRes, descriptions, brainAreas, timeBeforeAndAfterStart, timeBeforeAndAfterEnd, bssExp = bssPlotExp)

    return dimExp, dimMoreLL, gpfaOutDimAll, gpfaTestIndsOutAll, gpfaTrainIndsOutAll, gpfaAlignmentBins, scoreAll, gpfaCondLabels

def incaComputation(bssExpConditionPairs, numNeuronsInNetwork = 25, cvalFracTrain = 0.75, numCrossvalidations = 4, xDimInNetTest = [[[2,2],[5,5]],[[5,5],[5,5]]], plotOutput=True):
    
    from classes.InputNetworkCovarAnalysis import INCA
    bsi = BinnedSpikeSetInfo()
    dsi = DatasetInfo()

    dimResAll = []
    for bssPair in bssExpConditionPairs:
        print("** Running dataset '{}' **".format(dsi[bssPair[0]]['dataset_name'][0]))

        bssCondA = bssPair[0].grabBinnedSpikes()[0]
        bssCondB = bssPair[1].grabBinnedSpikes()[0]


        if bssCondA.shape[1] != bssCondB.shape[1]:
            raise Exception("Data from both conditions have to come from the same set of neurons! And at the very least here the populations have different numbers of neurons!")

        if numNeuronsInNetwork*2 > bssCondA.shape[1]:
            raise Exception("For the moment network subsets chosen will be for *distinct* populations of neurons, so the number of neurons in the binned spike set must be at least twice the size of the explore network!")

        randNeurs = np.random.permutation(bssCondA.shape[1])
        neursInNetX = randNeurs[:numNeuronsInNetwork]
        neursInNetY = randNeurs[numNeuronsInNetwork:2*numNeuronsInNetwork]
        bssNetXCondA = bssCondA[:,neursInNetX]
        bssNetXCondB = bssCondB[:,neursInNetX]
        bssNetYCondA = bssCondA[:,neursInNetY]
        bssNetYCondB = bssCondB[:,neursInNetY]
        pairInca = INCA(bssNetXCondA, bssNetXCondB, bssNetYCondA, bssNetYCondB)

        dimRes = []
        for xDimInNet in xDimInNetTest:
            print("Testing inputs dims {} and network dims {}".format(xDimInNet[0], xDimInNet[1]))
            res = pairInca.runIncaInMatlab(Path('test'), xDimInNet = xDimInNet, cvalFracTrain = cvalFracTrain, crossvalidationNum = numCrossvalidations)
            dimRes.append(res)


        dimResAll.append(dimRes)

    return dimResAll

# NOTE: this function does not act on any of the binned spikes (computing
# residuals, etc) before computing the relevant r_sc vals
def rscComputations(listBSS,descriptions, labelUse, separateNoiseCorrForLabels = True, normalize = False, plotResid = False):

    residCorrMeanOverCondAll = []
    residCorrPerCondAll = []
    geoMeanOverall = []

    residCorrPerCondOBPTAll = []
    residCorrMeanOverCondOBPTAll = []
    for idxSpks, (bnSp, description) in enumerate(zip(listBSS,descriptions)):

        if plotResid:
            scatterFig = plt.figure()
            if type(bnSp) is list:
                scatterFig.suptitle(description + " geo mean FR vs corr var multiple subsets")
            else:
                scatterFig.suptitle(description + " geo mean FR vs corr var")
        if type(bnSp) is not list:
            titleFig = description
            bnSp = [bnSp] # make it into a list--it's a subset of one!
            titleCorrTemplate = description 
            titleCorrOBPTTemplate= description + " 1 bin/trl"
        else:
            titleCorrTemplate = description + " subset %d"
            titleCorrOBPTTemplate= description + " 1 bin/trl subset %d"

        residCorrMeanOverCondAllHere = []
        residCorrPerCondAllHere = []
        geoMeanOverallHere = []
            
        residCorrPerCondOBPTAllHere = []
        residCorrMeanOverCondOBPTAllHere = []
        for idxSubset, bS in enumerate(bnSp):
            # All these metrics are taken on spike counts!
            bS = bS.convertUnitsTo('count')
            # for r_sc using one bin per trial, we rebin everything to... one
            # bin per trial
            if bS.dtype == 'object':
                trialLengthMs = bS.binSize*np.array([bnSpTrl[0].shape[0] for bnSpTrl in bS])
            else:
                trialLengthMs = np.array([bS.shape[2]*bS.binSize])

            try:
                titleCorr = titleCorrTemplate % idxSubset
                titleCorrOBPT = titleCorrOBPTTemplate % idxSubset
            except TypeError as e:
                titleCorr = titleCorrTemplate 
                titleCorrOBPT = titleCorrOBPTTemplate 

            labels = bS.labels[labelUse]

            bSOneBinPrTrl = bS.convertUnitsTo('count').increaseBinSize(trialLengthMs)

            residualSpikes, residCorrMeanOverCond, residCorrPerCond, geoMeanCntAll = bS.convertUnitsTo('count').residualCorrelations(labels, plot=plotResid, separateNoiseCorrForLabels = separateNoiseCorrForLabels, figTitle = titleCorr, normalize = normalize)
            residualSpikesOBPT, residCorrMeanOverCondOBPT, residCorrPerCondOBPT, geoMeanCntAllOBPT = bSOneBinPrTrl.residualCorrelations(labels, plot=plotResid, separateNoiseCorrForLabels = separateNoiseCorrForLabels, figTitle = titleCorrOBPT, normalize = normalize)
            
            
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
                
        residCorrMeanOverCondAll.append(residCorrMeanOverCondAllHere)
        geoMeanOverall.append(geoMeanOverallHere)
        residCorrPerCondAll.append(residCorrPerCondAllHere)
        
        residCorrPerCondOBPTAll.append(residCorrPerCondOBPTAllHere)
        residCorrMeanOverCondOBPTAll.append(residCorrMeanOverCondOBPTAllHere)


        



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
    fRChStdMnByArea = []
    fRChMnStdByArea = []
    fRChStdStdByArea = []
    fRChPrBinMnByArea = []
    fRChPrBinStdMnByArea = []
    fRChPrBinMnStdByArea = []
    fRChPrBinStdStdByArea = []
    fanoFactorChMnByArea = []
    fanoFactorChStdByArea = []
    fanoFactorChOBTMnByArea = []
    fanoFactorChOBTStdByArea = []
    labelUse = 'stimulusMainLabel'
    for bnSpCnt in listBSS:
        fRChMnByAreaHere = []
        fRChStdMnByAreaHere = []
        fRChMnStdByAreaHere = []
        fRChStdStdByAreaHere = []
        fRChPrBinMnByAreaHere = []
        fRChPrBinStdMnByAreaHere = []
        fRChPrBinMnStdByAreaHere = []
        fRChPrBinStdStdByAreaHere = []
        fanoFactorChMnByAreaHere = []
        fanoFactorChStdByAreaHere = []
        fanoFactorChOBTMnByAreaHere = []
        fanoFactorChOBTStdByAreaHere = []
        for bSC in bnSpCnt:
            # All these metrics are taken on spike counts!
            bSC = bSC.convertUnitsTo('count')

            if separateNoiseCorrForLabels:
                grpSpkCnt, uniqueLabel = bSC.groupByLabel(bSC.labels[labelUse])
            else:
                grpSpkCnt = [bSC]
            fRChMnByAreaHere.append([np.mean(gSC.avgFiringRateByChannel()) for gSC in grpSpkCnt])
            fRChStdMnByAreaHere.append([np.std(gSC.avgFiringRateByChannel()) for gSC in grpSpkCnt])
            fRChMnStdByAreaHere.append([np.mean(gSC.stdFiringRateByChannel()) for gSC in grpSpkCnt])
            fRChStdStdByAreaHere.append([np.std(gSC.stdFiringRateByChannel()) for gSC in grpSpkCnt])
            fRChPrBinMnByAreaHere.append([np.mean(gSC.convertUnitsTo('Hz').avgValByChannelOverBins()) for gSC in grpSpkCnt])
            fRChPrBinStdMnByAreaHere.append([np.std(gSC.convertUnitsTo('Hz').avgValByChannelOverBins()) for gSC in grpSpkCnt])
            fRChPrBinMnStdByAreaHere.append([np.mean(gSC.convertUnitsTo('Hz').stdValByChannelOverBins()) for gSC in grpSpkCnt])
            fRChPrBinStdStdByAreaHere.append([np.std(gSC.convertUnitsTo('Hz').stdValByChannelOverBins()) for gSC in grpSpkCnt])
            fanoFactorChMnByAreaHere.append([np.mean(gSC.fanoFactorByChannel()) for gSC in grpSpkCnt])
            fanoFactorChStdByAreaHere.append([np.std(gSC.fanoFactorByChannel()) for gSC in grpSpkCnt])

            if bSC.dtype == 'object':
                grpTrlLenMs = [np.array([gSC.binSize*gSC[0].shape[0] for gSC in gpSpCn]) for gpSpCn in grpSpkCnt]
            else:
                grpTrlLenMs = [np.array([gSC.shape[2]*gSC.binSize]) for gSC in grpSpkCnt]
            
            fanoFactorChOBTMnByAreaHere.append([np.mean(gSC.increaseBinSize(gTLms).fanoFactorByChannel()) for gSC, gTLms in zip(grpSpkCnt, grpTrlLenMs)])
            fanoFactorChOBTStdByAreaHere.append([np.std(gSC.increaseBinSize(gTLms).fanoFactorByChannel()) for gSC, gTLms in zip(grpSpkCnt, grpTrlLenMs)])

        fRChMnByArea.append(np.stack(fRChMnByAreaHere))
        fRChStdMnByArea.append(np.stack(fRChStdMnByAreaHere))
        fRChMnStdByArea.append(np.stack(fRChMnStdByAreaHere))
        fRChStdStdByArea.append(np.stack(fRChStdStdByAreaHere))
        fRChPrBinMnByArea.append(np.stack(fRChPrBinMnByAreaHere))
        fRChPrBinStdMnByArea.append(np.stack(fRChPrBinStdMnByAreaHere))
        fRChPrBinMnStdByArea.append(np.stack(fRChPrBinMnStdByAreaHere))
        fRChPrBinStdStdByArea.append(np.stack(fRChPrBinStdStdByAreaHere))
        fanoFactorChMnByArea.append(np.stack(fanoFactorChMnByAreaHere))
        fanoFactorChStdByArea.append(np.stack(fanoFactorChStdByAreaHere))
        fanoFactorChOBTMnByArea.append(np.stack(fanoFactorChOBTMnByAreaHere))
        fanoFactorChOBTStdByArea.append(np.stack(fanoFactorChOBTStdByAreaHere))
    
    resultsDict = {
        'mean(r_{sc})' : mnCorrPerCond,
        'std(r_{sc})' : stdCorrPerCond,
        'mean(r_{sc} 1 bn/tr)' : mnCorrPerCondOBPT,
        'std(r_{sc} 1 bn/tr)' : stdCorrPerCondOBPT,
        'mean channel fano factor' : fanoFactorChMnByArea,
        'std channel fano factor' : fanoFactorChStdByArea,
        'mean channel fano factor (1 bn/tr)' : fanoFactorChOBTMnByArea,
        'std channel fano factor (1 bn/tr)' : fanoFactorChOBTStdByArea,
        'mean of mean(firing rate (Hz) of channels per trial)' : fRChMnByArea,
        'std of mean(firing rate (Hz) of channels per trial)' : fRChStdMnByArea,
        'mean of std(firing rate (Hz) of channels per trial)' : fRChMnStdByArea,
        'std of std(firing rate (Hz) of channels per trial)' : fRChStdStdByArea,
        'mean of mean(firing rate (Hz) of channels per bin)' : fRChPrBinMnByArea,
        'std of mean(firing rate (Hz) of channels per bin)' : fRChPrBinStdMnByArea,
        'mean of std(firing rate (Hz) of channels per bin)' : fRChPrBinMnStdByArea,
        'std of std(firing rate (Hz) of channels per bin)' : fRChPrBinStdStdByArea,
    }

    return resultsDict

def decodeComputations(listBSS,descriptions, labelUse):

    decodeAcc = []
    decodeAccZSc = []
    pairedDecAcc = []
    pairedSvmDprime = []
    for bnSpCnt in listBSS:
        decAccHere = []
        decAccZScHere = []
        pairedDecAccHere = []
        pairedSvmDprimeHere = []
        for bSp in bnSpCnt:
            labelOrig = bSp.labels[labelUse]
            unLabs, labelCategory = np.unique(labelOrig, axis=0, return_inverse=True)

            # compute decoding accuracy of all categories
            acc, stdAcc = bSp.decode(labels=labelCategory)
            decAccHere.append(acc)

            # compute decoding accuracy of all categories after the spikes have
            # been z-scored per-category
            accZsc, accZscStd = bSp.decode(labels=labelCategory, zScoreRespFirst = True)
            decAccZScHere.append(accZsc)

            # compute decoding accuracy for pairs of categories
            prdDecAcc = []
            prdSvmDprime = []
            for unLabNum, unLabInit in enumerate(unLabs):
                for unLabComp in unLabs[unLabNum+1:]:
                    bSpPairedLabs = bSp[np.all(labelOrig==unLabInit, axis=1) | np.all(labelOrig==unLabComp, axis=1)]
                    pairedLabelCategory = labelCategory[np.all(labelOrig==unLabInit, axis=1) | np.all(labelOrig==unLabComp, axis=1)]
                    pairedAcc, pairedAccStd = bSpPairedLabs.decode(labels=pairedLabelCategory, decodeType = 'naiveBayes', zScoreRespFirst = True)
                    prdDecAcc.append(pairedAcc)

                    dprime, _ = bSpPairedLabs.decode(labels=pairedLabelCategory, decodeType = 'linearSvm', zScoreRespFirst = True)
                    prdSvmDprime.append(dprime)

            pairedDecAccHere.append(prdDecAcc)
            pairedSvmDprimeHere.append(prdSvmDprime)

        decodeAcc.append(decAccHere)
        decodeAccZSc.append(decAccZScHere)
        pairedDecAcc.append(pairedDecAccHere)
        pairedSvmDprime.append(prdSvmDprime)

    decodeDict = {
        'main label decode accuracy' : decodeAcc,
        'main label dec acc z-sc first' : decodeAccZSc,
        'decoding accuracy of pairs of conditions' : pairedDecAcc,
        'svm dprime of pairs of conditions' : pairedSvmDprime,
    }

    return decodeDict 

def informationComputations(listBSS, labelUse):

    informationCond = []
    for bnSpCnt in listBSS:
        infoCondHere = []
        for bSp in bnSpCnt:
            labelOrig = bSp.labels[labelUse]
            unLabs, labelCategory = np.unique(labelOrig, axis=0, return_inverse=True)

            # compute Fisher information for pairs of categories
            fishInfo = []
            for unLabNum, unLabInit in enumerate(unLabs):
                for unLabComp in unLabs[unLabNum+1:]:
                    bSpPairedLabs = bSp[np.all(labelOrig==unLabInit, axis=1) | np.all(labelOrig==unLabComp, axis=1)]
                    pairedLabelCategory = labelCategory[np.all(labelOrig==unLabInit, axis=1) | np.all(labelOrig==unLabComp, axis=1)]
                    info = bSpPairedLabs.fisherInformation(labels=pairedLabelCategory)
                    fishInfo.append(info)

            # infoCondHere.append(np.stack(fishInfo))
            infoCondHere.append(fishInfo)


        informationCond.append(infoCondHere)
        # breakpoint()

    infoDict = {
        'fisher information' : [np.hstack(iC) for iC in informationCond],
    }

    return infoDict 

def slowDriftComputation(listBse, preprocessInputs, dimResultsAll = None, testInds = None, binSizeForAvg = 500, labelUse='stimulusMainLabel', plotOutput = False, chRestrictRefExp = None):
    from classes.FA import FA
    defaultParams = loadDefaultParams()
    dataPath = Path(defaultParams['dataPath'])
    
    prepInputKeywords = ['sqrtSpikes', 'labelUse', 'condNums', 'combineConds', 'overallFiringRateThresh', 'perConditionGroupFiringRateThresh', 'balanceConds', 'computeResiduals']
    preprocessInputsFinal = {}
    [preprocessInputsFinal.update({pIK : preprocessInputs[pIK]}) for pIK in prepInputKeywords if pIK in preprocessInputs]
    preprocessInputsFinal['condNums'] = preprocessInputs['conditionNumbers']
    preprocessInputsFinal['combineConds'] = preprocessInputs['combineConditions']

    slowDriftDims = []
    slowDriftSpikes = []
    slowDriftRollingAverageSpikes = []
    spikesWithSlowDriftSubtracted = []
    spikesWithSlowDriftDimOneSubtracted = []
    bsi = BinnedSpikeSetInfo()
    dsi = DatasetInfo()
    num = 0
    latentVarExpBySlowDriftByArea, slowDriftVarExpTrialsByArea, slowDriftVarExpRAByArea, slowDriftVarExpSharedSpaceByArea, rollAvgPercOrigVarByArea, rollAvgSDPercOrigVarByArea = [],[],[],[], [], []
    if plotOutput:
        if dimResultsAll is None or testInds is None:
            raise("plotting the slow drift output must provide FA/GPFA comparison")
    for bssExp, trlIndAllCv in zip(listBse,testInds):
        print(num)
        if type(bssExp) is not dict:
            bssExpKeys = bssExp.fetch('KEY')
        else:
            print('Computed slow drift from spikes and assuming the spikes are first grouped by condition and *then* ordered by trial!')
            bssExpKeys = [dict(key=bssExp['subExp'].fetch('KEY'),spikes=bssExp['spikes'])]

        dimResult = dimResultsAll[num]
        numKey = 0
        latentVarExpBySlowDriftByCond, slowDriftVarExpTrialsByCond, slowDriftVarExpRAByCond, slowDriftVarExpSharedSpaceByCond, rollAvgPercOrigVar, rollAvgSDPercOrigVar = [],[],[],[],[],[]
        for bssKey, trlInds in zip(bssExpKeys, trlIndAllCv):
            if 'spikes' in bssKey:
                bssSpikes = bssKey['spikes']
                bssKey = bssKey['key'][0]
                bssExpHere = bssExp['subExp'][bssKey]
            else:
                bssSpikes = bssExp[bssKey].grabBinnedSpikes()[0]
                bssExpHere = bssExp[bssKey]
            dsParentExp = dsi[bssExpHere]
            trlFromDs = bssExpHere.trialAndChannelFilterFromParent(dsParentExp)[0]
            dsParent = dsParentExp.grabDataset()
            description = dsParentExp['dataset_name']
            # only use neurons GPFA uses
            *_, trlChKeep = bssSpikes.prepareGpfaOrFa(**preprocessInputsFinal)
            chUsedForFa = trlChKeep['channelsKeep']
            trlsUsedForFa = trlChKeep['trialsKeep']
            if chRestrictRefExp is not None:
                bssSpikesRef = chRestrictRefExp[num].grabBinnedSpikes()[0]
                *_, trlChKeep = bssSpikesRef.prepareGpfaOrFa(**preprocessInputsFinal)
                chFiltRefFromHere = chRestrictRefExp[num].trialAndChannelFilterFromParent(bssExpHere)[1]
                chUsedForFa = chFiltRefFromHere[trlChKeep['channelsKeep']]
            bssSpikes = bssSpikes[trlsUsedForFa][:,chUsedForFa]


            spikeBinStartInTrial = bssExpHere['start_time'][0]
            condLabels = bssSpikes.labels[labelUse]
            unLab = np.unique(condLabels, axis = 0)
            if unLab.shape[0]>1:
                # NOTE: this is for when conditions are combined; think
                # it'll be easy to do just haven't written the code yet
                trialsInBssWithLabel = np.hstack([(np.all(bssSpikes.labels[labelUse] == uL, axis=1)).nonzero()[0] for uL in unLab])
                # NOTE: you do NOT want to sort trialsInBssWithLabel. The
                # simplest way to give the reason is that before FA/GPFA was
                # run, when all the conditions were combined, the trials for
                # each condition were first grouped together. This means
                # that the trial index refers to the index of the trial not
                # ordered by when it was presented in the session, but
                # ordered first by the condition, and then *within* the
                # condition by when it was presented. By not sorting
                # trialsInBssWithLabel, I effectively replicate that
                # ordering here, so now testInds below is referring to the
                # same trials as indexed by trialsInBssWithLabel
                # trialsInBssWithLabel.sort()
                trialsForLabel = trlFromDs[trialsInBssWithLabel]
                trlTimes = dsParent.trialStartTimeInSession()[trialsForLabel] 
                strtTimesInTrial = spikeBinStartInTrial[trialsInBssWithLabel]
                spikeStartTimesInSession = trlTimes + strtTimesInTrial
            else:
                # note that eventually I might want to be flexible in using
                # another label... but for the moment this is hardcoded
                labelUse = 'stimulusMainLabel'
                trialsInBssWithLabel = np.all(spikesForGpfa.labels[labelUse] == unLab, axis=1)
                trialsForLabel = trlFromDs[trialsInBssWithLabel]
                trlTimes = dsParent.trialStartTimeInSession()[trialsForLabel] 
                strtTimesInTrial = spikeBinStartInTrial[trialsInBssWithLabel]
                spikeStartTimesInSession = trlTimes + strtTimesInTrial

            spikesUsedNorm = np.vstack([(br - br.mean(axis=(0,2))[:,None])/br.std(axis=(0,2))[:,None] for br in bssSpikes.groupByLabel(labels=bssSpikes.labels['stimulusMainLabel'])[0]])
            spikesUsedNorm = spikesUsedNorm.squeeze() # this tells me this wouldn't work with multi-timepoint trials...
            spikesUsedSortByTrials = spikesUsedNorm.copy()
            spikesUsedSortByTrials[trialsInBssWithLabel] = spikesUsedNorm
            slowDriftSpikes.append(spikesUsedNorm)
            # below wasn't doing appropriate residual-by-condition
            # bssSpikes, _ = bssSpikes.baselineSubtract(labels=bssSpikes.labels['stimulusMainLabel'])
            # spikesNorm = ((bssSpikes - bssSpikes.mean(axis=0))/bssSpikes.std(axis=0)).squeeze()
            # slowDriftSpikes.append(spikesNorm)
            # spikesUsedNorm = spikesNorm[trialsInBssWithLabel]

            if len(trlInds) > 1:
                breakpoint() # I think this means you've not combined conditions... unsure
            else:
                trlInds = trlInds[0].squeeze()

            minTime = 0 # ms
            maxTime = binSizeForAvg * np.ceil(spikeStartTimesInSession.max()/binSizeForAvg) # ms
            binnedSpikesOverSession, binEdges, binNum = sp.stats.binned_statistic(trlTimes, spikesUsedNorm.T, statistic='sum', bins=np.arange(minTime, maxTime+binSizeForAvg/2, binSizeForAvg))
            maskRecordedVals = np.zeros_like(binnedSpikesOverSession[0], dtype='bool')
            maskRecordedVals[binNum-1] = True # binNum is 1-indexed for these purposes...
            windowSizeForMeanMinutes = 20 # minutes
            minsToSecs = 60
            secsToMs = 1000
            windowSizeForMeanInBinSizes = np.round(windowSizeForMeanMinutes*minsToSecs*secsToMs/binSizeForAvg).astype(int)
            boxAverageFilter = np.ones(windowSizeForMeanInBinSizes, dtype=int)
            maskedVals = np.where(maskRecordedVals, binnedSpikesOverSession, 0)
            numValsExist = np.convolve(maskRecordedVals, boxAverageFilter, mode='valid')
            boxedAvgOut = [np.convolve(spkChan, boxAverageFilter, mode='valid')/numValsExist for spkChan in maskedVals]
            firstFullVal = boxAverageFilter.shape[0]
            # NOTE the binsToUse are the bins for which the filtered signal is
            # causal (i.e. filtering happened from spikes in the past)
            binsToUse = np.sort(binNum[binNum>firstFullVal])-firstFullVal
            binnedSpikeRunningAvgOverSession = np.stack(boxedAvgOut)[:, binsToUse]
            binnedSpikeRunningAvgOverSession = binnedSpikeRunningAvgOverSession.T
            spikesUsedMatchedToRunningAvg = spikesUsedNorm[np.argsort(binNum), :][np.sort(binNum)>firstFullVal, :]
            spikesUsedMinusRunningAvg = spikesUsedMatchedToRunningAvg - binnedSpikeRunningAvgOverSession
            pcs, evalsPca, _ = np.linalg.svd(np.cov(binnedSpikeRunningAvgOverSession.T,ddof=0))
            slowDriftRollingAverageSpikes.append(binnedSpikeRunningAvgOverSession)
            spikesWithSlowDriftSubtracted.append(spikesUsedMinusRunningAvg[:,:,None])
            slowDriftDim = 0
            pcSlowDrift = pcs[:, [slowDriftDim]]
            rollingAvgInSlowDriftDim = binnedSpikeRunningAvgOverSession @ pcSlowDrift @ pcSlowDrift.T
            percOrigVarOfRAInSD = 100*np.var(binnedSpikeRunningAvgOverSession @ pcSlowDrift) / np.cov(spikesUsedNorm.T, ddof=0).trace()
            percOrigVarOfRA = 100*np.cov(binnedSpikeRunningAvgOverSession.T, ddof=0).trace() / np.cov(spikesUsedNorm.T, ddof=0).trace()
            rollAvgPercOrigVar.append(percOrigVarOfRA)
            rollAvgSDPercOrigVar.append(percOrigVarOfRAInSD)
            allOnesDir = np.ones_like(pcSlowDrift)
            allOnesDir = allOnesDir/np.linalg.norm(allOnesDir)
            spikesUsedMinusSlowDriftOneDimRunAvg = spikesUsedMatchedToRunningAvg - rollingAvgInSlowDriftDim # + spikesUsedMatchedToRunningAvg @ allOnesDir @ allOnesDir.T
            spikesWithSlowDriftDimOneSubtracted.append(spikesUsedMinusSlowDriftOneDimRunAvg[:,:,None])
            # ** slow drift computations over **
            # breakpoint()
            # ** ATTEMPT AT FA
            # crossvalidateNum = 4
            # outputPathToConditions = ( dataPath / bssKey['bss_relative_path'] ).parent / 'slowDrift'
            # faScoreAll = []
            # for xDim in range(10):#np.arange(0,20, 5):
            #     faPrep = FA(binnedSpikeRunningAvgOverSession[:, :, None], crossvalidateNum=crossvalidateNum)
            #     # faPrep = FA(spikesUsedNorm[:, :, None], crossvalidateNum=crossvalidateNum)
            #     # faPrep = FA(((bssSpikes - bssSpikes.mean(axis=0))/bssSpikes.std(axis=0)), crossvalidateNum=crossvalidateNum)
            #     faScoreCond = np.empty((1,crossvalidateNum))
            #     fullOutputPath = outputPathToConditions / 'allCond'
            #     faScoreCond[0, :] = faPrep.runFa( numDim=xDim, gpfaResultsPath = fullOutputPath )[0]
            #     faScoreAll.append(faScoreCond)
            # finalAvgFaScore = np.stack(faScoreAll).squeeze().mean(axis=1)
            # ** END ATTEMPT

            slowDriftDims.append(pcSlowDrift)
            if plotOutput:
                title = plotOutput['title']
                plotTitle = description + ' ' + title
                dimResultHere = dimResult[numKey]
                if len(dimResultHere) > 1:
                    # I believe this is a case where conditions aren't combined,
                    # so there's one per condition; gotta think about how to
                    # make this work
                    breakpoint() 
                else:
                    dimResultHere = dimResultHere[0]
                binSize = bssSpikes.binSize
                latentVarExpBySlowDrift, slowDriftVarExpTrials, slowDriftVarExpRA, slowDriftVarExpSharedSpace = plotSlowDriftVsFaSpace(spikesUsedNorm, trlInds, trlTimes, binSize, pcSlowDrift, pcs, evalsPca, plotTitle, condLabels, dimResultHere)
                latentVarExpBySlowDriftByCond.append(latentVarExpBySlowDrift)
                slowDriftVarExpTrialsByCond.append(slowDriftVarExpTrials)
                slowDriftVarExpRAByCond.append(slowDriftVarExpRA)
                slowDriftVarExpSharedSpaceByCond.append(slowDriftVarExpSharedSpace)
               

            numKey += 1
        num+=1
        
        latentVarExpBySlowDriftByArea.append(latentVarExpBySlowDriftByCond)
        slowDriftVarExpTrialsByArea.append(slowDriftVarExpTrialsByCond)
        slowDriftVarExpRAByArea.append(slowDriftVarExpRAByCond)
        slowDriftVarExpSharedSpaceByArea.append(slowDriftVarExpSharedSpaceByCond)
        rollAvgPercOrigVarByArea.append(rollAvgPercOrigVar)
        rollAvgSDPercOrigVarByArea.append(rollAvgSDPercOrigVar)
    
    slowDriftDict = {
        '%ev of latent dimension by slow drift dim' : latentVarExpBySlowDriftByArea,
        '%ev of trials explained by slow drift dim' : slowDriftVarExpTrialsByArea,
        '%ev of rolling average by slow drift dim' : slowDriftVarExpRAByArea,
        '%ev of slow drift dim by shared space' : slowDriftVarExpSharedSpaceByArea,
        '%overall var of slow drift dim rolling avg' : rollAvgSDPercOrigVarByArea,
        '%overall var of rolling avg' : rollAvgPercOrigVarByArea,
    }
    return slowDriftDims, slowDriftDict, slowDriftSpikes, slowDriftRollingAverageSpikes, spikesWithSlowDriftSubtracted, spikesWithSlowDriftDimOneSubtracted

#%% Plotting and descriptive
def plotFiringRates(listBSS, descriptions, supTitle=None, cumulative = True):
    
    frFig = plt.figure()

    if type(listBSS) is not list:
        # we kind of assume that a single instance was passed here...
        listBSS = [listBSS]

    
    if cumulative:
        typeStr = " CDF"
    else:
        typeStr = " PDF"
        
    if supTitle is None:
        frFig.suptitle("Firing Rates" + typeStr)
    else:
        frFig.suptitle(supTitle + typeStr)
        
    ax = frFig.add_subplot(221)
    
    binWidth = 5
    for colInd, (bnSp, desc) in enumerate(zip(listBSS, descriptions)):
        if type(bnSp) is list:
            for bS in bnSp:
                units = bS.units
                if units == 'count':
                    histDat = bS.sumTrialCountByChannel()
                elif units == 'Hz':
                    histDat = bS.avgFiringRateByChannel()

                binWidthUse = np.min([binWidth, histDat.max()/10])
                bins = np.arange(0, histDat.max(), binWidthUse)
                ax.hist(histDat.view(np.ndarray), bins=bins, density=True, cumulative=cumulative, alpha = 0.8, label = desc, color = 'C%d' % colInd)
                desc = None # only label first shuffle
        else:
            units = bnSp.units
            if units == 'count':
                histDat = bnSp.sumTrialCountByChannel()
            elif units == 'Hz':
                histDat = bnSp.avgFiringRateByChannel()

            binWidthUse = np.min([binWidth, histDat.max()/10])
            bins = np.arange(0, histDat.max(), binWidthUse)
            ax.hist(histDat.view(np.ndarray), bins=bins, density=True, cumulative=cumulative, alpha = 0.8, label = desc, color = 'C%d' % colInd)
            units = bnSp.units

    if units == 'count':
        title = 'trial counts'
        xlabel = 'Spike Count'
    elif units == 'Hz':
        title = 'firing rate'
        xlabel = 'Firing Rate (Hz)'
        
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Probability')
    
    ax = frFig.add_subplot(223)
    binWidth = 1
    for colInd, (bnSp, desc) in enumerate(zip(listBSS, descriptions)):
        if type(bnSp) is list:
            for bS in bnSp:
                units = bS.units
                if units == 'count':
                    histDat = bS.stdSumTrialCountByChannel()
                elif units == 'Hz':
                    histDat = bS.timeAverage().trialStd().view(np.ndarray)

                binWidthUse = np.min([binWidth, histDat.max()/10])
                bins = np.arange(0, histDat.max(), binWidthUse)
                ax.hist(histDat, bins=bins, density=True, cumulative=cumulative, alpha = 0.8, label = desc, color = 'C%d' % colInd)
                desc = None # only label the first shuffle
            units = bS.units # replace every time, but should be the same for all...
        else:
            units = bnSp.units
            if units == 'count':
                histDat = bnSp.stdSumTrialCountByChannel()
            elif units == 'Hz':
                histDat = bnSp.timeAverage().trialStd()

            binWidthUse = np.min([binWidth, histDat.max()/10])
            bins = np.arange(0, histDat.max(), binWidthUse)
            ax.hist(histDat.view(np.ndarray), bins=bins, density=True, cumulative=cumulative, alpha = 0.8, label = desc, color = 'C%d' % colInd)
            units = bnSp.units # replace every time, but should be the same for all...
            
    if units == 'count':
        xlabel = 'Spike Count'
    elif units == 'Hz':
        xlabel = 'Firing Rate (Hz)'
       
    #ax.legend(loc="upper right")
    ax.set_title("by trial standard deviations")
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Probability')
    
    ax = frFig.add_subplot(122)
    for colInd, (bnSp, desc) in enumerate(zip(listBSS, descriptions)):
        if type(bnSp) is list:
            for bS in bnSp:
                units = bS.units
                if units == 'count':
                    scMn = bS.sumTrialCountByChannel()
                    scStd = bS.stdSumTrialCountByChannel()
                elif units == 'Hz':
                    scMn = bS.avgFiringRateByChannel()
                    scStd = bS.stdFiringRateByChannel()
                ax.scatter(scMn, scStd**2, label = desc, color = 'C%d' % colInd)
        else:
            units = bnSp.units # replace every time, but should be the same for all...
            if units == 'count':
                scMn = bnSp.sumTrialCountByChannel()
                scStd = bnSp.stdSumTrialCountByChannel()
            elif units == 'Hz':
                scMn = bnSp.avgFiringRateByChannel()
                scStd = bnSp.stdFiringRateByChannel()
            ax.scatter(scMn, scStd**2, label=desc, color = 'C%d' % colInd)
       
    maxX = np.max(ax.get_xlim())
    maxY = np.max(ax.get_ylim())
    maxDiagLine = np.min([maxX, maxY])
    ax.plot([0,maxDiagLine],[0,maxDiagLine], linestyle='--')
    # 4 is kind of mini... but they take up lots of space...
    ax.legend(loc="upper left", prop={'size':4})

    if units == 'count':
        xlabel = 'Spike Count'
        ylabel = 'Spike Count var'
    elif units == 'Hz':
        xlabel = 'Firing Rate (Hz)'
        ylabel = 'Firing Rate var'

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    frFig.tight_layout()
    
def plotExampleChannelResponses(groupedSpikesTrialAvg, groupedSpikesTrialAvgForMod, groupedSpikesEndTrialAvg, 
                                timeBeforeAndAfterStart, timeBeforeAndAfterEnd,
                                grpLabels, pltTitles, ylabel = 'Average Firing Rate (Hz)', chansForPlots = None):
    numCols = 3
    numRows = 3
    
    if chansForPlots is None:
        chansForPlots = np.repeat(None, len(groupedSpikesTrialAvg))
        
    for grpSpike, grpSpikeForMod, grpSpikeEnd, grpLabel, chan, plTtl in zip(groupedSpikesTrialAvg, groupedSpikesTrialAvgForMod, groupedSpikesEndTrialAvg, grpLabels, chansForPlots, pltTitles):
        
        if chan is None:
            chan = np.random.randint(grpSpike.shape[1])
            
        binSizeMs = grpSpike[0].binSize
        chanRespMean = np.squeeze(grpSpike[0][:,[chan], :])
        chanRespStd = np.squeeze(grpSpike[1][:,[chan],:])
        chanRespEndMean = np.squeeze(grpSpikeEnd[0][:, [chan], :])
        chanRespEndStd = np.squeeze(grpSpikeEnd[1][:, [chan], :])
    #    chanResp.timeAverage()
        chanTmAvg = grpSpikeForMod[0][:,[chan],:].timeAverage()
        chanTmStd = grpSpikeForMod[0][:,[chan],:].timeStd()
        
        
        plt.figure()
        plt.suptitle(plTtl + ': channel ' + str(chan))
        sbpltAngs = np.arange(3*np.pi/4, -5*np.pi/4, -np.pi/4)
        # Add nan to represent center for tuning polar curve...
        sbpltAngs = np.concatenate((sbpltAngs[0:3], sbpltAngs[[-1]], np.expand_dims(np.asarray(np.nan),axis=0), sbpltAngs[[3]], np.flip(sbpltAngs[4:-1])), axis=0)
        axVals = np.empty((0,4))
        axes = []
        for idx in range(0, len(grpLabel)):
            subplotChoose = np.where([np.allclose(grpLabel[idx] % (2*np.pi), sbpltAngs[i] % (2*np.pi)) for i in range(0,len(sbpltAngs))])[0]
            if not subplotChoose.size:
                subplotChoose = np.where(sbpltAngs==np.pi*(grpLabel[idx]-2))[0]
            subplotChooseDelayStart = subplotChoose[0]*2
            subplotChooseDelayEnd = subplotChoose[0]*2+1
            axes.append(plt.subplot(numRows, numCols*2, subplotChooseDelayStart+1))
            
            tmValStart = np.arange(timeBeforeAndAfterStart[0]+binSizeMs/2, timeBeforeAndAfterStart[1]+binSizeMs/2, binSizeMs)
            # Only plot what we have data for...
            startZeroBin = chanRespMean.alignmentBins[0]
            fstBin = 0
            tmBeforeStartZero = (fstBin-startZeroBin)*chanRespMean.binSize
            tmValStart = tmValStart[tmValStart>tmBeforeStartZero]
            
            plt.plot(tmValStart, chanRespMean[idx, :])
            plt.fill_between(tmValStart, chanRespMean[idx, :]-chanRespStd[idx,:], chanRespMean[idx, :]+chanRespStd[idx,:], alpha=0.2)
            axVals = np.append(axVals, np.array(plt.axis())[None, :], axis=0)
            axes.append(plt.subplot(numRows, numCols*2, subplotChooseDelayEnd+1))
            
            tmValEnd = np.arange(timeBeforeAndAfterEnd[0]+binSizeMs/2, timeBeforeAndAfterEnd[1]+binSizeMs/2, binSizeMs)
            # Only plot what we have data for...
            endZeroBin = chanRespEndMean.alignmentBins[1]
            lastBin = chanRespEndMean.shape[1]
            timeAfterEndZero = (lastBin-endZeroBin)*chanRespEndMean.binSize
            tmValEnd = tmValEnd[tmValEnd<timeAfterEndZero]
            
            plt.plot(tmValEnd, chanRespEndMean[idx, :])
            plt.fill_between(tmValEnd, chanRespEndMean[idx, :]-chanRespEndStd[idx,:], chanRespEndMean[idx, :]+chanRespEndStd[idx,:], alpha=0.2)
            axVals = np.append(axVals, np.array(plt.axis())[None, :], axis=0)
            if subplotChoose[0] == 0:
                axSt = axes[-2]
                axSt.set_ylabel(ylabel)
            elif subplotChoose[0] == (numCols*numRows - np.floor(numCols/2)):
                axSt = axes[-2]
                axSt.set_xlabel('Time Around Delay Start or End (ms)')
                
            if subplotChoose[0] == np.floor(numCols/2):
                axSt = axes[-2]
                axSt.set_title('Start')
                axEnd = axes[-1]
                axEnd.set_title('End')
            
            if subplotChoose[0] % numCols:
                axEn = axes[-1]
                axSt = axes[-2]
                axSt.yaxis.set_ticklabels('')
                axEn.yaxis.set_ticklabels('')
            else:
                axEn = axes[-1]
                axEn.yaxis.set_ticklabels('')
                
            if subplotChoose[0] < (numCols*numRows - numCols):
                axEn = axes[-1]
                axSt = axes[-2]
                axSt.xaxis.set_ticklabels('')
                axEn.xaxis.set_ticklabels('')
          
        ymin = np.min(np.append(0, np.min(axVals, axis=0)[2]))
        ymax = np.max(axVals, axis=0)[3]
        for ax in axes:
            ax.set_ylim(bottom = ymin, top = ymax )
            plt.axes(ax)
            plt.axvline(x=0, linestyle='--')
            
            
        plt.subplot(numRows, numCols, np.where(np.isnan(sbpltAngs))[0][0]+1, projection='polar')
        ptch = plt.fill(grpLabel, chanTmAvg)
        ptch[0].set_fill(False)
        ptch[0].set_edgecolor('k')
    #    plt.fill_between(grpLabel, chanTmAvg-chanTmStd, chanTmAvg+chanTmStd)

    
def plotStimDistributionHistograms(listBSS, plotTitles, labelUse='stimulusMainLabel'):
    for bnSp, plTtl in zip(listBSS, plotTitles):
        if type(bnSp) is list:
            plt.figure()
            plt.suptitle(plTtl)
            for bS in bnSp:
            #    _, trialsPresented = np.unique(dtst.markerTargAngles, axis=0, return_inverse=True)
                angsPres = bS.labels[labelUse].copy()
                # the bottom fits all the angles in the range -np.pi -> np.pi (angles remain
                # unchanged if they were in that range to begin with)
                angsPres = ((angsPres + np.pi) % (2*np.pi)) - np.pi
                plt.hist(angsPres,bins = np.arange(-9*np.pi/8, 7*np.pi/8, np.pi/4))
                plt.title('distribution of directions')
                plt.xlabel('direction angle (rad)')
                plt.ylabel('count')
        else:
            plt.figure()
            plt.suptitle(plTtl)
        #    _, trialsPresented = np.unique(dtst.markerTargAngles, axis=0, return_inverse=True)
            angsPres = bnSp.labels[labelUse].copy()
            # the bottom fits all the angles in the range -np.pi -> np.pi (angles remain
            # unchanged if they were in that range to begin with)
            angsPres = ((angsPres + np.pi) % (2*np.pi)) - np.pi
            plt.hist(angsPres,bins = np.arange(-9*np.pi/8, 7*np.pi/8, np.pi/4))
            plt.title('distribution of directions')
            plt.xlabel('direction angle (rad)')
            plt.ylabel('count')
