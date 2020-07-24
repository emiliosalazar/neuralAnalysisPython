#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 19:01:45 2020

@author: emilio

Here will be methods to act on lists of BinnedSpikeSets. Unclear whether these
are better set out as a class that contains lists of these... but methinks not
"""

import numpy as np
from classes.BinnedSpikeSet import BinnedSpikeSet
from matplotlib import pyplot as plt
from pathlib import Path
from methods.GeneralMethods import loadDefaultParams
# from decorators.ParallelProcessingDecorators import parallelize
from setup.DataJointSetup import BinnedSpikeSetProcessParams, BinnedSpikeSetInfo, DatasetInfo, GpfaAnalysisInfo, GpfaAnalysisParams, FilterSpikeSetParams
import hashlib
import json
import dill as pickle

from methods.GpfaMethods import crunchGpfaResults
from methods.plotUtils.GpfaPlotMethods import visualizeGpfaResults

def generateBinnedSpikeListsAroundState(data, keyStateName, trialType = 'successful', lenSmallestTrl=251, binSizeMs = 25, furthestTimeBeforeState=251, furthestTimeAfterState=251, setStartToStateEnd = False, setEndToStateStart = False, returnResiduals = False,  firingRateThresh = 1, fanoFactorThresh = 4, unitsOut = None):

#    if type(data) == DatasetInfo:
#    data = data.grabDatasets()

    defaultParams = loadDefaultParams(defParamBase = ".")
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

    trialFilterLambda = {'remove trials with catch state' : "lambda ds : ds.filterOutState('Catch')"}
    binnedSpikes, bssiKeys = data.computeBinnedSpikesAroundState(bSSProcParams, keyStateName, trialFilterLambda, units = unitsOut)

    
    return binnedSpikes, bssiKeys
#
#    if len(bsspp & bSSProcParams)>1:
#        raise Exception('multiple copies of binned spike set process params in here hm...')
#    elif len(bsspp & bSSProcParams)>0:
#        procParamId = (bsspp & bSSProcParams).fetch1('bss_params_id')
#    else:
#        # tacky, but it doesn't return the new id, syoo...
#        currIds = bsspp.fetch('bss_params_id')
#        bsspp.insert1(bSSProcParams)
#        newIds = bsspp.fetch('bss_params_id')
#        procParamId = list(set(newIds) - set(currIds))[0]
#
#    # return binned spikes
#    startDelaysList = []
#    endDelaysFromStartList = []
#    binnedSpikes = []
#    bssiKeys = []
#    chFanos = []
#    for dt, stateNameStateStart in zip(data, stateNamesStateStart):#dataset:#datasetSuccessNoCatch:
#        try:
#            dt = dt['dataset']
#        except TypeError:
#            pass
#        
#        dsId = dt.id
#
#        if trialType is 'successful':
#            dataStInit = dt.successfulTrials().filterOutCatch()
#        elif trialType is 'failure':
#            dataStInit = dt.failTrials().filterOutCatch()
#        else:
#            dataStInit = dt.filterOutCatch()
#        
#        alignmentStates = dt.metastates
#            
#        startDelay, endDelay, stateNameAfter = dataStInit.computeStateStartAndEnd(stateName = stateNameStateStart, ignoreStates=alignmentStates)
#        startDelayArr = np.asarray(startDelay)
#        endTimeArr = np.asarray(endDelay)
#        delayTimeArr = endTimeArr - startDelayArr
#        
#        
#       
#        dataSt = dataStInit.filterTrials(delayTimeArr>lenSmallestTrl)
#        # dataSt.computeCosTuningCurves()
#        startDelay, endDelay, stateNameAfter = dataSt.computeStateStartAndEnd(stateName = stateNameStateStart, ignoreStates=alignmentStates)
#        startDelayArr = np.asarray(startDelay)
#        startDelaysList.append(startDelayArr)
#        endDelayArr = np.asarray(endDelay)
#        endDelaysFromStartList.append(endDelayArr-startDelayArr)
#        
#        if setStartToDelayEnd:
#            startTimeArr = endDelayArr
#        else:
#            startTimeArr = startDelayArr
#        
#        if setEndToDelayStart:
#            endTimeArr = startDelayArr
#        else:
#            endTimeArr = endDelayArr
#
#        # here we want to check if we're going to load new data or if it already exists...
#        bssi = BinnedSpikeSetInfo()
#        dsi = DatasetInfo()
#
#        dsiPks = (dsi & ('dataset_id = %d' % dsId)).fetch('dataset_id', 'dataset_relative_path', 'ds_gen_params_id', as_dict=True)
#
#        if len(dsiPks) != 1:
#            raise Exception('There should be exactly one dataset record per binned spike set')
#        else:
#            dsiPks = dsiPks[0]
#        binnedSpikeSetDill = 'binnedSpikeSet.dill'
#        # a nice way to distinguish the path for each BSS based on extraction parameters...
#        bSSProcParamsJson = json.dumps(bSSProcParams, sort_keys=True) # needed for consistency as dicts aren't ordered
#        # encode('ascii') needed for json to be hashable...
#        bSSProcParamsHash = hashlib.md5(bSSProcParamsJson.encode('ascii')).hexdigest()
#        saveBSSRelativePath = Path(dsiPks['dataset_relative_path']).parent / ('binnedSpikeSet_%s' % bSSProcParamsHash[:5]) / binnedSpikeSetDill
#
#
#        saveBSSPath = dataPath / saveBSSRelativePath
#
#        if saveBSSPath.exists():
#            with saveBSSPath.open(mode='rb') as saveBSSFh:
#                binnedSpikesHere = pickle.load(saveBSSFh)
#        else:
#            # add binSizeMs/20 to endMs to allow for that timepoint to be included when using arange
#            binnedSpikesHere = dataSt.binSpikeData(startMs = list(startTimeArr-furthestTimeBeforeState), endMs = list(endTimeArr+furthestTimeAfterState+binSizeMs/20), binSizeMs=binSizeMs, alignmentPoints = list(zip(startTimeArr, endTimeArr)))
#            # first the firing rate thresh
#            binnedSpikesHere = binnedSpikesHere.channelsAboveThresholdFiringRate(firingRateThresh=firingRateThresh)[0]
#
#            # then we're doing fano factor, but for counts *over the trial*
#            if binnedSpikesHere.dtype == 'object':
#                trialLengthMs = binSizeMs*np.array([bnSpTrl[0].shape[0] for bnSpTrl in binnedSpikesHere])
#            else:
#                trialLengthMs = np.array([binnedSpikesHere.shape[2]*binSizeMs])
#            binnedCountsPerTrial = binnedSpikesHere.convertUnitsTo('count').increaseBinSize(trialLengthMs)
#
#            # NOTE: I believe this is okay because it's as if we were taking just a similarly sized window for each trial (under the assumption of uniform firing), but the std and mean is taken *afterwards*
#            if trialLengthMs.size > 1:
#                binnedCountsPerTrial = binnedCountsPerTrial * trialLengthMs[:,None,None]/trialLengthMs.max()
#            _, chansGood = binnedCountsPerTrial.channelsBelowThresholdFanoFactor(fanoFactorThresh=fanoFactorThresh)
#            chFano = binnedCountsPerTrial[:,chansGood].fanoFactorByChannel()
#            chFanos.append(chFano)
#            binnedSpikesHere = binnedSpikesHere[:,chansGood]
#
#            
#
#
#            try:
#                binnedSpikesHere.labels['stimulusMainLabel'] = dataSt.markerTargAngles
#            except AttributeError:
#                for bnSp, lab in zip(binnedSpikesHere, dataSt.markerTargAngles):
#                    bnSp.labels['stimulusMainLabel'] = lab
#                
#            if returnResiduals:
##                if binnedSpikesHere.dtype == 'object':
##                    raise Exception("Residuals can only be computed if all trials are the same length!")
##                else:
#                labels = binnedSpikesHere.labels['stimulusMainLabel']
#                binnedSpikesHere, labelBaseline = binnedSpikesHere.baselineSubtract(labels=labels)
#                
#
#            if not saveBSSPath.exists():
#                saveBSSPath.parent.mkdir(parents=True, exist_ok = True)
#                with saveBSSPath.open(mode='wb') as saveBSSFh:
#                    pickle.dump(binnedSpikesHere, saveBSSFh)
#            else:
#                raise Exception('Uh oh our BSS save paths are on a collision course!')
#            
#        binnedSpikeSetHereInfo = dict(
#            bss_params_id = procParamId,
#            bss_relative_path = str(saveBSSRelativePath),
#            start_alignment_state = stateNameStateStart,
#            end_alignment_state = stateNameStateStart, # in this function it's always from the start, either the end or the beginning of the start, but from the start
#        )
#
#        bssiKeys.append(binnedSpikeSetHereInfo.copy())
#        if len(bssi & binnedSpikeSetHereInfo) > 1:
#            raise Exception("we've saved this processing more than once...")
#        elif len(bssi & binnedSpikeSetHereInfo) == 0:
#            bssHash = hashlib.md5(str(binnedSpikesHere).encode('ascii')).hexdigest()
#
#
#            addlBssInfo = dict(
#                bss_hash = bssHash,
#                start_time_alignment = np.array(startTimeArr),
#                start_time = np.array(startTimeArr + furthestTimeBeforeState),
#                end_time_alignment = np.array(endTimeArr),
#                end_time = np.array(endTimeArr + furthestTimeAfterState)
#            )
#            binnedSpikeSetHereInfo.update(addlBssInfo)
#                
#            binnedSpikeSetHereInfo.update(dsiPks)
#
#
#            bssi.insert1(binnedSpikeSetHereInfo)
#            
#        if unitsOut is not None:
#            binnedSpikesHere.convertUnitsTo(units=unitsOut)
#        
#        binnedSpikes.append(binnedSpikesHere)
#    
#    return binnedSpikes, bssiKeys


# This function lets you match the number of neurons and trials for different
# sets of data you're comparing, to be fairer in the comparison. Note that what
# is referred to as 'neuron' in the method name could well be a thresholded
# channel with multineuron activity... but this nomencalture reminds us what
# the purpose is...
def subsampleBinnedSpikeSetsToMatchNeuronsAndTrialsPerCondition(bssExp, maxNumTrlPerCond, maxNumNeuron, labelName, numSubsamples = 1,  extraOpts = None, order=True):
    # I do, unfortunately, have to load them twice here--once up here to
    # compute the neuron/trial nums, and again below to make sure they're well
    # aligned

    dsi = DatasetInfo()

    if order:
        bssKeys = bssExp.fetch('KEY', order_by='dataset_id')
    else:
        bssKeys = bssExp.fetch('KEY')

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
    for bssKeyForOne in bssKeys:
        # reset the seed for every binned spike so we know that whenever a set
        # of data comes through here it gets split up identically... well...
        # assuming the same minima up there...
        bnSpOrigInfo = bssExp[bssKeyForOne]
        if extraOpts:
            extraFilt = []
            extraDescription = []
            anyMatch = False
            for extraOptExp, filterParams in extraOpts.items():
                match = bssExp[extraOptExp][bssKeyForOne]
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


        bnSpHereSubsample, trlNeurHereSubsamples, datasetName, subsampleKeyHere = bnSpOrigInfo.computeRandomSubsets("match for comparisons", numTrialsPerCond = maxNumTrlPerCond, numChannels = maxNumNeuron, labelName = labelName, numSubsets = numSubsamples, returnInfo = True)

        brainArea = dsi[bnSpOrigInfo].fetch('brain_area')[0] # returns array, grab the value (string)
        task = dsi[bnSpOrigInfo].fetch('task')[0]
#        bnSpOrig = bnSpOrigInfo.grabBinnedSpikes()
#        assert len(bnSpOrig) == 1, "Should only have one BSS to filter each time!"
#        bnSpOrig = bnSpOrig[0]
#        np.random.seed(0)
#        trlInds = range(bnSpOrig.shape[0])
#        neuronInds = range(bnSpOrig.shape[1])
#        bnSpHereSubsample = []
#        trlNeurHereSubsamples = []
#        for newSubset in range(numSubsamples):
#            # note that the trials are randomly chosen, but returned in sorted
#            # order (balancedTriaInds does this) from first to last... I think
#            # this'll make things easier to compare in downstream analyses that
#            # care about changes over time... neurons being sorted on the other
#            # hand... still for ease of comparison, maybe not for any specific
#            # analysis I can think of
#            trlsUse = bnSpOrig.balancedTrialInds(bnSpOrig.labels[labelName], minCnt = maxNumTrlPerCond)
#            neuronsUse = np.sort(np.random.permutation(neuronInds)[:maxNumNeur])
#
#            bnSpHereSubsample.append(bnSpOrigInfo.filterBSS("shuffle", "match for area comparisons", binnedSpikeSet=bnSpOrig, trialFilter = nonChoiceTrials, channelFilter = channelsUse, returnExisting=True))
#            trlNeurHereSubsamples.append((trlsUse, neuronsUse))


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
    groupedSpikesTrialAvg = []
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
            
            

        uniqueTargAngle = np.unique(binnedSpikes.labels[labelUse], axis=0)
    
        groupedSpikes, uniqueLabel = binnedSpikes.groupByLabel(binnedSpikes.labels[labelUse])
        
        
        # groupedSpikesEnd = dataSt.groupSpikes(trialsPresented, uniqueTargAngle, binnedSpikes = binnedSpikesHereEnd)
        # groupedSpikesShortStart = dataSt.groupSpikes(trialsPresented, uniqueTargAngle, binnedSpikes = binnedSpikesHereShortStart)
        alignmentBinsAll = binnedSpikes.alignmentBins[0] if binnedSpikes.alignmentBins is not None else None
        stAll = binnedSpikes.start
        endAll = binnedSpikes.end 
        targAvgList, targSemList = zip(*[(groupedSpikes[targ].trialAverage(), groupedSpikes[targ].trialSem()) for targ in range(0, len(groupedSpikes))])
#        breakpoint()
        targTmTrcAvgArr = BinnedSpikeSet(np.stack(targAvgList), start=stAll, end = endAll, binSize = binnedSpikes.binSize, labels = {labelUse: uniqueLabel}, alignmentBins = alignmentBinsAll)
        targTmTrcSemArr = BinnedSpikeSet(np.stack(targSemList), start=stAll, end = endAll, binSize = binnedSpikes.binSize, labels = {labelUse: uniqueLabel}, alignmentBins = alignmentBinsAll)
        # targAvgListEnd, targStdListEnd = zip(*[(groupedSpikesEnd[targ].trialAverage(), groupedSpikesEnd[targ].trialStd()) for targ in range(0, len(groupedSpikesEnd))])
        # targTmTrcAvgEndArr = np.stack(targAvgListEnd).view(BinnedSpikeSet)
        # targTmTrcStdEndArr = np.stack(targStdListEnd).view(BinnedSpikeSet)
        
        # targAvgShrtStartList, targStdShrtStartList = zip(*[(groupedSpikesShortStart[targ].trialAverage(), groupedSpikesShortStart[targ].trialStd()) for targ in range(0, len(groupedSpikesShortStart))])
        # targTmTrcShrtStartAvgArr = np.stack(targAvgShrtStartList).view(BinnedSpikeSet)
        # targTmTrcShrtStartStdArr = np.stack(targStdShrtStartList).view(BinnedSpikeSet)
        
        # groupedSpikesTrialShortStartAvg.append([targTmTrcShrtStartAvgArr, targTmTrcShrtStartStdArr])
        groupedSpikesAll.append(groupedSpikes)
        groupedSpikesTrialAvg.append([targTmTrcAvgArr, targTmTrcSemArr])
        # groupedSpikesEndTrialAvg.append([targTmTrcAvgEndArr, targTmTrcStdEndArr])
        grpLabels.append(uniqueTargAngle)
    
    return groupedSpikesTrialAvg, groupedSpikesAll, grpLabels
    

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
def gpfaComputation(bssExp, timeBeforeAndAfterStart = None, timeBeforeAndAfterEnd = None, balanceConds = True, computeResiduals = True, numStimulusConditions = 1,combineConditions = False, sqrtSpikes = False, forceNewGpfaRun=False,
                    crossvalidateNumFolds = 4, xDimTest = [2,5,8], overallFiringRateThresh=0.5, perConditionGroupFiringRateThresh = 0.5, signalDescriptor = "",plotOutput=True, units='count', useFa = False):
    
    # from classes.GPFA import GPFA
    from mpl_toolkits.mplot3d import Axes3D # for 3d plotting
    from methods.GeneralMethods import prepareMatlab
    from multiprocessing import pool
    

    gai = GpfaAnalysisInfo()
    gap = GpfaAnalysisParams()

    labelUse = 'stimulusMainLabel'
    for idxXdim,dim in enumerate(xDimTest):
        print("Testing/loading dimensionality %d. Left to test: " % dim + (str(xDimTest[idxXdim+1:]) if idxXdim+1<len(xDimTest) else "none"))
        gpfaParams = dict(
            dimensionality = dim,
            overall_fr_thresh = overallFiringRateThresh,
            balance_conds = balanceConds,
            sqrt_spikes = sqrtSpikes,
            num_conds = 1 if not combineConditions else (0 if numStimulusConditions is None else numStimulusConditions),
            combine_conditions = 'no' if not combineConditions else ('all' if numStimulusConditions is None else 'subset'),
            num_folds_crossvalidate = crossvalidateNumFolds,
            on_residuals = computeResiduals,
            units = units
        )

        if len(gap[gpfaParams]) == 0:
            gap.insert1(gpfaParams)

        if type(bssExp) is list:
            for subExp in bssExp:
                if not useFa:
                    bssExpComputed = subExp[gai[gap[gpfaParams]]]
                    bssExpToCompute = subExp - gai[gap[gpfaParams]]
                
                    # note that this *adds* values to GpfaAnalysisInfo, so we can't
                    # just filter gai by bssExpToCompute (nothing will be there!)
                    gai.computeGpfaResults(gap[gpfaParams], bssExpToCompute, labelUse=labelUse, conditionNumbersGpfa = numStimulusConditions, perCondGroupFiringRateThresh = perConditionGroupFiringRateThresh, useFa=useFa)

        else:
            if not useFa:
                bssExpComputed = bssExp[gai[gap[gpfaParams]]]
                bssExpToCompute = bssExp - gai[gap[gpfaParams]]
#            # NOTE CHANGE
#            bssExpToCompute = bssExpComputed
#            forceNewGpfaRun = True
            
            # note that this *adds* values to GpfaAnalysisInfo, so we can't
            # just filter gai by bssExpToCompute (nothing will be there!)
                gai.computeGpfaResults(gap[gpfaParams], bssExpToCompute, labelUse=labelUse, conditionNumbersGpfa = numStimulusConditions, perCondGroupFiringRateThresh = perConditionGroupFiringRateThresh, forceNewGpfaRun = forceNewGpfaRun, useFa=useFa)


    # this is gonna filter the GpfaAnalysisParams (gap) which will be used to
    # filter the infos
    gpfaParamsAll = [{'dimensionality' : d} for d in xDimTest]
    ind=0
    if type(bssExp) is list:
        # Um.... DataJoint can apparently handle bssExp being a list of expressions
        # and that's nuts. Unfortunately for the nuttiness, I *do* need to be able
        # to keep these grouped together.
        gpfaRes = []
        gpfaInfo = []
        lstSzs = []
        for subExp in bssExp:
            ind+=1
            lstSzs.append(len(gai[subExp]))
            gpfaResH, gpfaInfoH = gai[subExp][gap[gpfaParamsAll]].grabGpfaResults(returnInfo=True, useFa=useFa)

            gpfaRes.append(gpfaResH)
            gpfaInfo.append(gpfaInfoH)
    else:
        gpfaRes, gpfaInfo = gai[bssExp][gap[gpfaParamsAll]].grabGpfaResults(returnInfo=True, order=True, useFa=useFa)

# could probably make this work... but better to just keep the above, no?
#    gpfaRes2, gpfaInfo2 = gai[bssExp].grabGpfaResults(returnInfo=True, order=True)


    if type(gpfaRes) is not list:
        gpfaRes = [gpfaRes] # for the for loop below
        gpfaInfo = [gpfaInfo]

#    breakpoint()
    cvApproach = "logLikelihood"
    shCovThresh = 0.95
    lblLLErr = 'LL err over folds'
    lblLL= 'LL mean over folds'

    if plotOutput:
        plotInfo = {}
        figErr = plt.figure()
        figErr.suptitle('dimensionality vs. GPFA log likelihood')
        # might have to go back on these lines for how to define ncols...
        gpPathPer = [[Path(gpH).parent for gpH in gp.keys()] for gp in gpfaRes]
        gpUnPathNum = [len(np.unique(gpP)) for gpP in gpPathPer]
        ncols = np.max(gpUnPathNum) # len(bssExp)
        axs = figErr.subplots(nrows=2, ncols=ncols, squeeze=False)

    crossValsUse = [0]

    dimExp = []
    dimMoreLL = []
    normScoreAll = []
    gpfaOutDimAll = []
    gpfaTestIndsOutAll = []
    gpfaTrainIndsOutAll = []
    gpfaBinSize = []
    gpfaCondLabels = []
    gpfaAlignmentBins = []
    gpfaDimsTested = []

    for gpfaResHere, gpfaInfoHere in zip(gpfaRes, gpfaInfo):
        gpfaCrunchedResults = crunchGpfaResults(gpfaResHere, cvApproach = cvApproach, shCovThresh = shCovThresh)

        keysRes = [Path(pthAndCond).parent for pthAndCond in gpfaCrunchedResults.keys()]
        _, smDimGrph = np.unique(keysRes, return_inverse=True)
        dimExp.append([d['xDimBestAll'] for _, d in gpfaCrunchedResults.items()])
        dimMoreLL.append([d['xDimScoreBestAll'] for _, d in gpfaCrunchedResults.items()])
        normScoreAll.append([d['normalGpfaScoreAll'] for _, d in gpfaCrunchedResults.items()])
        gpfaOutDimAll.append([d['dimResults'] for _, d in gpfaCrunchedResults.items()])
        gpfaTestIndsOutAll.append([d['testInds'] for _, d in gpfaCrunchedResults.items()])
        gpfaTrainIndsOutAll.append([d['trainInds'] for _, d in gpfaCrunchedResults.items()])
        gpfaBinSize.append([d['binSize'] for _, d in gpfaCrunchedResults.items()])
        gpfaCondLabels.append([d['condLabel'] for _, d in gpfaCrunchedResults.items()])
        gpfaAlignmentBins.append([d['alignmentBins'] for _, d in gpfaCrunchedResults.items()])
        gpfaDimsTested.append([d['dimsTest'] for _, d in gpfaCrunchedResults.items()])


        if plotOutput:
            for idx, (description, dimsTest, testIndsCondAll, dimResult, normalGpfaScoreAll, binSize, condLabels, alignmentBins) in enumerate(zip(gpfaInfoHere['datasetNames'], gpfaDimsTested[-1], gpfaTestIndsOutAll[-1], gpfaOutDimAll[-1], normScoreAll[-1], gpfaBinSize[-1], gpfaCondLabels[-1], gpfaAlignmentBins[-1])):

                # grab first cval--they'll be the same for each cval, which is what
                # these lists store
                dimTest = dimsTest[0]
                binSize = binSize[0]
                testInds = testIndsCondAll[0]

                axScore = axs[0, :].flat[smDimGrph[idx]]
                axDim = axs[1,:].flat[smDimGrph[idx]]
                plotInfo['axScore'] = axScore
                plotInfo['axDim'] = axDim

                if timeBeforeAndAfterStart is not None:
                    tmValsStartBest = np.arange(timeBeforeAndAfterStart[0], timeBeforeAndAfterStart[1], binSize)
                else:
                    tmValsStartBest = np.ndarray((0,0))
                    
                if timeBeforeAndAfterEnd is not None:
                    tmValsEndBest = np.arange(timeBeforeAndAfterEnd[0], timeBeforeAndAfterEnd[1], binSize)  
                else:
                    tmValsEndBest = np.ndarray((0,0))

                plotInfo['lblLL'] = lblLL
                plotInfo['lblLLErr'] = lblLLErr
                plotInfo['description'] = description
                tmVals = [tmValsStartBest, tmValsEndBest]
                visualizeGpfaResults(plotInfo, dimResult,  tmVals, cvApproach, normalGpfaScoreAll, dimTest, testInds, shCovThresh, crossValsUse, binSize, condLabels, alignmentBins)

    return dimExp, dimMoreLL, gpfaOutDimAll, gpfaTestIndsOutAll, gpfaTrainIndsOutAll

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
    fRChStdByArea = []
    fanoFactorChMnByArea = []
    fanoFactorChStdByArea = []
    fanoFactorChOBTMnByArea = []
    fanoFactorChOBTStdByArea = []
    labelUse = 'stimulusMainLabel'
    for bnSpCnt in listBSS:
        fRChMnByAreaHere = []
        fRChStdByAreaHere = []
        fanoFactorChMnByAreaHere = []
        fanoFactorChStdByAreaHere = []
        fanoFactorChOBTMnByAreaHere = []
        fanoFactorChOBTStdByAreaHere = []
        for bSC in bnSpCnt:
            # All these metrics are taken on spike counts!
            bSC = bSC.convertUnitsTo('count')

            grpSpkCnt, uniqueLabel = bSC.groupByLabel(bSC.labels[labelUse])
            fRChMnByAreaHere.append([np.mean(gSC.avgFiringRateByChannel()) for gSC in grpSpkCnt])
            fRChStdByAreaHere.append([np.std(gSC.avgFiringRateByChannel()) for gSC in grpSpkCnt])
            fanoFactorChMnByAreaHere.append([np.mean(gSC.fanoFactorByChannel()) for gSC in grpSpkCnt])
            fanoFactorChStdByAreaHere.append([np.std(gSC.fanoFactorByChannel()) for gSC in grpSpkCnt])

            if bSC.dtype == 'object':
                grpTrlLenMs = [gSC.binSize*np.array([gSC[0].shape[0] for gSC in gpSpCn]) for gpSpCn in grpSpkCnt]
            else:
                grpTrlLenMs = [np.array([gSC.shape[2]*gSC.binSize]) for gSC in grpSpkCnt]
            
            fanoFactorChOBTMnByAreaHere.append([np.mean(gSC.increaseBinSize(gTLms).fanoFactorByChannel()) for gSC, gTLms in zip(grpSpkCnt, grpTrlLenMs)])
            fanoFactorChOBTStdByAreaHere.append([np.std(gSC.increaseBinSize(gTLms).fanoFactorByChannel()) for gSC, gTLms in zip(grpSpkCnt, grpTrlLenMs)])

        fRChMnByArea.append(np.stack(fRChMnByAreaHere))
        fRChStdByArea.append(np.stack(fRChStdByAreaHere))
        fanoFactorChMnByArea.append(np.stack(fanoFactorChMnByAreaHere))
        fanoFactorChStdByArea.append(np.stack(fanoFactorChStdByAreaHere))
        fanoFactorChOBTMnByArea.append(np.stack(fanoFactorChOBTMnByAreaHere))
        fanoFactorChOBTStdByArea.append(np.stack(fanoFactorChOBTStdByAreaHere))
    
    resultsDict = {
        'mean channel firing rate (Hz)' : fRChMnByArea,
        'std channel firing rate (Hz)' : fRChStdByArea,
        'mean(r_{sc})' : mnCorrPerCond,
        'std(r_{sc})' : stdCorrPerCond,
        'mean(r_{sc} 1 bn/tr)' : mnCorrPerCondOBPT,
        'std(r_{sc} 1 bn/tr)' : stdCorrPerCondOBPT,
        'mean channel fano factor' : fanoFactorChMnByArea,
        'std channel fano factor' : fanoFactorChStdByArea,
        'mean channel fano factor (1 bn/tr)' : fanoFactorChOBTMnByArea,
        'std channel fano factor (1 bn/tr)' : fanoFactorChOBTStdByArea
    }

    return resultsDict


#%% Plotting and descriptive
def plotFiringRates(listBSS, descriptions, supTitle=None, cumulative = True):
    
    frFig = plt.figure()

    
    if cumulative:
        typeStr = " CDF"
    else:
        typeStr = " PDF"
        
    if supTitle is None:
        frFig.suptitle("Firing Rates" + typeStr)
    else:
        frFig.suptitle(supTitle + typeStr)
        
    ax = frFig.add_subplot(221)
    
    for colInd, (bnSp, desc) in enumerate(zip(listBSS, descriptions)):
        if type(bnSp) is list:
            for bS in bnSp:
                units = bS.units
                if units == 'count':
                    histDat = bS.sumTrialCountByChannel()
                elif units == 'Hz':
                    histDat = bS.avgFiringRateByChannel()
                ax.hist(histDat.view(np.ndarray), density=True, cumulative=cumulative, alpha = 0.8, label = desc, color = 'C%d' % colInd)
                desc = None # only label first shuffle
        else:
            units = bnSp.units
            if units == 'count':
                histDat = bnSp.sumTrialCountByChannel()
            elif units == 'Hz':
                histDat = bnSp.avgFiringRateByChannel()
            ax.hist(histDat.view(np.ndarray), density=True, cumulative=cumulative, alpha = 0.8, label = desc, color = 'C%d' % colInd)
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
    for colInd, (bnSp, desc) in enumerate(zip(listBSS, descriptions)):
        if type(bnSp) is list:
            for bS in bnSp:
                ax.hist(bS.timeAverage().trialStd().view(np.ndarray), density=True, cumulative=cumulative, alpha = 0.8, label = desc, color = 'C%d' % colInd)
                desc = None # only label the first shuffle
            units = bS.units # replace every time, but should be the same for all...
        else:
            ax.hist(bnSp.timeAverage().trialStd().view(np.ndarray), density=True, cumulative=cumulative, alpha = 0.8, label = desc, color = 'C%d' % colInd)
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
