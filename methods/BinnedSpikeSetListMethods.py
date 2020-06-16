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
# from decorators.ParallelProcessingDecorators import parallelize

def generateBinnedSpikeListsAroundDelay(data, dataIndsProcess, stateNamesDelayStart, trialType = 'successful', lenSmallestTrl=251, binSizeMs = 25, furthestTimeBeforeDelay=251, furthestTimeAfterDelay=251, setStartToDelayEnd = False, setEndToDelayStart = False, returnResiduals = False, removeBadChannels = False, firingRateThresh = 1, fanoFactorThresh = 4, unitsOut = None):
    # return binned spikes
    startDelaysList = []
    endDelaysFromStartList = []
    binnedSpikes = []
    chFanos = []
    for ind, stateNameDelayStart in zip(dataIndsProcess, stateNamesDelayStart):#dataset:#datasetSuccessNoCatch:
        
        if trialType is 'successful':
            dataStInit = data[ind]['dataset'].successfulTrials().trialsWithoutCatch()
        else:
            dataStInit = data[ind]['dataset'].failTrials().trialsWithoutCatch()
        
        alignmentStates = data[ind]['alignmentStates']
            
        startDelay, endDelay = dataStInit.computeDelayStartAndEnd(stateNameDelayStart = stateNameDelayStart, ignoreStates=alignmentStates)
        startDelayArr = np.asarray(startDelay)
        endTimeArr = np.asarray(endDelay)
        delayTimeArr = endTimeArr - startDelayArr
        
        
       
        dataSt = dataStInit.filterTrials(delayTimeArr>lenSmallestTrl)
        # dataSt.computeCosTuningCurves()
        startDelay, endDelay = dataSt.computeDelayStartAndEnd(stateNameDelayStart = stateNameDelayStart, ignoreStates=alignmentStates)
        startDelayArr = np.asarray(startDelay)
        startDelaysList.append(startDelayArr)
        endDelayArr = np.asarray(endDelay)
        endDelaysFromStartList.append(endDelayArr-startDelayArr)
        
        if setStartToDelayEnd:
            startTimeArr = endDelayArr
        else:
            startTimeArr = startDelayArr
        
        if setEndToDelayStart:
            endTimeArr = startDelayArr
        else:
            endTimeArr = endDelayArr
        
        # add binSizeMs/20 to endMs to allow for that timepoint to be included when using arange
        binnedSpikesHere = dataSt.binSpikeData(startMs = list(startTimeArr-furthestTimeBeforeDelay), endMs = list(endTimeArr+furthestTimeAfterDelay+binSizeMs/20), binSizeMs=binSizeMs, notChan=[31, 0], alignmentPoints = list(zip(startTimeArr, endTimeArr)))
        # first the firing rate thresh
        binnedSpikesHere = binnedSpikesHere.channelsAboveThresholdFiringRate(firingRateThresh=firingRateThresh)[0]

        # then we're doing fano factor, but for counts *over the trial*
        trialLengthMs = furthestTimeBeforeDelay + furthestTimeAfterDelay - 1
        binnedCountsPerTrial = binnedSpikesHere.convertUnitsTo('count').increaseBinSize(trialLengthMs)
        _, chansGood = binnedCountsPerTrial.channelsBelowThresholdFanoFactor(fanoFactorThresh=fanoFactorThresh)
        chFano = binnedCountsPerTrial[:,chansGood].fanoFactorByChannel()
        chFanos.append(chFano)
        binnedSpikesHere = binnedSpikesHere[:,chansGood]

        

        if unitsOut is not None:
            binnedSpikesHere.convertUnitsTo(units=unitsOut)

        try:
            binnedSpikesHere.labels['stimulusMainLabel'] = dataSt.markerTargAngles
        except AttributeError:
            for bnSp, lab in zip(binnedSpikesHere, dataSt.markerTargAngles):
                bnSp.labels['stimulusMainLabel'] = lab
            
        if returnResiduals:
            labels = binnedSpikesHere.labels['stimulusMainLabel']
            binnedSpikesHere, labelBaseline = binnedSpikesHere.baselineSubtract(labels=labels)

        # This should also work if you don't returnResiduals, but I haven't thought through if the filtering would work appropriately, so I'll let it error for now if removeBadChannels=True but returnResiduals is not
#        if removeBadChannels:
#            unq, unqInv = np.unique(labels, return_inverse=True, axis=0)
#            binnedSpikesHere, goodChans = binnedSpikesHere.channelsNotRespondingSparsely(zeroRate = np.array(labelBaseline)[unqInv])
#            labelBaseline = [lb[goodChans] for lb in labelBaseline]
#            binnedSpikesHere, goodChans = binnedSpikesHere.removeInconsistentChannelsOverTrials(zeroRate = np.array(labelBaseline)[unqInv])
#            labelBaseline = [lb[goodChans] for lb in labelBaseline]

        
        binnedSpikes.append(binnedSpikesHere)
    
    return binnedSpikes, chFanos


# This function lets you match the number of neurons and trials for different
# sets of data you're comparing, to be fairer in the comparison. Note that what
# is referred to as 'neuron' in the method name could well be a thresholded
# channel with multineuron activity... but this nomencalture reminds us what
# the purpose is...
def subsampleBinnedSpikeSetsToMatchNeuronsAndTrialsPerCondition(binnedSpikesList, numSubsamples = 1, labelName='stimulusMainLabel'):
    minNumNeur = np.min([bnSp.shape[1] for bnSp in binnedSpikesList])
    numTrls = [np.unique(bnSp.labels[labelName],axis=0,return_counts=True)[1] for bnSp in binnedSpikesList]
    minNumTrlPerCond = np.min(np.hstack(numTrls))


    initRandSt = np.random.get_state()
    bnSpSubsamples = []
    trlNeurSubsamples = []
    for bnSpOrig in binnedSpikesList:
        # reset the seed for every binned spike so we know that whenever a set
        # of data comes through here it gets split up identically... well...
        # assuming the same minima up there...
        np.random.seed(0)
        trlInds = range(bnSpOrig.shape[0])
        neuronInds = range(bnSpOrig.shape[1])
        bnSpHereSubsample = []
        trlNeurHereSubsamples = []
        for newSubset in range(numSubsamples):
            # note that the trials are randomly chosen, but returned in sorted
            # order (balancedTriaInds does this) from first to last... I think
            # this'll make things easier to compare in downstream analyses that
            # care about changes over time... neurons being sorted on the other
            # hand... still for ease of comparison, maybe not for any specific
            # analysis I can think of
            trlsUse = bnSpOrig.balancedTrialInds(bnSpOrig.labels[labelName], minCnt = minNumTrlPerCond)
            neuronsUse = np.sort(np.random.permutation(neuronInds)[:minNumNeur])
            bnSpHereSubsample.append(bnSpOrig[trlsUse][:,neuronsUse])
            trlNeurHereSubsamples.append((trlsUse, neuronsUse))

        bnSpSubsamples.append(bnSpHereSubsample)
        trlNeurSubsamples.append(trlNeurHereSubsamples)






    np.random.set_state(initRandSt)

    return bnSpSubsamples, trlNeurSubsamples, minNumTrlPerCond, minNumNeur
    

#%% Analyses on binned spike sets
def generateLabelGroupStatistics(binnedSpikesListToGroup, labelUse = 'stimulusMainLabel'):
    
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
        alignmentBinsAll = binnedSpikes.alignmentBins[0]
        targAvgList, targStdList = zip(*[(groupedSpikes[targ].trialAverage(), groupedSpikes[targ].trialStd()) for targ in range(0, len(groupedSpikes))])
        targTmTrcAvgArr = BinnedSpikeSet(np.stack(targAvgList), start=binnedSpikes.start, end = binnedSpikes.end, binSize = binnedSpikes.binSize, labels = {labelUse: uniqueLabel}, alignmentBins = alignmentBinsAll)
        targTmTrcStdArr = BinnedSpikeSet(np.stack(targStdList), start=binnedSpikes.start, end = binnedSpikes.end, binSize = binnedSpikes.binSize, labels = {labelUse: uniqueLabel}, alignmentBins = alignmentBinsAll)
        # targAvgListEnd, targStdListEnd = zip(*[(groupedSpikesEnd[targ].trialAverage(), groupedSpikesEnd[targ].trialStd()) for targ in range(0, len(groupedSpikesEnd))])
        # targTmTrcAvgEndArr = np.stack(targAvgListEnd).view(BinnedSpikeSet)
        # targTmTrcStdEndArr = np.stack(targStdListEnd).view(BinnedSpikeSet)
        
        # targAvgShrtStartList, targStdShrtStartList = zip(*[(groupedSpikesShortStart[targ].trialAverage(), groupedSpikesShortStart[targ].trialStd()) for targ in range(0, len(groupedSpikesShortStart))])
        # targTmTrcShrtStartAvgArr = np.stack(targAvgShrtStartList).view(BinnedSpikeSet)
        # targTmTrcShrtStartStdArr = np.stack(targStdShrtStartList).view(BinnedSpikeSet)
        
        # groupedSpikesTrialShortStartAvg.append([targTmTrcShrtStartAvgArr, targTmTrcShrtStartStdArr])
        groupedSpikesTrialAvg.append([targTmTrcAvgArr, targTmTrcStdArr])
        # groupedSpikesEndTrialAvg.append([targTmTrcAvgEndArr, targTmTrcStdEndArr])
        grpLabels.append(uniqueTargAngle)
    
    return groupedSpikesTrialAvg, grpLabels
    

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

def gpfaComputation(listBSS, descriptions, outputPaths, timeBeforeAndAfterStart = None, timeBeforeAndAfterEnd = None, balanceDirs = True, baselineSubtract = True, numStimulusConditions = 1,combineConditions = False, sqrtSpikes = False, forceNewGpfaRun=False,
                    crossvalidateNumFolds = 4, xDimTest = [2,5,8], firingRateThresh=0.5, signalDescriptor = "",plotOutput=True):
    
    # from classes.GPFA import GPFA
    from mpl_toolkits.mplot3d import Axes3D # for 3d plotting
    from methods.GeneralMethods import prepareMatlab
    from multiprocessing import pool
    
    eng = None #prepareMatlab()
    
    
   
        
    if plotOutput:
        plotInfo = {}
        figErr = plt.figure()
        figErr.suptitle('dimensionality vs. GPFA log likelihood')
        axs = figErr.subplots(nrows=2, ncols=len(listBSS), squeeze=False)
        
    dimExp = []
    dimMoreLL = []
    gpfaOutDimAll = []
    gpfaTestIndsOutAll = []
    gpfaTrainIndsOutAll = []
    # with pool.Pool() as pool:
    #     results = []
    for idx, (bnSp, description, outputPath) in enumerate(zip(listBSS, descriptions, outputPaths)):
        
        print("*** Running GPFA for " + description.replace('\n', ' ') + " ***")

        if plotOutput:
            axScore = axs[0, :].flat[idx]
            axDim = axs[1,:].flat[idx]
            plotInfo['axScore'] = axScore
            plotInfo['axDim'] = axDim
        else:
            plotInfo = None # don't plot for now...
   
        if type(bnSp) is list:
            numDims = []
            numDimsLL = []
            gpfaOutDim = []
            gpfaTestIndsOut = []
            gpfaTrainIndsOut = []
            for subsetNum, bS in enumerate(bnSp):
                numDimsSubst, numDimsLLSubst, gpfaOutDimSubst, gpfaTestIndsOutSubst, gpfaTrainIndsOutSubst = BinnedSpikeSet.gpfa(bS, eng,description,outputPath, signalDescriptor = Path(signalDescriptor) / ("shuff_%d" % subsetNum) , forceNewGpfaRun = forceNewGpfaRun,
                          xDimTest = xDimTest, crossvalidateNum = crossvalidateNumFolds, firingRateThresh = firingRateThresh, baselineSubtract = baselineSubtract, numConds=numStimulusConditions,combineConds = combineConditions, sqrtSpikes=sqrtSpikes,
                          timeBeforeAndAfterStart = timeBeforeAndAfterStart, timeBeforeAndAfterEnd=timeBeforeAndAfterEnd, balanceDirs=balanceDirs, plotInfo = plotInfo)

                numDims.append(numDimsSubst)
                numDimsLL.append(numDimsLLSubst)
                gpfaOutDim.append(gpfaOutDimSubst)
                gpfaTestIndsOut.append(gpfaTestIndsOutSubst)
                gpfaTrainIndsOut.append(gpfaTrainIndsOutSubst)
        else:
            numDims, numDimsLL, gpfaOutDim, gpfaTestIndsOut, gpfaTrainIndsOut = BinnedSpikeSet.gpfa(bnSp, eng,description,outputPath, signalDescriptor = signalDescriptor, forceNewGpfaRun = forceNewGpfaRun,
                      xDimTest = xDimTest, crossvalidateNum = crossvalidateNumFolds, firingRateThresh = firingRateThresh, baselineSubtract = baselineSubtract, numConds=numStimulusConditions,combineConds = combineConditions, sqrtSpikes=sqrtSpikes,
                      timeBeforeAndAfterStart = timeBeforeAndAfterStart, timeBeforeAndAfterEnd=timeBeforeAndAfterEnd, balanceDirs=balanceDirs, plotInfo = plotInfo)
            
        dimExp.append(numDims)
        dimMoreLL.append(numDimsLL)
        gpfaOutDimAll.append(gpfaOutDim)
        gpfaTestIndsOutAll.append(gpfaTestIndsOut)
        gpfaTrainIndsOutAll.append(gpfaTrainIndsOut)
        # 
        # resGrouped = list(zip(*results))
        # dimExp = list(resGrouped[0])
        # gpfaPrepAll = list(resGrouped[1])
        
    if plotOutput:
        pass
#        axScore.legend()
    
    return dimExp, dimMoreLL, gpfaOutDimAll, gpfaTestIndsOutAll, gpfaTrainIndsOutAll

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
        
    ax.legend(loc="upper right")
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
                ax.scatter(scMn, scStd**2, color = 'C%d' % colInd)
        else:
            units = bnSp.units # replace every time, but should be the same for all...
            if units == 'count':
                scMn = bnSp.sumTrialCountByChannel()
                scStd = bnSp.stdSumTrialCountByChannel()
            elif units == 'Hz':
                scMn = bnSp.avgFiringRateByChannel()
                scStd = bnSp.stdFiringRateByChannel()
            ax.scatter(scMn, scStd**2, color = 'C%d' % colInd)
       
    maxX = np.max(ax.get_xlim())
    maxY = np.max(ax.get_ylim())
    maxDiagLine = np.min([maxX, maxY])
    ax.plot([0,maxDiagLine],[0,maxDiagLine], linestyle='--')

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
