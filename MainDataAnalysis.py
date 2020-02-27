#%%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 15:37:02 2020

@author: emilio
"""
from MatFileMethods import LoadMatFile#, QuashStructure
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import use as mtpltUse
from matplotlib import get_backend as mtpltWhatBackend
mtpltUse('qt5agg') # nobody knows why, but this is necessary to not plot a million figures and *also* the combo tabs...
from IPython import get_ipython
from plotWindow.plotWindow import plotWindow

from classes.Dataset import Dataset
from classes.BinnedSpikeSet import BinnedSpikeSet

import os

data = []
data.append({'description': 'Earl 2019-03-18 M1 - MGR',
              'path': '/Users/emilio/Documents/BatistaLabData/memoryGuidedReach/Earl/2019/03/18/',
              'processor': 'Erinn'});
data.append({'description': 'Earl 2019-03-22 M1 - MGR',
              'path': '/Users/emilio/Documents/BatistaLabData/memoryGuidedReach/Earl/2019/03/22/',
              'processor': 'Erinn'});
data.append({'description': 'Pepe Array 1 2018-07-14 PFC - MGS',
            'path': '/Users/emilio/Documents/BatistaLabData/memoryGuidedSaccade/Pepe/2018/07/14/Array1/',
            'processor': 'Yuyan'});
data.append({'description': 'Pepe Array 2 2018-07-14 PFC - MGS',
            'path': '/Users/emilio/Documents/BatistaLabData/memoryGuidedSaccade/Pepe/2018/07/14/Array2/',
            'processor': 'Yuyan'});
data.append({'description': 'Wakko Array 1 2018-02-11 PFC - MGS',
            'path': '/Users/emilio/Documents/BatistaLabData/memoryGuidedSaccade/Wakko/2018/02/11/Array1/',
            'processor': 'Yuyan'});
data.append({'description': 'Wakko Array 2 2018-02-11 PFC - MGS',
            'path': '/Users/emilio/Documents/BatistaLabData/memoryGuidedSaccade/Wakko/2018/02/11/Array2/',
            'processor': 'Yuyan'});

processedDataMat = 'processedData.mat'

dataset = []
cosTuningCurves = []
for dataUse in data:
#    dataUse = data[1]
    dataMatPath = os.path.join(dataUse['path'], processedDataMat)
    
    print('processing data set ' + dataUse['description'])
    datasetHere = Dataset(dataMatPath, dataUse['processor'], notChan = [31,0])
    dataset.append(datasetHere)
    
    
    cosTuningCurves.append(datasetHere.computeCosTuningCurves())
    #dataset.plotTuningCurves()
    
datasetSuccess = []
datasetSuccessNoCatch = []
for dset in dataset:
    dsetSuccess = dset.successfulTrials()
    datasetSuccess.append(dsetSuccess)
    dsetSuccessNoCatch = dsetSuccess.trialsWithoutCatch()
    datasetSuccessNoCatch.append(dsetSuccessNoCatch)

#%%
# return binned spikes
binnedSpikes = []
binnedSpikesAll = []
binnedSpikesOnlyDelay = []
binnedSpikesEnd = []
groupedSpikesTrialAvg = []
groupedSpikesEndTrialAvg = []
grpLabels = []

endDelaysFromStartList = []
startDelaysList = []
binnedSpikesShortStart = []
binnedSpikesShortEnd = []
groupedSpikesTrialShortStartAvg = []

dtstTimelim = []
for dataStInit in datasetSuccessNoCatch:
    startDelay, endDelay = dataStInit.computeDelayStartAndEnd()
    startDelayArr = np.asarray(startDelay)
    endTimeArr = np.asarray(endDelay)
    delayTimeArr = endTimeArr - startDelayArr
    
    lenSmallestTrl = 251 #ms
    furthestBack = 251 #ms
    furthestForward = 251
    binSizeMs = 50# 1000 # good for PFC LDA
    dtstTimelim.append(dataStInit.filterTrials(delayTimeArr>lenSmallestTrl))
    dataSt = dtstTimelim[-1]
    dataSt.computeCosTuningCurves()
    startDelay, endDelay = dataSt.computeDelayStartAndEnd()
    startDelayArr = np.asarray(startDelay)
    startDelaysList.append(startDelayArr)
    endDelayArr = np.asarray(endDelay)
    endDelaysFromStartList.append(endDelayArr-startDelayArr)
    endTimePad = 0
    endTimeArr = startDelayArr+endTimePad
    
    binnedSpikesHere = dataSt.binSpikeData(startMs = list(startDelayArr-furthestBack), endMs = list(endTimeArr+furthestForward), binSizeMs=binSizeMs, notChan=[31, 0])
    binnedSpikes.append(binnedSpikesHere)
    
    binnedSpikesHereAll = dataSt.binSpikeData(startMs = list(startDelayArr-furthestBack), endMs = list(endDelayArr+furthestForward), binSizeMs=binSizeMs, notChan=[31, 0])
    binnedSpikesAll.append(binnedSpikesHereAll)
    
    binnedSpikesHereOnlyDelay = dataSt.binSpikeData(startMs = list(startDelayArr), endMs = list(endDelayArr), binSizeMs=binSizeMs, notChan=[31, 0])
    binnedSpikesOnlyDelay.append(binnedSpikesHereOnlyDelay)
    
    binnedSpikesHereEnd = dataSt.binSpikeData(startMs = list(endDelayArr-furthestBack), endMs = list(endDelayArr+furthestForward), binSizeMs=binSizeMs, notChan=[31, 0])
    binnedSpikesEnd.append(binnedSpikesHereEnd)
    
    binnedSpikesHereShortStart = dataSt.binSpikeData(startMs = list(startDelayArr), endMs = list(endTimeArr+lenSmallestTrl), binSizeMs=binSizeMs, notChan=[31, 0])
    binnedSpikesShortStart.append(binnedSpikesHereShortStart)
    
    binnedSpikesHereShortEnd = dataSt.binSpikeData(startMs = list(endDelayArr-lenSmallestTrl), endMs = list(endDelayArr), binSizeMs=binSizeMs, notChan=[31, 0])
    binnedSpikesShortEnd.append(binnedSpikesHereShortEnd)
    
    uniqueTargAngle, trialsPresented = np.unique(dataSt.markerTargAngles, axis=0, return_inverse=True)

    groupedSpikes = dataSt.groupSpikes(trialsPresented, uniqueTargAngle, binnedSpikes = binnedSpikesHere)
    groupedSpikesEnd = dataSt.groupSpikes(trialsPresented, uniqueTargAngle, binnedSpikes = binnedSpikesHereEnd)
    groupedSpikesShortStart = dataSt.groupSpikes(trialsPresented, uniqueTargAngle, binnedSpikes = binnedSpikesHereShortStart)
    
    targAvgList, targStdList = zip(*[(groupedSpikes[targ].trialAverage(), groupedSpikes[targ].trialStd()) for targ in range(0, len(groupedSpikes))])
    targTmTrcAvgArr = np.stack(targAvgList).view(BinnedSpikeSet)
    targTmTrcStdArr = np.stack(targStdList).view(BinnedSpikeSet)
    targAvgListEnd, targStdListEnd = zip(*[(groupedSpikesEnd[targ].trialAverage(), groupedSpikesEnd[targ].trialStd()) for targ in range(0, len(groupedSpikesEnd))])
    targTmTrcAvgEndArr = np.stack(targAvgListEnd).view(BinnedSpikeSet)
    targTmTrcStdEndArr = np.stack(targStdListEnd).view(BinnedSpikeSet)
    
    targAvgShrtStartList, targStdShrtStartList = zip(*[(groupedSpikesShortStart[targ].trialAverage(), groupedSpikesShortStart[targ].trialStd()) for targ in range(0, len(groupedSpikesShortStart))])
    targTmTrcShrtStartAvgArr = np.stack(targAvgShrtStartList).view(BinnedSpikeSet)
    targTmTrcShrtStartStdArr = np.stack(targStdShrtStartList).view(BinnedSpikeSet)
    
    groupedSpikesTrialShortStartAvg.append([targTmTrcShrtStartAvgArr, targTmTrcShrtStartStdArr])
    groupedSpikesTrialAvg.append([targTmTrcAvgArr, targTmTrcStdArr])
    groupedSpikesEndTrialAvg.append([targTmTrcAvgEndArr, targTmTrcStdEndArr])
    grpLabels.append(uniqueTargAngle)

#%% dimensionality calculation
numDims = []
for idx, bnSp in enumerate(binnedSpikesShortEnd):
    # grpSpikes = bnSp.groupByLabel(dtstTimelim[idx].markerTargAngles)
    # [bnSpGrp.numberOfDimensions(title = data[idx]['description'], maxDims = 30, baselineSubtract = True) for bnSpGrp in grpSpikes]
    baselineSubtract = True
    numDims.append(bnSp.numberOfDimensions(title = data[idx]['description'], labels=dtstTimelim[idx].markerTargAngles, maxDims = 30, baselineSubtract = False))
#    numDims.append(bnSp.numberOfDimensions(title = data[idx]['description'], labels=None, maxDims = 30))
    
#%% PCA projections
for idx, bnSp in enumerate(binnedSpikesShortEnd):
    bnSp.channelsAboveThresholdFiringRate(firingRateThresh=1)[0].pca(labels = dtstTimelim[idx].markerTargAngles, plot = True)
    plt.suptitle(data[idx]['description'])
    
#%% LDA projections
for idx, bnSp in enumerate(binnedSpikesShortEnd):
    bnSp.lda(dtstTimelim[idx].markerTargAngles, plot=True)
    plt.suptitle(data[idx]['description'])
    
#%% Noise correlations
for idx, bnSp in enumerate(binnedSpikesShortEnd):
    plotResid = True
    separateNoiseCorrForLabels = True
    residualSpikes = bnSp.channelsAboveThresholdFiringRate(firingRateThresh=1)[0].residualCorrelations(dtstTimelim[idx].markerTargAngles, plot=plotResid, separateNoiseCorrForLabels = separateNoiseCorrForLabels, figTitle = data[idx]['description'])
    # residualSpikes.pca(labels = dtstTimelim[idx].markerTargAngles, plot = True)
    # plt.suptitle(data[idx]['description'] + " residuals PCA")
    # residualSpikes.lda(labels = dtstTimelim[idx].markerTargAngles, plot = True)
    # plt.suptitle(data[idx]['description'] + " residuals LDA")
    
    labels=dtstTimelim[idx].markerTargAngles
    uniqueLabel, labelPresented = np.unique(labels, axis=0, return_inverse=True)
    # residualSpikes[labelPresented==0].numberOfDimensions(title=data[idx]['description'] + " residuals", maxDims = 30)
    
    
#%% some descriptive data plots
numCols = 3
numRows = 3

ind = 0
notRndRndChan =  [23,83, 20, 10, 15, 20]# [,23] [23, 79] [5,83]
for grpSpike, grpSpikeShrtStrt, grpSpikeEnd, grpLabel in zip(groupedSpikesTrialAvg, groupedSpikesTrialShortStartAvg, groupedSpikesEndTrialAvg, grpLabels):
    
    rndChan = notRndRndChan[ind]#np.random.randint(grpSpike.shape[1])
    chanRespMean = np.squeeze(grpSpike[0][:,[rndChan], :])
    chanRespStd = np.squeeze(grpSpike[1][:,[rndChan],:])
    chanRespEndMean = np.squeeze(grpSpikeEnd[0][:, [rndChan], :])
    chanRespEndStd = np.squeeze(grpSpikeEnd[1][:, [rndChan], :])
#    chanResp.timeAverage()
    chanTmAvg = grpSpikeShrtStrt[0][:,[rndChan],:].timeAverage()
    chanTmStd = grpSpikeShrtStrt[0][:,[rndChan],:].timeStd()
    
    
    plt.figure()
    plt.suptitle(data[ind]['description'] + ': channel ' + str(rndChan))
    sbpltAngs = np.arange(3*np.pi/4, -5*np.pi/4, -np.pi/4)
    # Add nan to represent center for tuning polar curve...
    sbpltAngs = np.concatenate((sbpltAngs[0:3], sbpltAngs[[-1]], np.expand_dims(np.asarray(np.nan),axis=0), sbpltAngs[[3]], np.flip(sbpltAngs[4:-1])), axis=0)
    axVals = np.empty((0,4))
    axes = []
    for idx in range(0, len(grpLabel)):
        subplotChoose = np.where([np.allclose(grpLabel[idx] % (2*np.pi), sbpltAngs[i] % (2*np.pi)) for i in range(0,len(sbpltAngs))])
        subplotChooseDelayStart = subplotChoose[0][0]*2
        subplotChooseDelayEnd = subplotChoose[0][0]*2+1
        axes.append(plt.subplot(numRows, numCols*2, subplotChooseDelayStart+1))
        tmValStart = np.arange(-furthestBack+binSizeMs/2, furthestForward+endTimePad-binSizeMs/2, binSizeMs)
        plt.plot(tmValStart, chanRespMean[idx, :])
        plt.fill_between(tmValStart, chanRespMean[idx, :]-chanRespStd[idx,:], chanRespMean[idx, :]+chanRespStd[idx,:], alpha=0.2)
        axVals = np.append(axVals, np.array(plt.axis())[None, :], axis=0)
        axes.append(plt.subplot(numRows, numCols*2, subplotChooseDelayEnd+1))
        tmValEnd = np.arange(-furthestBack+binSizeMs/2, furthestForward-binSizeMs/2, binSizeMs)
        plt.plot(tmValEnd, chanRespEndMean[idx, :])
        plt.fill_between(tmValEnd, chanRespEndMean[idx, :]-chanRespEndStd[idx,:], chanRespEndMean[idx, :]+chanRespEndStd[idx,:], alpha=0.2)
        axVals = np.append(axVals, np.array(plt.axis())[None, :], axis=0)
        if subplotChoose[0][0] == 0:
            axSt = axes[-2]
            axSt.set_ylabel('Average Firing Rate (Hz)')
        elif subplotChoose[0][0] == (numCols*numRows - np.floor(numCols/2)):
            axSt = axes[-2]
            axSt.set_xlabel('Time Around Delay Start or End (ms)')
            
        if subplotChoose[0][0] == np.floor(numCols/2):
            axSt = axes[-2]
            axSt.set_title('Start')
            axEnd = axes[-1]
            axEnd.set_title('End')
        
        if subplotChoose[0][0] % numCols:
            axEn = axes[-1]
            axSt = axes[-2]
            axSt.yaxis.set_ticklabels('')
            axEn.yaxis.set_ticklabels('')
        else:
            axEn = axes[-1]
            axEn.yaxis.set_ticklabels('')
            
        if subplotChoose[0][0] < (numCols*numRows - numCols):
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
    ind += 1

ind = 0
for dtst in dtstTimelim:
    plt.figure();
    plt.suptitle(data[ind]['description'])
#    _, trialsPresented = np.unique(dtst.markerTargAngles, axis=0, return_inverse=True)
    angsPres = dtst.markerTargAngles
    angsPres[angsPres>np.pi] = np.pi - (angsPres[angsPres>np.pi] % (2*np.pi))
    plt.hist(angsPres,bins = np.arange(-7*np.pi/8, 10*np.pi/8, np.pi/4))
    plt.title('distribution of directions')
    plt.xlabel('direction angle (rad)')
    plt.ylabel('count')
    ind += 1
#%%
plt.figure()
[plt.boxplot(dts.cosTuningCurveParams['bslnPerChan'], positions=[idx], labels=[data[idx]['description']]) for idx, dts in enumerate(dtstTimelim)]
plt.title('Baseline Firing Rate')
plt.ylabel('firing rate (Hz)')
plt.figure()
[plt.boxplot(dts.cosTuningCurveParams['modPerChan']/dts.cosTuningCurveParams['bslnPerChan'], positions=[idx], labels=[data[idx]['description']]) for idx, dts in enumerate(dtstTimelim)]
plt.title('Firing Rate Modulation (normalized by baseline)')
plt.ylabel('normalized modulation around baseline')

#%% some GPFA junk...
from matlab import engine
from classes.GPFA import GPFA
from mpl_toolkits.mplot3d import Axes3D # for 3d plotting


try:
    eng.workspace
except (engine.RejectedExecutionError, NameError) as e:
    eng = engine.start_matlab()
finally:
    eng.clear('all', nargout=0)
    
binnedSpikesShort = binnedSpikesShortEnd
dataInd = -2
# for dataInd in range(len(binnedSpikesAll)):
for _ in range(1):
    onlyDelay = True
    initRandSt = np.random.get_state()
    np.random.seed(0)
    if onlyDelay:
        trlIndsUse = BinnedSpikeSet.balancedTrialInds(binnedSpikesOnlyDelay[dataInd], dtstTimelim[dataInd].markerTargAngles)
        tmValsStart = np.arange(0, 250, binSizeMs)
        tmValsEnd = np.arange(-250, 0, binSizeMs)       
        # tmValsStart = np.arange(0, furthestBack if furthestBack else 251, binSizeMs)
        # tmValsEnd= np.arange(-furthestForward if furthestForward else 251, 0, binSizeMs)
        binnedSpikesBalanced = [binnedSpikesOnlyDelay[dataInd][trl] for trl in trlIndsUse]
    else:
        trlIndsUse = BinnedSpikeSet.balancedTrialInds(binnedSpikesAll[dataInd], dtstTimelim[dataInd].markerTargAngles)
        tmValsStart = np.arange(-furthestBack, furthestBack if furthestBack else 251, binSizeMs)
        tmValsEnd = np.arange(-(furthestForward if furthestForward else 251)+binSizeMs, furthestForward, binSizeMs)
        binnedSpikesBalanced = [binnedSpikesAll[dataInd][trl] for trl in trlIndsUse]
        
    np.random.set_state(initRandSt)
        
    endDelaysListBal = endDelaysFromStartList[dataInd][trlIndsUse]
    uniqueTargAngle, trialsPresented = np.unique(dtstTimelim[dataInd].markerTargAngles[trlIndsUse], axis=0, return_inverse=True)
    binSpkAllArr = np.empty(len(binnedSpikesBalanced), dtype=object)
    for idx, _ in enumerate(binSpkAllArr):
        binSpkAllArr[idx] = binnedSpikesBalanced[idx]
        
    groupedBalancedSpikes = Dataset.groupSpikes(_, trialsPresented, uniqueTargAngle, binnedSpikes = binSpkAllArr); trialsPresented = np.sort(trialsPresented) #[binSpkAllArr] #
    # binnedSpikesBalanced.groupByLabel(dtstTimelim[dataInd].markerTargAngles[trlIndsUse])
    #groupedBalancedEndDelays = Dataset.groupSpikes(_, trialsPresented, uniqueTargAngle, binnedSpikes= endDelaysListBal)
    groupedBalancedSpikes = [grp.tolist() for grp in groupedBalancedSpikes]
    
    uniqueTargAngleDeg = uniqueTargAngle*180/np.pi
    
    colorset = np.array([[127,201,127],[190,174,212],[253,192,134],[200,200,153],[56,108,176],[240,2,127],[191,91,23],[102,102,102]])/255
    for idx, grpSpks in enumerate(groupedBalancedSpikes):
        xDim = 8; # default, haven't decided how to be smart about choice yet
        gpfaPrep = GPFA(grpSpks)
        estParams, seqTrainNew, seqTestNew = gpfaPrep.runGpfaInMatlab(eng=eng, crossvalidate=True, xDim=xDim)
        
        
        rowsPlt = 2
        colsPlt = xDim/rowsPlt*2 # align to start and to end...
        axesStart = []
        axesEnd = []
        axVals = np.empty((0,4))
        figSep = plt.figure()
        fig3 = plt.figure()
        axStart3d = plt.subplot(1,3,1,projection='3d')
        axEnd3d = plt.subplot(1,3,2,projection='3d')
        axAll3d = plt.subplot(1,3,3,projection='3d')
        plt.suptitle(data[dataInd]['description'] + " " + str(int(uniqueTargAngleDeg[idx])) + " deg")
        
        for sq, trlInd in zip(seqTestNew, gpfaPrep.testInds):
            
            plt.figure(fig3.number)
            axStart3d.plot(sq['xorth'][2,:tmValsStart.shape[0]], sq['xorth'][3,:tmValsStart.shape[0]], sq['xorth'][4,:tmValsStart.shape[0]],
                       color=colorset[trialsPresented[trlInd],:], linewidth=0.4)
            axStart3d.set_title('Start')
            axEnd3d.plot(sq['xorth'][2,-tmValsEnd.shape[0]:], sq['xorth'][3,-tmValsEnd.shape[0]:], sq['xorth'][4,-tmValsEnd.shape[0]:],
                       color=colorset[trialsPresented[trlInd],:], linewidth=0.4)
            axEnd3d.set_title('End')
            axAll3d.plot(sq['xorth'][2,:], sq['xorth'][3,:], sq['xorth'][4,:],
                       color=colorset[trialsPresented[trlInd],:], linewidth=0.4)
            axAll3d.set_title('All')
            
            if True:#trialsPresented[trlInd] == 3:
                pltNum = 1
                plt.figure(figSep.number)
                for dim in sq['xorth']:
                    if len(axesStart) + len(axesEnd) <rowsPlt*colsPlt:
                        axesHere = figSep.add_subplot(rowsPlt, colsPlt, pltNum)
                        if pltNum <= colsPlt:
                            axesHere.set_title("Start")
                    else:
                        axesHere = axesStart[int((pltNum-1)/2)]
                        plt.axes(axesHere)
                    
                    
                    plt.plot(tmValsStart, dim[:tmValsStart.shape[0]], color=colorset[trialsPresented[trlInd],:], linewidth=0.4)
                    
                    
                    
                    axVals = np.append(axVals, np.array(axesHere.axis())[None, :], axis=0)
                    axesStart.append(axesHere)
                    pltNum += 1
                    
                    if len(axesStart) + len(axesEnd) <rowsPlt*colsPlt:
                        axesHere = figSep.add_subplot(rowsPlt, colsPlt, pltNum)
                        if pltNum <= colsPlt:
                            axesHere.set_title("End")
                    else:
                        axesHere = axesEnd[int(pltNum/2-1)]
                        plt.axes(axesHere)
                        
            #        endTm = groupedBalancedEndDelays[0][trlInd]
                    # endTm = 0 # we align to the end now...
                    plt.plot(tmValsEnd, dim[-tmValsEnd.shape[0]:], color=colorset[trialsPresented[trlInd],:], linewidth=0.4)
            #        endInd = endTm/binSizeMs
            #        endIndFl = int(np.floor(endTm/binSizeMs))
            #        endIndNxt = endIndFl + 1
            #        endTmVal = (dim[endIndNxt] - dim[endIndFl])*(endInd-endIndFl) + dim[endIndFl]
            #        plt.plot(endTm, endTmVal, 'o',color = [1,0,0])
                    axVals = np.append(axVals, np.array(axesHere.axis())[None, :], axis=0)
                    axesEnd.append(axesHere)
                    pltNum += 1
                
        ymin = np.min(np.append(0, np.min(axVals, axis=0)[2]))
        ymax = np.max(axVals, axis=0)[3]
        for ax in axesStart:
            ax.set_ylim(bottom = ymin, top = ymax )
            plt.axes(ax)
            plt.axvline(x=0, linestyle='--')
            
        for ax in axesEnd:
            ax.set_ylim(bottom = ymin, top = ymax )
            plt.axes(ax)
            plt.axvline(x=0, linestyle='--')
            
        

