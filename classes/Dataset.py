#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 10:46:04 2020

@author: emilio
"""
from MatFileMethods import LoadMatFile
import numpy as np
from classes.BinnedSpikeSet import BinnedSpikeSet
from copy import copy
from matplotlib import pyplot as plt
from mayavi import mlab

class Dataset():
    
    colorset = np.array([[127,201,127],[190,174,212],[253,192,134],[200,200,153],
                         [56,108,176],[240,2,127],[191,91,23],[102,102,102]])/255
    colorsetMayavi = [tuple(col) for col in colorset]
    
    def __init__(self, dataMatPath, preprocessor, notChan=None, removeCoincidentChans = True):
        annots = LoadMatFile(dataMatPath)
        self.cosTuningCurveParams = {'thBestPerChan': np.empty(0), 'modPerChan': np.empty(0), 'bslnPerChan': np.empty(0), 'tuningCurves': np.empty(0)}
        if preprocessor == 'Erinn':
        
            self.trialStatuses = np.stack([trlParam['trialStatus'][0][0] for trlParam in annots['Data']['Overview'][0]]).squeeze()
            
            self.spikeDatTimestamps = np.stack([trlDat['spikes'][0][0]['timestamps'][0] for trlDat in annots['Data']['TrialData'][0]])
            self.isi = np.empty(self.spikeDatTimestamps.shape, dtype=object) # initialize to right size
            with np.nditer([self.isi, self.spikeDatTimestamps], ['refs_ok'], [['writeonly'], ['readonly']]) as iterRef:
                for valOut, valIn in iterRef:
                    if valIn[()].size != 0:
                        valOut[()] = 1/np.diff(valIn[()])
            
            self.spikeDatChannels = np.stack([trlDat['spikes'][0][0]['channel'][0] for trlDat in annots['Data']['TrialData'][0]])
            
            self.spikeDatSort = np.stack([trlDat['spikes'][0][0]['sort'][0] for trlDat in annots['Data']['TrialData'][0]])
            
            markerTargWindows = np.stack([trlParam['MarkerTargets'][0][0]['window'][0] for trlParam in annots['Data']['Parameters'][0]])
            
            markerTargWindowNames = np.stack([trlParam['MarkerTargets'][0][0]['targetName'][0] for trlParam in annots['Data']['Parameters'][0]])
            
            _, self.markerTargAngles = self.computeTargetCoordsAndAngle(markerTargWindows, markerTargWindowNames, 'ReachTargetAppear')
            
            self.statesPresented = [trlDat['stateTransitions'][0][0] for trlDat in annots['Data']['TrialData'][0]]
            
            self.stateNames = annots['Data']['Parameters'][0][0]['stateNames'][0][0]
            stateHandLoc = annots['Data']['Parameters'][0][0]['StateTable'][0,0][0,0]['Hand'][0,0]['window'][0,0:3]
            
            kinematicsTemp = [trlDat['HandKinematics'][0][0]['cursorKinematics'][0][0] for trlDat in annots['Data']['TrialData'][0]]
            for idx, struc in enumerate(kinematicsTemp):
                emptyObjArr = np.empty((0,0), dtype='uint8')
                # fa = np.array([[(emptyObjArr)]], dtype=[('hah', 'object')])
                #np.ndarray((0,0), dtype='int8')# np.array(([[np.ndarray((0,0), dtype='int8')]]), dtype=[('0', 'object')])
                np.array(())
                if not struc.size:
                    kinematicsTemp[idx] = np.array([[(emptyObjArr,emptyObjArr,emptyObjArr,emptyObjArr)]], dtype=[('time', 'O'), ('position', 'O'), ('velocity', 'O'), ('acceleration', 'O')])
            self.kinematics = np.stack(kinematicsTemp)
            self.kinematicCenter = stateHandLoc
        elif preprocessor == 'Yuyan':
            spikesArrayTrlChanTm = annots['S']['spikes'][0]
            self.trialStatuses = np.ones((spikesArrayTrlChanTm.shape[0]))
            
            # note that the [:,None].T part is to make the shape the same as Erinn's preprocessing above...
            weirdIntermediate = np.squeeze(np.stack([[np.where(chan)[0][:, None] for chan in trl] for trl in spikesArrayTrlChanTm]))
            with np.nditer(weirdIntermediate, ['refs_ok'], ['readwrite']) as iterRef:
                for valIn in iterRef:
                    valIn[()] = valIn[()].T
                    
            self.spikeDatTimestamps = weirdIntermediate
            self.isi = np.empty(self.spikeDatTimestamps.shape, dtype=object) # initialize to right size
            with np.nditer([self.isi, self.spikeDatTimestamps], ['refs_ok'], [['writeonly'], ['readonly']]) as iterRef:
                for valOut, valIn in iterRef:
                    if valIn[()].size != 0:
                        valOut[()] = 1/np.diff(valIn[()])
            
            spikeDatChannels = np.stack(range(0, spikesArrayTrlChanTm.shape[1]))
            self.spikeDatChannels = np.repeat(spikeDatChannels[None, :], len(self.trialStatuses), axis=0)
            
            self.spikeDatSort = np.ones(spikesArrayTrlChanTm.shape[0:2])
            
            self.markerTargAngles = np.expand_dims(np.squeeze(annots['S']['angle']),axis=1)/180*np.pi #gotta conver these angles to radians to match above...
            
            # this is a (hopefully) temporary hard coding to reflect how Yuyan
            # preprocessed MGS data to include only the delay period 
            # Note that it's one-indexed to reflect Matlab's 1-indexing from Erinn's data...
            # Also note that we're adding a separate 'Target Flash' state, as in MGR
            # data the 'Delay Period' indicates the time when the monkey would have been
            # consciously aware of the flash, so ~100ms later. Here, we're just assuming
            # they happen at the same time
            self.statesPresented = [np.stack([[1, 2, -1], [0, 0, sATCT.shape[1]]]) for sATCT in spikesArrayTrlChanTm]
            
            stateNames = np.array(['Target Flash', 'Delay Period'])
            self.stateNames = stateNames[None, :]
            
            self.kinematics = None
        
        if notChan is not None:
            self.removeChannels(notChan)
        if removeCoincidentChans:
            self.removeCoincidentSpikes()
            
    def filterTrials(self, mask):
        filteredTrials = copy(self)
        filteredTrials.trialStatuses = filteredTrials.trialStatuses[mask]
        filteredTrials.spikeDatTimestamps = filteredTrials.spikeDatTimestamps[mask, :]
        filteredTrials.isi = filteredTrials.isi[mask, :]
        filteredTrials.spikeDatChannels = filteredTrials.spikeDatChannels[mask, :]
        filteredTrials.spikeDatSort = filteredTrials.spikeDatSort[mask, :]
        filteredTrials.markerTargAngles = filteredTrials.markerTargAngles[mask, :]
        filteredTrials.stateNames = filteredTrials.stateNames
        filteredTrials.statesPresented = [filteredTrials.statesPresented[hr] for hr in range(0, len(filteredTrials.statesPresented)) if mask[hr]]
        filteredTrials.cosTuningCurveParams = {'thBestPerChan': np.empty(0), 'modPerChan': np.empty(0), 'bslnPerChan': np.empty(0), 'tuningCurves': np.empty(0)}
        if filteredTrials.kinematics is not None:
            filteredTrials.kinematics = filteredTrials.kinematics[mask]
        return filteredTrials
        
    def successfulTrials(self):
        successfulTrials = self.filterTrials(self.trialStatuses==1)
        return successfulTrials
        
    def failTrials(self):
        failTrials = self.filterTrials(self.trialStatuses==0)
        return failTrials
    
    def trialsWithoutCatch(self):
        stNm = self.stateNames
        #stNm = stNm[None, :]
        stNm = np.repeat(stNm, len(self.statesPresented), axis=0)
        stPres = self.statesPresented
        stPres = [stPr[0, stPr[0]!=-1] for stPr in stPres]
        stPresTrl = zip(stPres, stNm, range(0, len(self.statesPresented)))
        targetWithCatch = np.stack([np.any(np.core.defchararray.find(np.stack(stNm[stPres-1]),'Catch')!=-1) for stPres, stNm, trialNum in stPresTrl])
        trialsWithoutCatchLog = np.logical_not(targetWithCatch)
        datasetWithoutCatch = self.filterTrials(trialsWithoutCatchLog)
        return datasetWithoutCatch
    
    def binSpikeData(self, startMs=0, endMs=3600, binSizeMs=50, notChan = None):
        if type(endMs) is list:
            spikeDatBinnedList = [[[[counts for counts in np.histogram(chanSpks, bins=np.arange(sMs, eMs, binSizeMs))][0] for chanSpks in trlChans][0] for trlChans in trl] for trl, sMs, eMs in zip(self.spikeDatTimestamps, startMs, endMs)]
        else:
            spikeDatBinnedList = [[[[counts for counts in np.histogram(chanSpks, bins=np.arange(startMs, endMs, binSizeMs))][0] for chanSpks in trlChans][0] for trlChans in trl] for trl in self.spikeDatTimestamps]
        spikeDatBinnedArr = [np.stack(trl) for trl in spikeDatBinnedList]
        
        if notChan is not None:
            chansOfInt = self.maskAgainstChannels(notChan)
        else:
            chansOfInt = np.ones(spikeDatBinnedArr[0].shape[0])
            
        try:
            spikeDatBinned = np.stack(spikeDatBinnedArr)
        except ValueError:
            spikeDatBinned = [sp.view(BinnedSpikeSet) for sp in spikeDatBinnedArr]
            spikeDatBinned = [sp[chansOfInt, :] for sp in spikeDatBinned]
            for idx, sp in enumerate(spikeDatBinned):
                sp = sp/(binSizeMs/1000)
                sp.binSize = binSizeMs
                sp.start = startMs[idx]
                sp.end = endMs[idx]
                spikeDatBinned[idx] = sp
        else:
            spikeDatBinned = spikeDatBinned/(binSizeMs/1000)
            spikeDatBinned = spikeDatBinned.view(BinnedSpikeSet)
            spikeDatBinned.binSize = binSizeMs
            spikeDatBinned.start = startMs
            spikeDatBinned.end = endMs
            spikeDatBinned = spikeDatBinned[:, chansOfInt, :]
        
        
            
        return spikeDatBinned
    
    # coincidenceTime is how close spikes need to be, in ms, to be considered coincicdent, 
    # coincidenceThresh in percent/100 (i.e. 20% = 0.2)
    # checkNumTrls is the percentage of trials to go through
    def removeCoincidentSpikes(self, coincidenceTime=1, coincidenceThresh=0.2, checkNumTrls=0.1):
        
        spikeCountOverallPerChan = np.stack(np.stack([np.concatenate(trlSpks[:], axis=1).shape[1] for trlSpks in self.spikeDatTimestamps.T]))
        
        # spikeCountOverallPerChan = [len(trlStmps)]
        
        coincCnt = np.zeros((spikeCountOverallPerChan.shape[0],  spikeCountOverallPerChan.shape[0]))
        
        numTrls = len(self.spikeDatTimestamps)
        trlIdxes = np.arange(0, numTrls)
        randIdxes = np.random.permutation(trlIdxes)
        idxesUse = randIdxes[:round(numTrls*checkNumTrls)]
        
        for numRound, trlIdx in enumerate(idxesUse):
            if not numRound % 10 and numRound != len(idxesUse)-1:
                print(str(100*numRound/(len(idxesUse)-1)) + "% done")
            if numRound == len(idxesUse)-1:
                print("100% done")
            trlTmstmp = self.spikeDatTimestamps[trlIdx]
            for idx1, ch1 in enumerate(trlTmstmp):
                # print(str(idx1))
                for idx2, ch2 in enumerate(trlTmstmp):
                    if idx2>idx1:
                        ch1ch2TDiff = ch1 - ch2.T
                        coincCnt[idx1, idx2] = coincCnt[idx1, idx2]+np.where(abs(ch1ch2TDiff)<coincidenceTime)[0].shape[0]
                        coincCnt[idx2, idx1] = coincCnt[idx1, idx2]
                    
        coincProp = coincCnt/spikeCountOverallPerChan
        
        chansKeep = np.unique(np.where(coincProp<coincidenceThresh)[1]) # every column is the division with that channel's spike count
        # badChans = np.unique(np.where(coincProp>=coincidentThresh)[1]) # every column is the division with channel's spike count
        
        self.keepChannels(chansKeep)
        
        # in place change, though returns channels with too much coincidence if desired...
        return chansKeep
    
    def removeChannels(self, notChan):
        keepChanMask = self.maskAgainstChannels(notChan)
        self.spikeDatTimestamps = np.stack([trlTmstmp[keepChanMask] for trlTmstmp in self.spikeDatTimestamps])
        self.spikeDatChannels = self.spikeDatChannels[:, keepChanMask]
        self.spikeDatSort = self.spikeDatSort[:, keepChanMask]
        
    def keepChannels(self, keepChan):
        keepChanMask = self.maskForChannels(keepChan)
        self.spikeDatTimestamps = np.stack([trlTmstmp[keepChanMask] for trlTmstmp in self.spikeDatTimestamps])
        self.spikeDatChannels = self.spikeDatChannels[:, keepChanMask]
        self.spikeDatSort = self.spikeDatSort[:, keepChanMask]
    
    def maskForChannels(self, chanList):
        channelMask = np.full(self.spikeDatSort[0].shape, False)
        for chan in chanList:
            channelMask = np.logical_or(channelMask, self.spikeDatSort[0] == chan)
            
        return channelMask
            
    def maskAgainstChannels(self, chanList):
        channelMask = np.full(self.spikeDatSort[0].shape, True)
        for chan in chanList:
            channelMask = np.logical_and(channelMask, self.spikeDatSort[0] != chan)
            
        return channelMask
    
    def computeTargetCoordsAndAngle(self, markTargWin, markTargWinNm, stateName = 'ReachTargetAppear'):
        targetInTrialsWithTrial = np.stack([np.append(target[0][0:3], trialNum) for targetsTrial, targetNamesTrial, trialNum in zip(markTargWin, markTargWinNm, range(0, len(markTargWin))) for target, name in zip(targetsTrial, targetNamesTrial) if name[0][0][0].find(stateName)!=-1])
        _, idx = np.unique(targetInTrialsWithTrial, axis=0, return_index=True)
        #idx = list(np.sort(idx)) # because targetInTrialsWithTrial is still a list, not an nparray
        targetsCoords = targetInTrialsWithTrial[np.sort(idx), 0:3]
        
        targCenter = np.median(targetsCoords, axis=0)
        targetsCoords = targetsCoords - targCenter
        targAngle = np.arctan2(targetsCoords[:, 1], targetsCoords[:, 0])
        targAngle = np.expand_dims(targAngle, axis=1)
        
        return targetsCoords, targAngle
    
    
    #%% methods to do stuff on data
    def computeCosTuningCurves(self, spikeDatBinnedArr = None, notChan=[31, 0]):
        uniqueTargAngle, trialsPresented = np.unique(self.markerTargAngles, axis=0, return_inverse=True)
        
        if spikeDatBinnedArr is None:
            spikeDatBinnedArr = self.binSpikeData(notChan=notChan)

#        spikeDatBinnedArrImpTmAvg = spikeDatBinnedArr.timeAverage()

        groupedSpikes = self.groupSpikes(trialsPresented, uniqueTargAngle, spikeDatBinnedArr)

        targAvgList = [groupedSpikes[targ].trialAverage() for targ in range(0, len(groupedSpikes))]
        targTmTrcAvgArr = np.stack(targAvgList).view(BinnedSpikeSet)
        
        self.cosTuningCurveParams['targAvgNumSpks'] = targTmTrcAvgArr.timeAverage()
        targAvgNumSpks = self.cosTuningCurveParams['targAvgNumSpks']

        predictors = np.concatenate((np.ones_like(uniqueTargAngle), np.sin(uniqueTargAngle), np.cos(uniqueTargAngle)), axis=1)
        out = [np.linalg.lstsq(predictors, np.expand_dims(targAvgNumSpks[:, chan], axis=1), rcond=None)[0]for chan in range(targAvgNumSpks.shape[1])]
        
        paramsPerChan = np.stack(out).squeeze()
        self.cosTuningCurveParams['thBestPerChan'] = np.arctan2(paramsPerChan[:, 1], paramsPerChan[:, 2])
        self.cosTuningCurveParams['modPerChan'] = np.sqrt(pow(paramsPerChan[:,2], 2) + pow(paramsPerChan[:,1], 2))
        self.cosTuningCurveParams['bslnPerChan'] = paramsPerChan[:, 0]
        
        thBestPerChan = self.cosTuningCurveParams['thBestPerChan']
        modPerChan = self.cosTuningCurveParams['modPerChan']
        bslnPerChan = self.cosTuningCurveParams['bslnPerChan']
        
        angs = np.linspace(np.min(uniqueTargAngle), np.min(uniqueTargAngle)+2*np.pi)
        cosTuningCurves = np.stack([bslnPerChan[chan] + modPerChan[chan]*np.cos(thBestPerChan[chan]-angs) for chan in range(0, len(modPerChan))])
        
        self.cosTuningCurveParams['tuningCurves'] = cosTuningCurves
        
        return cosTuningCurves
    
    # note that they get grouped in the order of the given labels
    def groupSpikes(self, groupAssignment, groupLabels, binnedSpikes = None, notChan = None):
        if binnedSpikes is None:
            binnedSpikes = self.binSpikeData(notChan = notChan)
            
        groupedSpikes = [binnedSpikes[groupAssignment==i] for i in range(0, len(groupLabels))]
        
        return groupedSpikes
    
    def timeOfState(self, state):
        statePresNum = np.where(self.stateNames == state)[1][0]+1 # remember Matlab is 1-indexed
        
        stateTmPres = []
        for statesPres in self.statesPresented:
            locStateLog = statesPres[0,:]==(statePresNum) # remember Matlab is 1-indexed
            if np.any(locStateLog):
                locState = np.where(locStateLog)[0][0]
                stateTmPres.append(statesPres[1, locState])
            else:
                stateTmPres.append(np.nan)
                
        return stateTmPres
            
    def computeDelayStartAndEnd(self):
        indDelayPres = np.where(self.stateNames == 'Delay Period')[1][0] # this is a horizontal vector...
        startTmsPres = []
        endTmsPres = []
        for statesPres in self.statesPresented:
            locDelayLog = statesPres[0,:]==(indDelayPres+1) # remember Matlab is 1-indexed
            if np.any(locDelayLog):
                locDelayStrtSt = np.where(locDelayLog)[0][0]
                targAppearSt = locDelayStrtSt - 1
                locDelayEndSt = locDelayStrtSt + 1
                startTmsPres.append(statesPres[1, targAppearSt])
                endTmsPres.append(statesPres[1, locDelayEndSt])
            else:
                startTmsPres.append(np.nan)
                endTmsPres.append(np.nan)
#        startEndIndsPres = [ ]
        
        return startTmsPres, endTmsPres
    
#%% Plotting functions and such
    def plotTuningCurves(self, subpltRows = 5, subpltCols = 2):
        if self.cosTuningCurveParams['tuningCurves'].size == 0:
            self.computeCosTuningCurves()
        
        rowColCombos = range(1, subpltRows*subpltCols+1)
        #pw = plotWindow()
        uniqueTargAngle, trialsPresented = np.unique(self.markerTargAngles, axis=0, return_inverse=True)
        angs = np.linspace(np.min(uniqueTargAngle), np.min(uniqueTargAngle)+2*np.pi)
        for pltNum in range(len(self.cosTuningCurveParams['tuningCurves'])):
            if not pltNum % (subpltRows*subpltCols):
                plotInd = 0
                pltFig = plt.figure()
                #pw.addPlot('plot set ' + str(pltNum), pltFig)
            
            plt.subplot(subpltRows, subpltCols, rowColCombos[plotInd])
            plt.plot(angs, self.cosTuningCurveParams['tuningCurves'][pltNum, :])
            plt.plot(uniqueTargAngle, self.cosTuningCurveParams['targAvgNumSpks'][0:, pltNum], 'o')
            
            if plotInd>=((subpltRows*subpltCols) - 2):
                plt.xlabel('angle (rad)')
            elif not (plotInd % 2):
                plt.ylabel('average firing rate (Hz)')

            plotInd += 1
        # for that final plot
        plt.xlabel('angle (rad)')
    
    def plotKinematics(self, groupByTarget=True):
        
        # startTmsPres, endTmsPres = self.computeDelayStartAndEnd()
        
        # tmMin = endTmsPres
        
        tmMin = self.timeOfState('Target Reach')
        tmMax = self.timeOfState('Target Hold')
        
        if self.kinematics is not None:
            if groupByTarget is True:
                angs, trlsPres = np.unique(self.markerTargAngles, return_inverse=True)
            else:
                angs = 0
                trlsPres = np.zeros_like(self.markerTargAngles)

            # mlab.figure()   
            plt.figure()
            # kinematPlot = plt.subplot(111, projection='3d')
            kinematPlot = plt.subplot(111)
            for idx, ang in enumerate(angs):
                for trlAng in np.where(trlsPres==idx)[0]:
                    kinsHere = self.kinematics[trlAng]['position'][0,0]
                    kinsHere = kinsHere-self.kinematicCenter
                    tmsHere = self.kinematics[trlAng]['time'][0,0].squeeze()
                    tmMinHere = tmMin[trlAng]
                    tmMaxHere = tmMax[trlAng]
                    ptsUse = np.logical_and(tmsHere>tmMinHere, tmsHere<tmMaxHere)
                    # mlab.plot3d(kinsHere[ptsUse,0], kinsHere[ptsUse,1], kinsHere[ptsUse,2], color = self.colorsetMayavi[idx])
                    kinematPlot.plot(kinsHere[ptsUse,0], kinsHere[ptsUse,1], color = self.colorset[idx])
                    
        else:
            print('no kinematics sorry!')