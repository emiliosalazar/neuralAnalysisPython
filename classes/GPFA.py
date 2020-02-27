#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 15:26:50 2020

@author: emilio
"""
from classes.MatlabConversion import MatlabConversion as mc
import numpy as np
from classes.BinnedSpikeSet import BinnedSpikeSet

class GPFA:
    def __init__(self, binnedSpikes):
        allTrlTogether = np.concatenate(binnedSpikes, axis=1)[None, :, :].view(BinnedSpikeSet)
        allTrlTogether.binSize = binnedSpikes[0].binSize
        _, chansKeep = allTrlTogether.channelsAboveThresholdFiringRate(firingRateThresh=5)
        # chansKeepCoinc = allTrlTogether.removeCoincidentSpikes()
        # chansKeep = np.logical_and(chansKeepThresh, chansKeepCoinc)
        self.binnedSpikes = [bnSp[chansKeep, :] for bnSp in binnedSpikes]
        self.gpfaSeqDict = self.binSpikesToGpfaInputDict()
        self.trainInds = None
        self.testInds = None
        
        if type(binnedSpikes) is list:
            self.binSize = binnedSpikes[0].binSize
        else:
            self.binSize = binnedSpikes.binSize
    
    def binSpikesToGpfaInputDict(self, sqrtSpks = True):
        binnedSpikes = self.binnedSpikes
        
        gpfaSeqDict = []
        for trlNum, trl in enumerate(binnedSpikes):
            gpfaSeqDict.append({'trialId':trlNum,
                            'T': (trl.shape[1])*1.0,
                            'y': mc.convertNumpyArrayToMatlab(trl)})
        
        return gpfaSeqDict
    
    def trainAndTestSeq(self, numFolds = 4):
        seqAll = self.gpfaSeqDict
        numTrls = len(seqAll)
        
        initSt = np.random.get_state()
        np.random.seed(seed=0)
        randomizedIndOrder = np.random.permutation(np.arange(numTrls))
        self.testInds = randomizedIndOrder[:round((1/numFolds)*numTrls)]
        self.trainInds = randomizedIndOrder[round((1/numFolds)*numTrls):]
        
        seqTrain = [seqAll[idx] for idx in self.trainInds]
        seqTest = [seqAll[idx] for idx in self.testInds]
        
        np.random.set_state(initSt)
        
        return seqTrain, seqTest # for splitting binned spikes into train and test sets
        
    def runGpfaInMatlab(self, fname="~/Documents/BatistaLabData/analysesOutput/gpfaTest", crossvalidate = False, xDim = 8, eng=None, segLength = None):
        
        if eng is None:
            from matlab import engine 
            eng = engine.start_matlab()
        
        if not crossvalidate:
            seqTrain = self.gpfaSeqDict
            seqTest = []
            seqTrainStr = "[seqTrain{:}]"
            seqTestStr = "subsref([seqTrain{:}], struct('type', '()', 'subs', {{[]}}))"
        else:
            # NOTE: INCOMPLETE!
            seqTrain, seqTest = self.trainAndTestSeq()
            seqTrainStr = "[seqTrain{:}]"
            seqTestStr = "[seqTest{:}]"
            
        if segLength is None:
            segLength = min([sq['T'] for sq in self.gpfaSeqDict])
            
        binWidth = self.binSize
        eng.workspace['seqTrain'] = seqTrain
        eng.workspace['seqTest'] = seqTest
        eng.workspace['binWidth'] = binWidth*1.0
        eng.workspace['xDim'] = xDim*1.0
        eng.workspace['fname'] = fname
        eng.workspace['segLength'] = segLength*1.0
        
        
        
        eng.eval("gpfaEngine(" + seqTrainStr + ", " + seqTestStr + ", fname, 'xDim', xDim, 'binWidth', binWidth, 'segLength', segLength)", nargout=0)

        eng.evalc("results = load('"+fname+"')")
        eng.evalc("results.method='gpfa'") # this is needed for the next step...
        eng.evalc("[estParams, seqTrain, seqTest] = postprocess(results)")
        
        estParams = mc.convertMatlabDictToNumpy(eng.eval('estParams'))
        
        seqTrainNew = [mc.convertMatlabDictToNumpy(eng.eval("seqTrain(" + str(seqNum+1) + ")")) for seqNum in range(len(seqTrain))]
        seqTestNew = [mc.convertMatlabDictToNumpy(eng.eval("seqTest(" + str(seqNum+1) + ")")) for seqNum in range(len(seqTest))]
                
        return estParams, seqTrainNew, seqTestNew