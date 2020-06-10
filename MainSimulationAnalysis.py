#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 09:50:31 2020

@author: emilio
"""


#%% GPFA Simulations
from matplotlib import pyplot as plt
from pathlib import Path
import numpy as np
import multiprocessing as mp

from classes.Simulations import GPFAGenerator
from methods.GeneralMethods import loadDefaultParams
from methods.BinnedSpikeSetListMethods import gpfaComputation

defaultParams = loadDefaultParams(defParamBase = ".")
dataPath = defaultParams['dataPath'] / Path('simulations')


# plotting info
#plotInfo = {}
#figSim = plt.figure()
#figSim.suptitle('dimensionality vs. GPFA log likelihood')
#axs = figSim.subplots(nrows=2, ncols=1, squeeze=False)
#
#axScore = axs[0, :].flat[0]
#axDim = axs[1,:].flat[0]
#plotInfo['axScore'] = axScore
#plotInfo['axDim'] = axDim



if __name__ == '__main__':
#    mp.set_start_method('spawn')
    totTime = 600
    numNeurons = 90
    numSimulations = 1000
    binSize = 25
    maxTau = 100
    neuronMeansFrMax = 0.05 # this is 0.05 sp/ms = 50 sp/s = 50 Hz

    gpfaSim = GPFAGenerator(numGPs = 10, endT = totTime, binSize = binSize,
                     numNeurons = numNeurons, neuronMeansFrMax = neuronMeansFrMax, maxTau = maxTau) 
    outNeurTraj = gpfaSim.runSimulation(numSimulations=numSimulations)
    breakpoint()

    xDimTest = [8,9,10,11,12]
    descriptions = ['gpfa sim']
    paths = [dataPath]
    signalDescriptor = 'simNum%dNeurs%dTmLen%dBinSize%d' % (numSimulations, numNeurons, totTime, binSize)
    timeBeforeAndAfterStart=None
    timeBeforeAndAfterEnd=(-250,0)
    numStimulusConditions = None
    combineConditions = False
    baselineSubtract = False
    crossvalidateNumFolds = 4
    firingRateThresh = -np.inf
    plotOutput = True
    forceNewGpfaRun = True

    dims, dimsLL, gpfaDimOut, gpfaTestInds, gpfaTrainInds = gpfaComputation(
        [outNeurTraj], descriptions, paths, signalDescriptor = signalDescriptor,
                    # [listBSS[-1]], [descriptions[-1]], [paths[-1]],
                              timeBeforeAndAfterStart = timeBeforeAndAfterStart, timeBeforeAndAfterEnd = timeBeforeAndAfterEnd,
                              balanceDirs = True, numStimulusConditions = numStimulusConditions, combineConditions=combineConditions, baselineSubtract = baselineSubtract, sqrtSpikes = False, forceNewGpfaRun = forceNewGpfaRun,
                              crossvalidateNumFolds = crossvalidateNumFolds, xDimTest = xDimTest, firingRateThresh=firingRateThresh, plotOutput = plotOutput)



#    eng = None
#    numDimsSubst, numDimsLLSubst, gpfaOutDimSubst, gpfaTestIndsOutSubst, gpfaTrainIndsOutSubst = \
#        outNeurTraj.gpfa(eng, "GPFA Sim", dataPath, plotInfo=plotInfo,
#            baselineSubtract=False, # aleady mean centered
#            firingRateThresh=-np.inf, # let's ignore FR thresh
#            xDimTest = xDimTest, 
#            timeBeforeAndAfterStart=None, 
#            timeBeforeAndAfterEnd=(-250,0))

from methods.GeneralMethods import saveFiguresToPdf
saveFiguresToPdf(pdfname=signalDescriptor)
breakpoint()
