#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 16:10:29 2020

@author: emilio
"""
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from classes.BinnedSpikeSet import BinnedSpikeSet

# note timestep of simulations will be 0.1ms (Litwin-Kumar et al 2012)
# also note all times are in ms units
# numClusters=1 is the non-clustered network
class SpikingNetwork:
    def __init__(self, excitNeuronParams = {'number': 4000, 'biasRange':[1.1, 1.2]}, 
                 inhibNeuronParams = {'number':1000, 'biasRange':[1, 1.05]},
                 numClusters=1, clusterWeightParams = 
                     {'excitExcit': [0.024, 0.2], # 1st index, the actual weight, 2nd index, the probability
                      'eeInCluster': [0.0456, 0.4854, 0.1942], # 1 and 2 index, the probability (when clustered) for within (1) and b/t (2) clusters; note that b/t is paired with excitExcit value (1)
                      'excitInhib': [-0.045, .5], 
                      'inhibExcit': [0.014, .5], 
                      'inhibInhib': [-0.057, .5]},
                 tauMembrane = 15, refractoryPeriod = 5,
                 filterParams = {'tau1': 1, 'tau2Excit': 2, 'tau2Inhib': 3}, 
                 timestep=0.1, numTimesteps = 10000):
        self.numExcitNeurons = excitNeuronParams['number']
        self.excitNeuronBias = excitNeuronParams['biasRange']
        self.numInhibNeurons = inhibNeuronParams['number']
        self.inhibNeuronBias = inhibNeuronParams['biasRange']
        self.tauMembrane = tauMembrane
        self.refractoryPeriod = refractoryPeriod
        self.voltages = np.zeros((0,self.numExcitNeurons+self.numInhibNeurons, 1))
        self.spikes = np.zeros((0,self.numExcitNeurons+self.numInhibNeurons, 1), dtype='bool')
        
        self.numClusters = 1
        self.clustInd, self.inhibInd = self.partitionNeurons()
        self.clusterWeightParams = clusterWeightParams
        self.synapticWeights = self.generateSynapticWeights()
        self.bias = self.computeBiasMat()

        self.timestep = timestep
        self.numTimesteps = numTimesteps
        
        # filter properties
        self.filterTau1 = filterParams['tau1'] # ms, remember
        self.filterTau2Excit = filterParams['tau2Excit'] # ms
        self.filterTau2Inhib = filterParams['tau2Inhib'] # ms
        self.filters, self.filterArray = self.generateFilters()
        
    def partitionNeurons(self):
        excitNeuronInd = np.arange(self.numExcitNeurons)
        
        clustSize = excitNeuronInd.shape[0]/self.numClusters
        
        if clustSize % 1:
            raise("number of neurons need to be a whole multiple of number of clusters")
            
            
        clustInd = [np.arange(clustNum*clustSize, (clustNum+1)*clustSize, dtype='int') for clustNum in range(self.numClusters)]
        
        inhibInd = self.numExcitNeurons + np.arange(self.numInhibNeurons)    
        
        return clustInd, inhibInd
            
    def computeBiasMat(self):
        excitNeuronBias = np.diff(self.excitNeuronBias)*np.random.rand(self.numExcitNeurons, 1) + self.excitNeuronBias[0]
        inhibNeuronBias = np.diff(self.inhibNeuronBias)*np.random.rand(self.numInhibNeurons, 1) + self.inhibNeuronBias[0]
        
        biasExcitMat = excitNeuronBias*np.ones((self.numExcitNeurons,1))
        biasInhibMat = inhibNeuronBias*np.ones((self.numInhibNeurons,1))
        biasMat = np.concatenate((biasExcitMat, biasInhibMat), axis=0)
        biasMat[self.inhibInd, :] = inhibNeuronBias*np.ones_like(biasMat[self.inhibInd, :])
        return biasMat
        
    def voltageUpdate(self):
        tauMembrane = self.tauMembrane
        bias = self.bias
        prevVoltage = self.voltages[-1, :, None, -1]
        totalSynapticInputNow = self.totalSynapticInput()
        dv = 1/tauMembrane * (bias - prevVoltage) + totalSynapticInputNow
        
        newVoltage = prevVoltage+dv
        
        return newVoltage
    
    def excitVoltageUpdate(self, currVoltage):
        return self.voltageUpdate(currVoltage, self.tauMembrane, self.excitNeuronBias, self.totalSynapticInput)
    
    def inhibVoltageUpdate(self, currVoltage):
        return self.voltageUpdate(currVoltage, self.tauMembrane, self.inhibNeuronBias, self.totalSynapticInput)
    
    def simulationStep(self, spikesRecent):
        
        # newInhibVoltage = self.inhibVoltageUpdate(currInhibVoltage)
        # newExcitVoltage = self.excitVoltageUpdate(currExcitVoltage)
        spikesRecent = np.maximum(spikesRecent - 1, 0)
        np.max
        newVoltage = self.voltageUpdate()
        spikesNow = newVoltage>1
        # NOTE it's unclear what would happen if a neuron goes from voltage
        # zero to spiking--I don't think that's possible, though, but if it did
        # happen this prevents it
        spikesNow[spikesRecent>0] = False
        newVoltage[spikesNow] = 0
        newVoltage[spikesRecent > 0] = 0 # they have a refractory period
        spikesRecent[spikesNow] = self.refractoryPeriod/self.timestep
        
        
        
        return newVoltage, spikesNow, spikesRecent
        
    def totalSynapticInput(self):
        # do the excitatory neurons first
        
        filteredInput = np.zeros_like(self.filterArray)
        voltagesView = self.voltages[-1] #.transpose([1, 2, 0])
        spikesView = self.spikes[-1].astype('int')
        # A = 0
        with np.nditer([filteredInput, self.filterArray], ['refs_ok', 'multi_index'], [['readwrite'], ['readonly']]) as iterRef:
            #while not iterRef.finished:
            for filtOut, filtNum in iterRef:
                # filtOut[()] = self.filterInput(filtNum[()], np.flip(voltagesView[iterRef.multi_index]))[0]
                filtOut[()] = self.filterInput(filtNum[()], np.flip(spikesView[iterRef.multi_index]))[0]
                # A = A+1
                # print(str(A))
        
        totalSynapticInput =  self.synapticWeights @ filteredInput
        return totalSynapticInput
    
    def generateSynapticWeights(self): # still need to build in probabilities
        synapticWeights = np.empty((self.numExcitNeurons+self.numInhibNeurons,self.numExcitNeurons+self.numInhibNeurons))
        
        if self.numClusters == 1:
            clustNum = 0
            clustInds = self.clustInd[clustNum]
            connectionMask = np.random.rand(clustInds.shape[0],clustInds.shape[0])<self.clusterWeightParams['excitExcit'][1]
            synapticWeights[np.ix_(self.clustInd[clustNum],self.clustInd[clustNum])] = connectionMask*self.clusterWeightParams['excitExcit'][0]
        else:
            for clustNum in range(self.numClusters):
                clustOneInds = self.clustInd[clustNum]
                for clustNumOther in range(self.numClusters):
                    clustTwoInds = self.clustInd[clustNumOther]
                    if clustNumOther==clustNum:
                        connectionMask = np.random.rand(clustOneInds.shape[0],clustTwoInds.shape[0])<self.clusterWeightParams['eeInCluster'][1]
                        synapticWeights[np.ix_(clustOneInds,clustTwoInds)] = connectionMask*self.clusterWeightParams['eeInCluster'][0]
                    else:
                        # note that the cluster mask probability between clusters lies with the eeInCluster when there's more than one cluster...
                        connectionMask = np.random.rand(clustOneInds.shape[0],clustTwoInds.shape[0])<self.clusterWeightParams['eeInCluster'][2]
                        synapticWeights[np.ix_(clustOneInds,clustTwoInds)] = connectionMask*self.clusterWeightParams['excitExcit'][0]
                    
        connectionMask = np.random.rand(self.inhibInd.shape[0],synapticWeights.shape[1])<self.clusterWeightParams['inhibExcit'][1]
        synapticWeights[self.inhibInd, :] = connectionMask*self.clusterWeightParams['inhibExcit'][0]
        connectionMask = np.random.rand(synapticWeights.shape[0],self.inhibInd.shape[0])<self.clusterWeightParams['excitInhib'][1]
        synapticWeights[:, self.inhibInd] = connectionMask*self.clusterWeightParams['excitInhib'][0]
        connectionMask = np.random.rand(self.inhibInd.shape[0],self.inhibInd.shape[0])<self.clusterWeightParams['inhibInhib'][1]
        synapticWeights[np.ix_(self.inhibInd, self.inhibInd)] = connectionMask*self.clusterWeightParams['inhibInhib'][0]
        
        return synapticWeights
            
    
    def generateFilters(self):
        maxFilterTau = np.max([self.filterTau1, self.filterTau2Excit, self.filterTau2Inhib])
        timestepsBack = np.minimum(300, 7*maxFilterTau/self.timestep)
       
        
        times = self.timestep*np.arange(timestepsBack)
        
        # excitatory filter
        excitFilter =  1/(self.filterTau2Excit-self.filterTau1) * (np.exp(-times/self.filterTau2Excit) - np.exp(-times/self.filterTau1))
        inhibFilter =  1/(self.filterTau2Inhib-self.filterTau1) * (np.exp(-times/self.filterTau2Inhib) - np.exp(-times/self.filterTau1))
        

        # zero index will be excit filter, 1 will be inhib filter
        filterArray = np.zeros((self.numExcitNeurons+self.numInhibNeurons, 1), dtype='int')
        filterArray[self.inhibInd, :] = np.ones_like(filterArray[self.inhibInd, :], dtype='int')
        # filterArray[:, self.inhibInd] = np.ones_like(filterArray[:, self.inhibInd], dtype='int')

        
        return (excitFilter, inhibFilter), filterArray

    def filterInput(self, filterFunctionNum, timepointData):
        
        return np.convolve(self.filters[filterFunctionNum], timepointData, mode='valid')
    
    def runSimulation(self, numRuns = 1):
        
        for run in range(numRuns):
            self.voltages = np.concatenate((self.voltages, np.zeros((1,self.numExcitNeurons+self.numInhibNeurons, 1))), axis=0)
            self.spikes = np.concatenate((self.spikes, np.zeros((1,self.numExcitNeurons+self.numInhibNeurons, 1))), axis=0)
            
            spikesRecent = np.zeros_like(self.voltages[0], dtype='int')
            tstep = 1
            while tstep<self.numTimesteps:
                print(str(tstep))
                newVoltage, newSpikes, spikesRecent = self.simulationStep(spikesRecent)
                self.voltages = np.concatenate((self.voltages, newVoltage[None, :, :]), axis=2)
                self.spikes = np.concatenate((self.spikes, newSpikes[None, :, :]), axis=2)
                tstep = tstep+1
            
        outSpikes = self.spikes.view(BinnedSpikeSet)
        outSpikes.binSize = self.timestep
        outSpikes.start = 0
        outSpikes.end = tstep*self.timestep
        return outSpikes

class GPFAGenerator:
    def __init__(self, numGPs = 10, maxTau = 100, gpMeans = 0,
                 startT = 0, endT = 1000, binSize = 25,
                 # Do note that since these times are all in ms, the default 0.1 neuronMean is actually 0.1 sp/ms = 100sp/s!
                 # The neuronNoiseStdScale tells us what the max percent (from a uniform distribution), the neuron mean the noise will b
                 numNeurons = 100, neuronMeansFrMax = 0.1, neuronNoiseStdScaleMax = 1, sharedVar = 0.2, randSeed = None):
        
        if randSeed is not None:
            initSt = np.random.get_state()
            np.random.seed(seed=randSeed)

        self.time = np.arange(startT, endT, binSize) + binSize/2
        self.start = np.array([startT])
        self.end = np.array([endT])
        self.binSize = binSize
        self.numGPs = numGPs
        
        
        self.gpTaus = np.random.uniform(high=maxTau, size = numGPs)
        self.gpMeans = gpMeans
        self.gpCovars = self.computeAllLowDimCovariances()
        
        neuronMeansCntMax = neuronMeansFrMax * binSize
        # self.neuronMeans = np.random.uniform(high=neuronMeansCntMax, size=numNeurons)
        self.neuronMeans = np.ones((numNeurons,))*neuronMeansCntMax
        self.neuronMeansFr = np.ones((numNeurons,)) * neuronMeansFrMax
        self.neuronNoise = np.random.uniform(high=neuronNoiseStdScaleMax,size=numNeurons)*self.neuronMeans
        
        unorthoDirsInit = np.random.uniform(low=-1,high=1,size=(numNeurons, numGPs))
        unorthoDirsInitNorm = unorthoDirsInit / np.sqrt(np.sum(unorthoDirsInit**2, axis=0))
        totVarInDirs = sharedVar*self.neuronNoise.sum()/(1-sharedVar)
        gpVarSpread = 1/np.arange(1, numGPs+1)
        gpVarSpread = gpVarSpread/gpVarSpread.sum()
        varPerOrthoDir = totVarInDirs * gpVarSpread # currently a uniform variance per direction
        orthoDirs, sv, _ = np.linalg.svd(unorthoDirsInit)
        orthoDirs = orthoDirs[:,:numGPs]
        orthoDirsScaled = orthoDirs * np.sqrt(varPerOrthoDir) # sqrt b/c variance comes from inner product of direction
        varInUnorthoDirs = np.sum((orthoDirsScaled.T @ unorthoDirsInitNorm) ** 2,axis=0)
        self.unorthoDirs = orthoDirsScaled # for now... ; np.sqrt(varInUnorthoDirs) * unorthoDirsInitNorm # self.unorthoDirs = np.random.multivariate_normal(np.zeros(numGPs),np.eye(numGPs), numGPs)

        if randSeed is not None:
            np.random.set_state(initSt)
        
        
    def computeAllLowDimCovariances(self):
        gpCovars = np.empty((self.gpTaus.shape[0],self.time.shape[0],self.time.shape[0]))
        for ind,tau in enumerate(self.gpTaus):
            gpCovars[ind, :, :] = self.computeLowDimCovariance(self.time, tau)
            # gpCovars[ind, :, :] = self.computeLowDimCovariance(self.time, self.gpTaus[0])
            
        return gpCovars
    
    def computeLowDimCovariance(self, time, tau, noiseVar = 1e-3):
        signalVar = 1-noiseVar # note this *could* be a matrix, but by default it's a single value
        
        timeDiffMat = np.subtract.outer(time, time)
        
        expTerm = -np.power(timeDiffMat, 2)/2/np.power(tau, 2)
        
        fullCovar = signalVar * np.exp(expTerm) + noiseVar*np.eye(expTerm.shape[0])
        
        return fullCovar
    
    # note all times here are in ms
    def computeGaussianProcess(self, numGPs = 10, means = 0):
        
        # gonna start with a uniform distribution of taus here...
        
        gpTraj = np.empty((numGPs, self.time.shape[0]))
        for ind, tau in enumerate(self.gpTaus):
            # print("covar computed for gp " + str(ind))
            means = np.zeros((self.time.shape[0])) + means
            gpTraj[ind, :] = np.random.multivariate_normal(means, self.gpCovars[ind])
            
        return gpTraj
    
    def computeNeuralTraj(self, gpTraj, underlyingMean=None):
        
        C = self.unorthoDirs # reflect naming of loading matrix in GPFA paper
        numNeurons = C.shape[0]

        highDimProjBase = C @ gpTraj
        
        neuralTrajMean = highDimProjBase+self.neuronMeans[:,None]
        # if underlyingMean is not None:
        #     neuralTrajMean += underlyingMean
        neuralTrajNoise = np.diag(self.neuronNoise)
        
        neuralTrajObserved = np.empty((numNeurons,self.time.shape[0]))
        
        timePeriod = self.end - self.start
        for ind, trajNeur in enumerate(neuralTrajMean):
            # neuralTrajObserved[ind, :] = np.random.normal(trajNeur, np.sqrt(neuralTrajNoise[ind,ind]))
            # neuralTrajObserved[ind, :] = np.random.normal(trajNeur, np.sqrt(neuralTrajNoise[ind,ind]))
            numSpPerNeuron = np.random.poisson(self.neuronMeansFr[ind]*timePeriod)
            spTmsPerNeuron = [np.random.uniform(low=self.start, high=self.end, size = numSp) for numSp in numSpPerNeuron]
            numSpPerBinPerNeuron = [np.bincount(np.digitize(spTms, np.arange(self.start, self.end+self.binSize/2, self.binSize))-1, minlength=np.arange(self.start, self.end+self.binSize/2, self.binSize).shape[0]-1) for spTms in spTmsPerNeuron]
            numSpPerBinPerNeuron = np.stack(numSpPerBinPerNeuron)
            neuralTrajObserved[ind, :] =  highDimProjBase[ind] + numSpPerBinPerNeuron # np.maximum(highDimProjBase[ind] + numSpPerBinPerNeuron, 0)
        # for ind, trajTP in enumerate(neuralTrajMean.T):
        #     # print("traj computed for time point " + str(ind))
        #     neuralTrajObserved[:, ind] = np.random.multivariate_normal(trajTP, neuralTrajNoise)
        #     # neuralTrajObserved[:, ind] = trajTP + np.random.poisson(self.neuronMeans)
        #     # breakpoint()
        #     # neuralTrajObserved[:, ind] = np.random.poisson(np.maximum(trajTP + self.neuronMeans, 0))
            
        return neuralTrajObserved

    def runSimulation(self, numSimulations = 1, underlyingMean = None, newGpPerSim = True, randSeed = None):
        
        if randSeed is not None:
            initSt = np.random.get_state()
            np.random.seed(seed=randSeed)

        numGPs = self.gpTaus.shape[0]
        numNeurons = self.neuronMeans.shape[0]
        
        self.gpTraj = np.empty((numSimulations, numGPs, self.time.shape[0]))
        self.neuralTraj = np.empty((numSimulations, numNeurons, self.time.shape[0]))
        
        if not newGpPerSim:
            singleGp = self.computeGaussianProcess(numGPs = numGPs, means = self.gpMeans)

        for simNum in range(numSimulations):
            if newGpPerSim:
                self.gpTraj[simNum, :, :] = self.computeGaussianProcess(numGPs = numGPs, means = self.gpMeans)
            else:
                self.gpTraj[simNum, :, :] = singleGp
            self.neuralTraj[simNum, :, :] = self.computeNeuralTraj(self.gpTraj[simNum], underlyingMean)

            if not (simNum+1) % 10 and simNum>0:
                print("done with simulation " + str(simNum+1))
            
        outNeuralTraj = self.neuralTraj.view(BinnedSpikeSet)
        outNeuralTraj.binSize = self.binSize
        outNeuralTraj.start = np.repeat(self.start, numSimulations)
        outNeuralTraj.end = np.repeat(self.end, numSimulations)
        outNeuralTraj.units = 'count'
        outNeuralTraj.labels['stimulusMainLabel'] = np.ones((outNeuralTraj.shape[0],1))
        outNeuralTraj.alignmentBins = np.hstack([np.zeros((outNeuralTraj.shape[0],1)),outNeuralTraj.shape[2]*np.ones((outNeuralTraj.shape[0],1))])

        if randSeed is not None:
            np.random.set_state(initSt)
            
        return outNeuralTraj

    def drawGps(self):
        if self.numGPs > 1:
            numRows = 2
            numCols = int(np.ceil(self.numGPs/numRows))
            figH, axsH = plt.subplots(numRows, numCols)
            axsH = axsH.flatten()
        else:
            numRows = 1
            numCols = 1
            figH, axsH = plt.subplots(numRows, numCols)
            axsH = [axsH]

        tms = np.arange(self.start, self.end, self.binSize)
        for gpT, ax in enumerate(axsH):
            if gpT < self.gpTraj.shape[1]:
                ax.plot(tms, self.gpTraj[:,gpT, :].T)
                ax.set_title('gp number {}'.format(gpT))
                ax.set_xlabel('time (ms)')
                ax.set_ylabel('spike count')