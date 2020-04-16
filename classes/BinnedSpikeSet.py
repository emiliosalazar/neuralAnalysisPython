#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 15:53:35 2020

@author: emilio
"""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from classes.FactorAnalysis import FactorAnalysis
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # for 3d plotting
#from mayavi import mlab

from copy import copy

import pdb

class BinnedSpikeSet(np.ndarray):
    dimDescription = {'rows':'trials',
                      'cols':'channel',
                      'depth':'time'}
    
    
    
    colorset = np.array([[127,201,127],[190,174,212],[253,192,134],[200,200,153],
                         [56,108,176],[240,2,127],[191,91,23],[102,102,102]])/255
    colorsetMayavi = [tuple(col) for col in colorset]
    
    # this has to do with how it's suggested that ndarray be subclassed...
    # https://docs.scipy.org/doc/numpy/user/basics.subclassing.html
    def __new__(cls, input_array, start=None, end=None, binSize=None, labels = {}):
        # Create the ndarray instance of our type, given the usual
        # ndarray input arguments.  This will call the standard
        # ndarray constructor, but return an object of our type.
        # It also triggers a call to BinnedSpikeSet.__array_finalize__
        obj = np.asarray(input_array).view(cls)
        # obj = super(BinnedSpikeSet, subtype).__new__(subtype, shape, dtype,
        #                                         buffer, offset, strides,
        #                                         order)
        
        obj.binSize = binSize # in milliseconds
        obj.start = start
        obj.end = end
        obj.labels = labels # this is meant to hold various types of labels, especially for trials
                # and also to be expansible if new/overlapping labels present
        obj._new_label_index = []
                
        # Finally, we must return the newly created object:
        return obj
    
    # this has to do with how it's suggested that ndarray be subclassed...
    # https://docs.scipy.org/doc/numpy/user/basics.subclassing.html
    def __array_finalize__(self, obj):
        # ``self`` is a new object resulting from
        # ndarray.__new__(BinnedSpikeSet, ...), therefore it only has
        # attributes that the ndarray.__new__ constructor gave it -
        # i.e. those of a standard ndarray.
        #
        # We could have got to the ndarray.__new__ call in 3 ways:
        # From an explicit constructor - e.g. BinnedSpikeSet():
        #    obj is None
        #    (we're in the middle of the BinnedSpikeSet.__new__
        #    constructor, and self.binSize will be set when we return to
        #    InfoArray.__new__)
        if obj is None: return
        # From view casting - e.g arr.view(BinnedSpikeSet):
        #    obj is arr
        #    (type(obj) can be BinnedSpikeSet)
        # From new-from-template - e.g spikeSet[:3]
        #    type(obj) is BinnedSpikeSet
        #
        # Note that it is here, rather than in the __new__ method,
        # that we set the default value for 'binSize', because this
        # method sees all creation of default objects - with the
        # BinnedSpikeSet.__new__ constructor, but also with
        # arr.view(BinnedSpikeSet).
        self.binSize = getattr(obj, 'binSize', None) # in milliseconds
        self.start = getattr(obj, 'start', None)
        self.end = getattr(obj, 'end', None)
        self.labels = copy(getattr(obj, 'labels', {})) # this is meant to hold various types of labels, especially for trials
        try:
            # print('size = ' + str(len(obj._new_label_index)))
            # print('first ' + str(obj._new_label_index))
            # print('last ' + str(obj._new_label_index) + '\n')
            for key, val in self.labels.items():
                self.labels[key] = val[obj._new_label_index]
                
            self._new_label_index = getattr(obj, '_new_label_index', [])
            setattr(obj, '_new_label_index', [])
        except Exception as e:
            pass
                # and also to be expansible if new/overlapping labels present
        self._new_label_index = getattr(self, '_new_label_index', [])
        # We do not need to return anything
        
    # this little tidbit of info inspired by
    # https://stackoverflow.com/questions/16343034/numpy-subclass-attribute-slicing
    # But for the moment... seems a bit too complex to want to deal with...
    def __getitem__(self,item):
        try:
            # print(type(self))
            if isinstance(item, (slice, int, np.ndarray)):
                # print('test')
                self._new_label_index = item
                #setattr(self,'_new_label_index', item)
            elif isinstance(item[0], (slice, int, np.ndarray)):
                # print('test2')
                # print(str(item))
                # print(str(self.shape))
                # print(str(self.labels))
                self._new_label_index = item[0]
                # setattr(self,'_new_label_index', item[0])
            # else:
                # print('test3')
                # print(str(self.shape))
                # self._new_label_index = item[0]
                # print('hello')
        except: 
            pass
        # print("0")
        return super().__getitem__(item)
        
    def copy(self):
        out = BinnedSpikeSet(self.view(np.ndarray).copy(), start=self.start, end=self.end, binSize=self.binSize, labels=self.labels.copy())
        
        # # copy back the labels
        # for labelName, labels in self.labels.items():
        #     out.labels[labelName] = labels
        
        return out
#%% general methods
        
    def timeAverage(self):
        return np.average(self, axis=2)
    
    def timeStd(self):
        return np.std(self, axis=2, ddof=1) # ddof=1 makes this sample standard deviation #np.sqrt(self.timeAverage()) #
    
    def timeSpikeCount(self):
        return np.sum(self, axis=2)
    
    def trialAverage(self):
        if self.size > 0:
            return np.average(self, axis=0)
        else:
            return None
        
    def trialStd(self):
        if self.size > 0:
            return np.std(self, axis=0, ddof=1) #np.sqrt(self.trialAverage()) #
        else:
            return None
        
    def trialSpikeCount(self):
        if self.size > 0:
            return np.sum(self, axis=0)
        else:
            return None
        
    # timestamps are computed by centering on bin
    def computeTimestamps(self):
        trlTmStmp = []
        for trl in self:
            trlTmStmp.append([self.binSize*(np.where(trlChan)[0][None,:]+0.5) for trlChan in trl])
            
        return trlTmStmp
        
    def balancedTrialInds(self, labels):
        unq, unqCnts = np.unique(labels, return_counts=True)
        minCnt = min(unqCnts)
        idxUse = []
        
        for idx, cnt in enumerate(unqCnts):
            labHere = unq[idx]
            inds, _ = np.nonzero(labels==labHere)
            if cnt>minCnt:
                # grab a random subset of labelled data of the correct balanced size
                idxUse.append(np.random.permutation(inds)[:minCnt])
            else:
                idxUse.append(inds)
                
        idxUseAll = np.stack(idxUse, axis=0).flatten()
        idxUseAll.sort()
        
        return idxUseAll
    
    def baselineSubtract(self, labels=None):
        if labels is None:
            print("baseline subtracting time trace with no groups")
            overallBaseline = self.trialAverage()
            spikesUse = self.copy()
            spikesUse = spikesUse[:,:,:] - overallBaseline[:,:] # this indexing is important for the moment to keep the labels field correct...
        else:
            unq, unqCnts = np.unique(labels, return_counts=True)
            spikesUse = self.copy()
            for lbl in unq:
                spikesUse[labels.squeeze()==lbl, :, :] = self[labels.squeeze()==lbl, :, :] - self[labels.squeeze()==lbl, :, :].trialAverage()

        return spikesUse 

    # group by all unique values in label and/or group by explicit values in labelExtract
    def groupByLabel(self, labels, labelExtract=None):
        unq, unqCnts = np.unique(labels, return_counts=True)
        if labelExtract is None:
            groupedSpikes = [self[labels.squeeze()==lbl, :, :] for lbl in unq]
        else:
            groupedSpikes = [self[labels.squeeze()==lbl, :, :] for lbl in labelExtract]
            unq = labelExtract
            
        
        return groupedSpikes, unq  

    def avgFiringRateByChannel(self):
        avgFiringChan = self.timeAverage().trialAverage()
        
        return avgFiringChan

    def channelsAboveThresholdFiringRate(self, firingRateThresh=1): #low firing thresh in Hz
        avgFiringChan = self.timeAverage().trialAverage() 
        
        highFiringChannels = avgFiringChan>firingRateThresh
        
        return self[:,highFiringChannels,:], np.where(highFiringChannels)[0]
    
    # coincidenceTime is how close spikes need to be, in ms, to be considered coincicdent, 
    # coincidentThresh in percent/100 (i.e. 20% = 0.2)
    def removeCoincidentSpikes(self, coincidenceTime=1, coincidentThresh=0.2): 
        spikeCountOverallPerChan = self.timeSpikeCount().trialSpikeCount()
        
        timestampSpikes = self.computeTimestamps()
        
        coincCnt = np.zeros((spikeCountOverallPerChan.shape[0],  spikeCountOverallPerChan.shape[0]))
        
        for trlTmstmp in timestampSpikes:
            for idx1, ch1 in enumerate(trlTmstmp):
                print(str(idx1))
                for idx2, ch2 in enumerate(trlTmstmp):
                    if idx2>idx1:
                        ch1ch2TDiff = ch1 - ch2.T
                        coincCnt[idx1, idx2] = coincCnt[idx1, idx2]+np.where(abs(ch1ch2TDiff)<coincidenceTime)[0].shape[0]
                        coincCnt[idx2, idx1] = coincCnt[idx1, idx2]
                    
        coincProp = coincCnt/spikeCountOverallPerChan
        
        chansKeep = np.unique(np.where(coincProp<coincidentThresh)[1]) # every column is the division with that channel's spike count
        # badChans = np.unique(np.where(coincProp>=coincidentThresh)[1]) # every column is the division with channel's spike count
        
        self = self[:,chansKeep,:]
        
        # in place change, though returns channels with too much coincidence if desired...
        return chansKeep
                    
    def increaseBinSize(self, newBinSize):
        if (newBinSize/self.binSize) % 1:
            raise Exception('BinnedSpikeSet:BadBinSize', 'New bin size must be whole multiple of previous bin size!')
            return None
        
        bins = np.arange(self.start, self.end, newBinSize)
        spikeTimestamps = self.computeTimestamps()
        
        spikeDatBinnedList = [[[counts for counts in np.histogram(spikeTimestamps[trlNum][chanNum], bins=bins, weights=chanSpks[np.where(chanSpks)][None,:])][0] for chanNum, chanSpks in enumerate(trl)] for trlNum,trl in enumerate(self)]
        newBinnedSpikeSet = np.stack(spikeDatBinnedList).view(BinnedSpikeSet)
        
        newBinnedSpikeSet.binSize = newBinSize
        newBinnedSpikeSet.start = self.start
        newBinnedSpikeSet.end = self.end
        
        return newBinnedSpikeSet
        
#%% Analysis methods
    def pca(self, baselineSubtract = False, labels = None, crossvalid=True, n_components = None, plot = False):
        if n_components is None:
            if self.shape[0] < self.shape[1]:
                print("PCA not doable with so few trials")
                return
            n_components =  self.shape[1]
            
        if baselineSubtract and labels is not None:
            chanSub = self.baselineSubtract(labels=labels)
        else:
            chanSub = self
            
        chanFirst = chanSub.swapaxes(0, 1) # channels are now main axis
        chanFlat = chanFirst.reshape((self.shape[1], -1))
        dataUse = chanFlat.T
            
        pcaModel = PCA(n_components = n_components)
        pcaModel.fit(dataUse)
        
        if plot:
            
            xDimRed = pcaModel.transform(chanFlat.T)
            
            if labels is not None:
                uniqueTargAngle, trialsPresented = np.unique(labels, axis=0, return_inverse=True)
            else:
                uniqueTargAngles = 0
                trialsPresented = np.ones(self.shape[0]) # ones array the size of number of trials--will repeat later
                
            # mlab.figure()
            plt.figure()
            ax01 = plt.subplot(221)
            ax02 = plt.subplot(222)
            ax12 = plt.subplot(223)
            ax3D = plt.subplot(224,projection='3d')
            trialNumbersAll = np.arange(trialsPresented.shape[0])
            trialNumbersAllExp = np.repeat(trialNumbersAll, self.shape[2])
            for idx, ang in enumerate(uniqueTargAngle):
                trlsUseAll = trialsPresented == idx
                trlNums = np.where(trlsUseAll)[0]
                # trlNumsAllExp = np.repeat(trlNums, self.shape[2])
                # trlsUse = np.repeat(trlsUse, self.shape[2])
                for trlNum in trlNums:
                    trlsUse = trialNumbersAllExp == trlNum
                # colorUse = self.colorsetMayavi[idx]
                    colorUse = self.colorset[idx]
                    # mlab.points3d(xDimRed[trlsUse, 0],xDimRed[trlsUse, 1],xDimRed[trlsUse, 2], color = colorUse, scale_factor=10)
                    if np.sum(trlsUse)>1:
                        ax01.plot(xDimRed[trlsUse, 0],xDimRed[trlsUse, 1], '-', color = colorUse)
                        ax02.plot(xDimRed[trlsUse, 0],xDimRed[trlsUse, 2], '-', color = colorUse)
                        ax12.plot(xDimRed[trlsUse, 1],xDimRed[trlsUse, 2], '-', color = colorUse)
                        ax3D.plot(xDimRed[trlsUse, 0],xDimRed[trlsUse, 1],xDimRed[trlsUse, 2], '-', color = colorUse)
                    else:
                        ax01.plot(xDimRed[trlsUse, 0],xDimRed[trlsUse, 1], '.', color = colorUse)
                        ax02.plot(xDimRed[trlsUse, 0],xDimRed[trlsUse, 2], '.', color = colorUse)
                        ax12.plot(xDimRed[trlsUse, 1],xDimRed[trlsUse, 2], '.', color = colorUse)
                        ax3D.plot(xDimRed[trlsUse, 0],xDimRed[trlsUse, 1],xDimRed[trlsUse, 2], '.', color = colorUse)
        
        return pcaModel
    
    def lda(self, labels, n_components = None, baselineSubtract = False, plot = False):
        uniqueLabel, labelPresented = np.unique(labels, axis=0, return_inverse=True)
        if n_components is None:
            if self.shape[0] < self.shape[1] or self.size == 0:
                print("LDA not doable with so few trials")
                return
            n_components =  uniqueLabel.shape[0]-1
            
        if baselineSubtract and labels is not None:
            chanSub = self.baselineSubtract(labels=labels)
        else:
            chanSub = self
            
        chanFirst = chanSub.swapaxes(0, 1) # channels are now main axis, for sampling in reshape below
        chanFlat = chanFirst.reshape((self.shape[1], -1)) # chan are now rows, time is cols
        dataUse = chanFlat.T
        
        # using the eigen solve that's described as optmizing between class scatter 
        # with within class scatter... I think this is the one I expect. I'm actually
        # not sure what the 'svd' default solver does... or whether they output
        # different results, but one certain difference is that somehow 'svd' doesn't
        # need to compute the data's covariance matrix... 
        # to be clear they seem to output visually similar answers...
        ldaModel = LDA(solver='eigen', n_components = n_components)
        labelPresented = np.repeat(labelPresented, self.shape[2])
        ldaModel.fit(dataUse, labelPresented)
        
        if plot:
            
            xDimRed = ldaModel.transform(chanFlat.T)
                
            # mlab.figure()
            plt.figure()
            if xDimRed.shape[1] >= 3:
                ax01 = plt.subplot(221)
                ax02 = plt.subplot(222)
                ax12 = plt.subplot(223)
                ax3D = plt.subplot(224,projection='3d')
                for idx, ang in enumerate(uniqueLabel):
                    trlsUse = labelPresented == idx
                    # colorUse = self.colorsetMayavi[idx]
                    colorUse = self.colorset[idx]
                    # mlab.points3d(xDimRed[trlsUse, 0],xDimRed[trlsUse, 1],xDimRed[trlsUse, 2], color = colorUse, scale_factor=10)
                    ax01.plot(xDimRed[trlsUse, 0],xDimRed[trlsUse, 1], '.', color = colorUse)
                    ax02.plot(xDimRed[trlsUse, 0],xDimRed[trlsUse, 2], '.', color = colorUse)
                    ax12.plot(xDimRed[trlsUse, 1],xDimRed[trlsUse, 2], '.', color = colorUse)
                    ax3D.plot(xDimRed[trlsUse, 0],xDimRed[trlsUse, 1],xDimRed[trlsUse, 2], '.', color = colorUse)
            else:
                ax1 = plt.subplot(111)
                for idx, ang in enumerate(uniqueLabel):
                    trlsUse = labelPresented == idx
                    # colorUse = self.colorsetMayavi[idx]
                    colorUse = self.colorset[idx]
                    # mlab.points3d(xDimRed[trlsUse, 0],xDimRed[trlsUse, 1],xDimRed[trlsUse, 2], color = colorUse, scale_factor=10)
                    ax1.hist(xDimRed[trlsUse, 0], color = colorUse)
        
        return ldaModel
    
    def numberOfDimensions(self, labels=None, title='', maxDims = None, baselineSubtract = False, plot=True):
        if labels is not None:
            idxUse = self.balancedTrialInds(labels)
        else:
            idxUse = np.arange(self.shape[0])
            
        if baselineSubtract:
            spikesUse = self.baselineSubtract(labels)
        else:
            spikesUse = self
           
        if self.shape[0] < self.shape[1] or self.size == 0:
               print("dimensional analysis not doable with so few trials")
               return
            
        tmAvgBins = spikesUse.timeAverage()
        trls, chans = tmAvgBins.shape # trials are samples, channels are features
        
        
        
            
        randomizedIndOrder = np.random.permutation(idxUse)
        trainInds = randomizedIndOrder[:round(0.75*trls)]
        testInds = randomizedIndOrder[round(0.75*trls):]
        
        ll = np.ndarray(0)
        cvLL = np.ndarray(0)
        fADims = []
        if maxDims is None:
           
            maxDims = chans
            
        for numDims in range(1, maxDims+1):
            fA = FactorAnalysis(n_components = numDims)
            print(numDims)
            fA.fit(tmAvgBins[trainInds, :])
            
            ll = np.append(ll, fA.loglike_[-1])
            fADims.append(fA.components_)
            cvLL = np.append(cvLL, fA.score(tmAvgBins[testInds, :]))
        
#        plt.figure()
#        plt.plot(ll)
#        plt.ylabel('ll mag')
#        plt.xlabel('dim num')
#        plt.title(title + ' initial ll')
#        highestLL = np.argmax(ll)
#        fADimUse = fADims[highestLL]
        
        if plot:
            plt.figure().suptitle(title)
            cvLLPl = plt.subplot(131)
            plt.plot(np.arange(len(cvLL))+1, cvLL)
            plt.ylabel('cross validated ll')
            plt.xlabel('dim num')
            plt.title('cross validated ll')
            
        highestLL = np.argmax(cvLL)
        fADimUse = fADims[highestLL]
        
        shCovMat = fADimUse.T @ fADimUse
        eInfo = np.linalg.eig(shCovMat)
        pairedEvecEvalList = list(zip(*eInfo))
        dtype = [('eval', np.float64),('evec', np.ndarray)]
        pairedEvecEvalArr = np.array(pairedEvecEvalList, dtype)
        # sortInds = np.argsort(pairedEvecEvalArr['eval'])
        # sortInds = np.flip(sortInds)
        pairedEvecEvalArr.sort(order='eval')
        pairedEvecEvalArr = np.flip(pairedEvecEvalArr)
        
        eVals = pairedEvecEvalArr['eval']
        eVecs = np.stack(pairedEvecEvalArr['evec'])
#        plt.figure()
#        plt.plot(eVals)
#        plt.ylabel('eval mag')
#        plt.xlabel('eval num')
#        plt.title(title + ' shared covariance eval mag')
        
        eValSum = np.sum(eVals)
        shCovByMode = 100*eVals/eValSum
        
        if plot:
            shCovPl = plt.subplot(132)
            plt.plot(np.arange(len(shCovByMode))+1, shCovByMode)
            plt.ylabel('% shared covariance')
            plt.xlabel('eigenvalue number')
            plt.title('% shared cov by mode')
        
        percShCovarThresh = 95
        cumShCovByMode = np.cumsum(shCovByMode)
        
        if plot:
            cumShCov = plt.subplot(133)
            plt.plot(np.arange(len(cumShCovByMode))+1, cumShCovByMode)
            plt.axhline(y=percShCovarThresh, linestyle='--')
            plt.ylabel('cumulative % covariance')
            plt.xlabel('eigenvalue number')
            plt.title('cumul % shared cov by mode')
        
        areaDim = np.where(cumShCovByMode>percShCovarThresh)[0][0]+1 # so it's not zero-indexed
        
        if plot:
            cumShCov.annotate('# dims = ' + str(areaDim), xy = (areaDim, cumShCovByMode[areaDim-1]), xycoords='data', 
                              arrowprops=dict(facecolor='black', shrink=0.05, width=1),
                            xytext=(0.75,0.75), textcoords='axes fraction', horizontalalignment='center', verticalalignment='top')
            
            
            cvLLPl.plot(areaDim, cvLL[areaDim-1], 'rx')
            cvLLPl.axvline(x=areaDim, linestyle='--', color='r')
            cvLLPl.annotate('# dims = ' + str(areaDim), xy = (areaDim, np.mean(cvLLPl.axis()[2:])), xycoords='data', 
                            arrowprops=dict(facecolor='black', shrink=0.05, width=1),
                            xytext=(0.7,0.5), textcoords='axes fraction', horizontalalignment='left', verticalalignment='top')
        
        
        return areaDim
    
    def residualCorrelations(self, labels, separateNoiseCorrForLabels=False, plot=False, figTitle = "", normalize=False):
        residualSpikes = self.baselineSubtract(labels=labels)
        
        
        chanFirst = residualSpikes.swapaxes(0, 1) # channels are now main axis, for sampling in reshape below
        chanFlat = chanFirst.reshape((self.shape[1], -1)) # chan are now rows, time is cols
        
        # let's remove trials that are larger than 3*std
        stdChanResp = np.std(chanFlat, axis=1)
        chanMask = np.abs(chanFlat) > (3*stdChanResp[:,None])
        maskChanFlat = np.ma.array(chanFlat, mask = chanMask)
        
        # do the same as above for the baseline subtracted values, without baseline
        # subtracting--now we can find the true geometric mean firing rate
        flatCnt = self.swapaxes(0,1).reshape((self.shape[1],-1))
        flatCntMn = flatCnt.mean(axis=1)
        flatCntMn = np.expand_dims(flatCntMn, axis=1) # need to add back a lost dimension
        geoMeanCnt = np.sqrt(flatCntMn @ flatCntMn.T)
        
        uniqueLabel, labelPresented = np.unique(labels, axis=0, return_inverse=True)
        
        if plot:
            if separateNoiseCorrForLabels:
                numRows = 2
                numCols = int(np.floor(len(uniqueLabel)/numRows))
            else:
                numRows = 1
                numCols = 1
            figCorr,axsCorr = plt.subplots(numRows, numCols)
            figHist,axsHist = plt.subplots(numRows, numCols)
            
            figCorr.suptitle(figTitle + " paired correlations")
            figHist.suptitle(figTitle + " paired correlation histogram")
            imgs = []
        
        minCorr = 1
        maxCorr = -1
        
        corrSpks = np.empty(geoMeanCnt.shape + (0,))
        
        if separateNoiseCorrForLabels:
            labelPresented = np.repeat(labelPresented, self.shape[2])
            for lblNum, _ in enumerate(uniqueLabel):
                lblSpks = maskChanFlat[:, labelPresented==lblNum]
                covLblSpks = np.ma.cov(lblSpks)
                stdSpks = np.ma.std(lblSpks, axis=1)
                stdMult = np.outer(stdSpks, stdSpks)
                
                corrLblSpks = covLblSpks.data/stdMult
                
                if normalize:
                    corrLblSpks = corrLblSpks/geoMeanCnt
                np.fill_diagonal(corrLblSpks, 0)
                
                
                if plot:
                    
                    upTriInd = np.triu_indices(corrLblSpks.shape[0], 1)
                    imgs.append(axsCorr.flat[lblNum].imshow(corrLblSpks))
                    minCorr = np.minimum(np.min(corrLblSpks.flat), minCorr)
                    maxCorr = np.maximum(np.max(corrLblSpks.flat), maxCorr)
                    
                    axsHist.flat[lblNum].hist(corrLblSpks[upTriInd].flat)
                
                corrSpks = np.concatenate((corrSpks, corrLblSpks[:,:,None]), axis=2) 
                    
            corrSpks = corrSpks.mean(axis=2)
        else:
            covSpks = np.ma.cov(maskChanFlat)
            stdSpks = np.ma.std(maskChanFlat, axis=1)
            stdMult = np.outer(stdSpks, stdSpks)
            
            corrSpks = covSpks.data/stdMult
            if normalize:
                corrSpks = corrSpks/geoMeanCnt
                
            np.fill_diagonal(corrSpks, 0)
            
            if plot:
                imgs.append(axsCorr.imshow(corrSpks))
                minCorr = np.min(corrSpks.flat)
                maxCorr = np.max(corrSpks.flat)
                
                axsHist.hist(corrSpks.flat)
                
                
        if plot:
            if separateNoiseCorrForLabels:
                for im, axCorr, axHist in zip(imgs, axsCorr.flat, axsHist.flat):
                    plt.colorbar(im, ax = axCorr)
                    im.set_clim(minCorr,maxCorr)
                    
                    axHist.set_xlim(minCorr,maxCorr)
            else:
                plt.colorbar(imgs[0], ax = axsCorr)
                imgs[0].set_clim(minCorr, maxCorr)
                
                axsHist.set_xlim(minCorr, maxCorr)
                
        # the returned value is identical to having done a baseline subtract
        # with labels, but this allows the nice call of .noiseCorrelations(labels=<labels>)
        return residualSpikes, corrSpks, geoMeanCnt
    
    # note that timeBeforeAndAfterStart asks about what the first and last points of
    # the binned spikes represent as it relates to the first and last alignment point
    # i.e. the default (0,250) tells us that the first bin is aligned with whatever
    # alignment was used to generate this spike set (say, the delay start), and should
    # be plotted out to 250ms from that point. The default timeBeforeAndAfterEnd
    # of (-250,0) tells us that the *last* bin is aligned at the alignment point
    # for the end (whatever that may be--say, the delay end), and should be plotted
    # starting -250 ms before
    def gpfa(self, eng, description, outputPath, signalDescriptor = "", xDimTest = [2,5,8], 
             labelUse = 'stimulusMainLabel', numConds=1, combineConds = False, firingRateThresh = 1, balanceDirs = True, baselineSubtract = True,
             crossvalidateNum = 4, timeBeforeAndAfterStart=(0,250), timeBeforeAndAfterEnd=(-250, 0), plotInfo=None):
        from matlab import engine
        from classes.GPFA import GPFA
        from mpl_toolkits.mplot3d import Axes3D # for 3d plotting
        from methods.GeneralMethods import prepareMatlab
                

        # eng was input and should be on... but let's check
        # eng = prepareMatlab(eng)
        eng = None;
            
        # dataInd = -2
        # for dataInd in range(len(binnedSpikesAll)):
        # for _ in range(1):
        # onlyDelay = True
        # initRandSt = np.random.get_state()
        # np.random.seed(0)
        # if onlyDelay:
        #     trlIndsUse = BinnedSpikeSet.balancedTrialInds(binnedSpikesOnlyDelay[dataInd], datasets[dataInd].markerTargAngles)
        #     tmValsStart = np.arange(0, 250, binSizeMs)
        #     tmValsEnd = np.arange(-250, 0, binSizeMs)       
        #     # tmValsStart = np.arange(0, furthestBack if furthestBack else 251, binSizeMs)
        #     # tmValsEnd= np.arange(-furthestForward if furthestForward else 251, 0, binSizeMs)
        #     binnedSpikesBalanced = [binnedSpikesOnlyDelay[dataInd][trl] for trl in trlIndsUse]
        # else:
        #     trlIndsUse = BinnedSpikeSet.balancedTrialInds(binnedSpikesAll[dataInd], datasets[dataInd].markerTargAngles)
        #     tmValsStart = np.arange(-furthestBack, furthestBack if furthestBack else 251, binSizeMs)
        #     tmValsEnd = np.arange(-(furthestForward if furthestForward else 251)+binSizeMs, furthestForward, binSizeMs)
        #     binnedSpikesBalanced = [binnedSpikesAll[dataInd][trl] for trl in trlIndsUse]

        # np.random.set_state(initRandSt)
        
        if timeBeforeAndAfterStart is not None:
            tmValsStart = np.arange(timeBeforeAndAfterStart[0], timeBeforeAndAfterStart[1], self.binSize)
        else:
            tmValsStart = np.ndarray((0,0))
            
        if timeBeforeAndAfterEnd is not None:
            tmValsEnd = np.arange(timeBeforeAndAfterEnd[0], timeBeforeAndAfterEnd[1], self.binSize)  
        else:
            tmValsEnd = np.ndarray((0,0))
        
        _, trlIndsUseFR = self.channelsAboveThresholdFiringRate(firingRateThresh=firingRateThresh)
        binnedSpikeHighFR = self[:,trlIndsUseFR,:]
        
        if balanceDirs:
            trlIndsUseLabel = binnedSpikeHighFR.balancedTrialInds(labels=binnedSpikeHighFR.labels[labelUse])
            binnedSpikesUse = binnedSpikeHighFR[trlIndsUseLabel]#.view(np.ndarray)
        else:
            binnedSpikesUse = binnedSpikeHighFR
        # binnedSpikesBalanced = binnedSpikesBalanced.view(BinnedSpikeSet)
        # binnedSpikesBalanced = [self[trl] for trl in trlIndsUse] # FOR LATER
        
        
        newLabels = binnedSpikesUse.labels[labelUse]
        
        
        if baselineSubtract:
            binnedSpikesUse = binnedSpikesUse.baselineSubtract(labels = newLabels)
            firingRateThresh = -1
        # binnedSpikesBalanced.labels[labelUse] = newLabels
        # binnedSpikesBalanced.binSize = self.binSize
        # binnedSpikesBalanced.start = self.start
        # binnedSpikesBalanced.end = self.end
        
        
        uniqueTargAngle, trialsPresented = np.unique(newLabels, return_inverse=True)
        
        if numConds is None:
            stimsUse = np.arange(uniqueTargAngle.shape[0])
        else:
            initRandSt = np.random.get_state()
            np.random.seed(0)
            stimsUse = np.random.randint(0,high=uniqueTargAngle.shape[0],size=numConds)
            stimsUse.sort() # in place sort
            np.random.set_state(initRandSt)
        # binSpkAllArr = np.empty(len(binnedSpikesUse), dtype=object)
        # for idx, _ in enumerate(binSpkAllArr):
        #     binSpkAllArr[idx] = binnedSpikesUse[idx]
            
        # groupedBalancedSpikes = Dataset.groupSpikes(_, trialsPresented, uniqueTargAngle, binnedSpikes = binSpkAllArr); 
        # trialsPresented = np.sort(trialsPresented) #[binSpkAllArr] #
        # groupedBalancedSpikes = [grp.tolist() for grp in groupedBalancedSpikes]
        
        uniqueTargAngleDeg = uniqueTargAngle*180/np.pi
        
        colorset = self.colorset #np.array([[127,201,127],[190,174,212],[253,192,134],[200,200,153],[56,108,176],[240,2,127],[191,91,23],[102,102,102]])/255
        
        grpSpksNpArr, _ = binnedSpikesUse.groupByLabel(newLabels, labelExtract=uniqueTargAngle[stimsUse,None]) # extract one label...
        if combineConds and (numConds is None or numConds>1):
            groupedBalancedSpikes = [BinnedSpikeSet(np.concatenate(grpSpksNpArr, axis=0), binSize = grpSpksNpArr[0].binSize)] # grpSpksNpArr
            condDescriptors = ['s' + '-'.join(['%d' % stN for stN in stimsUse]) + 'Grpd']
        else:
            groupedBalancedSpikes = grpSpksNpArr
            condDescriptors = ['%d' % stN for stN in stimsUse]
            
        ## Start here
        xDimBestAll = []
        gpfaPrepAll = []
        for idx, (grpSpks, condDesc) in enumerate(zip(groupedBalancedSpikes,condDescriptors)):
            
            gpfaPrep = GPFA(grpSpks, firingRateThresh=firingRateThresh)
            gpfaPrepAll.append(gpfaPrep)
            for xDim in xDimTest:
                try:
                    estParams, seqTrainNew, seqTestNew = gpfaPrep.runGpfaInMatlab(eng=eng, fname=outputPath, runDescriptor = signalDescriptor, condDescriptor = condDesc, crossvalidateNum=crossvalidateNum, xDim=xDim)
                except Exception as e:
                    if type(e) is engine.MatlabExecutionError:
                        print(e)
                        continue
                    else:
                        raise(e)
                        
        for idx, gpfaPrep in enumerate(gpfaPrepAll):
            cvApproach = "logLikelihood"
            normalGpfaScore, normalGpfaScoreErr, reducedGpfaScore = gpfaPrep.crossvalidatedGpfaError(eng=eng, approach = cvApproach)
            # best xDim is our lowest error from the normalGpfaScore... for now...
            if cvApproach is "logLikelihood":
                xDimScoreBest = xDimTest[np.argmax(normalGpfaScore)]
            elif cvApproach is "squaredError":
                xDimScoreBest = xDimTest[np.argmin(normalGpfaScore)]
            seqTestUse = gpfaPrep.dimOutput[xDimScoreBest]['seqsTestNew'][0] # just use the first one...
            
            dimsAll = list(gpfaPrep.dimOutput.keys())
            Cparams = [prm['C'] for prm in gpfaPrep.dimOutput[xDimScoreBest]['allEstParams']]
            shEigs = [np.flip(np.sort(np.linalg.eig(C.T @ C)[0])) for C in Cparams]
            percAcc = np.stack([np.cumsum(eVals)/np.sum(eVals) for eVals in shEigs])
            
            shCovThresh = 0.9
            meanPercAcc = np.mean(percAcc, axis=0)
            stdPercAcc = np.std(percAcc, axis = 0)
            xDimBest = np.where(meanPercAcc>shCovThresh)[0][0]+1
            xDimBestAll.append(xDimBest)
            
            if plotInfo is not None:
                
                axScore = plotInfo['axScore']
                
                if cvApproach is "logLikelihood":
                    axScore.plot(xDimTest,normalGpfaScore, label = 'GPFA mean LL over folds')
                    axScore.fill_between(xDimTest, normalGpfaScore-normalGpfaScoreErr, normalGpfaScore+normalGpfaScoreErr, alpha=0.2, label = 'LL err over folds')
                    axScore.set_title(description)
                elif cvApproach is "squaredError":
                    axScore.plot(xDimTest,normalGpfaScore, label = 'Summed GPFA Error Over Folds')
                    axScore.plot(np.arange(len(reducedGpfaScore))+1, reducedGpfaScore, label='Summed Reduced GPFA Error Over Folds')
                    axScore.set_title(description)
                axScore.set_xlabel("num dims")
                # axScore.legend(loc='upper right')
                
                axDim = plotInfo['axDim']
                
                axDim.plot(np.arange(len(meanPercAcc))+1,meanPercAcc)
                axDim.fill_between(np.arange(len(meanPercAcc))+1, meanPercAcc-stdPercAcc,meanPercAcc+stdPercAcc,alpha=0.2)
                
                axDim.axvline(xDimBest, linestyle='--')
                axDim.axhline(shCovThresh, linestyle='--')
                
                axScore.axvline(xDimBest, linestyle='--')
                
                
                rowsPlt = 2
                if tmValsStart.size and tmValsEnd.size:
                    colsPlt = np.ceil(xDimBest/rowsPlt)*2 # align to start and to end...
                else:
                    colsPlt = np.ceil(xDimBest/rowsPlt) # aligning to only one of them...
                axesStart = []
                axesEnd = []
                axVals = np.empty((0,4))
                figSep = plt.figure()
                if xDimBest>1:
                    fig3 = plt.figure()
                    axStart3d = plt.subplot(1,3,1,projection='3d')
                    axEnd3d = plt.subplot(1,3,2,projection='3d')
                    axAll3d = plt.subplot(1,3,3,projection='3d')
                plt.suptitle(description + " " + str(int(uniqueTargAngleDeg[idx])) + " deg")
                
                for sq in seqTestUse:
                    
                    if xDimBest>2:
                        plt.figure(fig3.number)
                        if tmValsStart.size:
                            axStart3d.plot(sq['xorth'][0,:tmValsStart.shape[0]], sq['xorth'][1,:tmValsStart.shape[0]], sq['xorth'][2,:tmValsStart.shape[0]],
                                   color=colorset[idx,:], linewidth=0.4)
                            axStart3d.set_title('Start')
                            axStart3d.set_xlabel('gpfa 1')
                            axStart3d.set_xlabel('gpfa 2')
                            axStart3d.set_xlabel('gpfa 3')
                        
                        if tmValsEnd.size:
                            axEnd3d.plot(sq['xorth'][0,-tmValsEnd.shape[0]:], sq['xorth'][1,-tmValsEnd.shape[0]:], sq['xorth'][2,-tmValsEnd.shape[0]:],
                                       color=colorset[idx,:], linewidth=0.4)
                            axEnd3d.set_title('End')
                            axEnd3d.set_xlabel('gpfa 1')
                            axEnd3d.set_xlabel('gpfa 2')
                            axEnd3d.set_xlabel('gpfa 3')
                        
                        axAll3d.plot(sq['xorth'][0,:], sq['xorth'][1,:], sq['xorth'][2,:],
                                   color=colorset[idx,:], linewidth=0.4)
                        axAll3d.set_title('All')
                    elif xDimBest>1:
                        plt.figure(fig3.number)
                        if tmValsStart.size:
                            axStart3d.plot(sq['xorth'][0,:tmValsStart.shape[0]], sq['xorth'][1,:tmValsStart.shape[0]],
                                   color=colorset[idx,:], linewidth=0.4)
                            axStart3d.set_title('Start')
                            axStart3d.set_xlabel('gpfa 1')
                            axStart3d.set_xlabel('gpfa 2')
                            axStart3d.set_xlabel('gpfa 3')
                        
                        if tmValsEnd.size:
                            axEnd3d.plot(sq['xorth'][0,-tmValsEnd.shape[0]:], sq['xorth'][1,-tmValsEnd.shape[0]:],
                                       color=colorset[idx,:], linewidth=0.4)
                            axEnd3d.set_title('End')
                            axEnd3d.set_xlabel('gpfa 1')
                            axEnd3d.set_xlabel('gpfa 2')
                            axEnd3d.set_xlabel('gpfa 3')
                        
                        axAll3d.plot(sq['xorth'][0,:], sq['xorth'][1,:],
                                   color=colorset[idx,:], linewidth=0.4)
                        axAll3d.set_title('All')
                    
                    if True:
                        pltNum = 1
                        plt.figure(figSep.number)
                        plt.suptitle(description + " " + str(int(uniqueTargAngleDeg[idx])) + " deg")
                        for dimNum, dim in enumerate(sq['xorth']):
                            dimNum = dimNum+1 # first is 1-dimensional, not zero-dimensinoal
                            #we'll only plot the xDimBest dims...
                            if dimNum > xDimBest:
                                continue
                            
                            if tmValsStart.size:
                                if len(axesStart) + len(axesEnd) <rowsPlt*colsPlt:
                                    axesHere = figSep.add_subplot(rowsPlt, colsPlt, pltNum)
                                    axesStart.append(axesHere)
                                    if pltNum <= colsPlt:
                                        axesHere.set_title("Start " + str(dimNum))
                                else:
                                    if tmValsEnd.size:
                                        axesHere = axesStart[int((pltNum-1)/2)]
                                    else:
                                        axesHere = axesStart[int(pltNum-1)]    
                                    plt.axes(axesHere)
                            
                                plt.plot(tmValsStart, dim[:tmValsStart.shape[0]], color=colorset[idx,:], linewidth=0.4)
                            
                            
                            
                                axVals = np.append(axVals, np.array(axesHere.axis())[None, :], axis=0)
                                pltNum += 1
                            
                            if tmValsEnd.size:
                                if len(axesStart) + len(axesEnd) <rowsPlt*colsPlt:
                                    axesHere = figSep.add_subplot(rowsPlt, colsPlt, pltNum)
                                    axesEnd.append(axesHere)
                                    if pltNum <= colsPlt:
                                        axesHere.set_title("End " + str(dimNum))
                                else:
                                    if tmValsStart.size:
                                        axesHere = axesEnd[int(pltNum/2-1)]
                                    else:
                                        axesHere = axesEnd[int(pltNum-1)]
                                    plt.axes(axesHere)
                        
                                plt.plot(tmValsEnd, dim[-tmValsEnd.shape[0]:], color=colorset[idx,:], linewidth=0.4)
                 
                                axVals = np.append(axVals, np.array(axesHere.axis())[None, :], axis=0)
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
       
        return xDimBestAll, gpfaPrepAll
