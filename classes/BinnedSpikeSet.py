#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 15:53:35 2020

@author: emilio
"""
import numpy as np
import scipy as sp
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from classes.FactorAnalysis import FactorAnalysis
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # for 3d plotting
#from mayavi import mlab

from copy import copy

import pdb

HANDLED_FUNCTIONS = {}

class BinnedSpikeSet(np.ndarray):
    dimDescription = {'rows':'trials',
                      'cols':'channel',
                      'depth':'time'}
    
    
    
    colorset = np.array([[127,201,127],[190,174,212],[253,192,134],[200,200,153],
                         [56,108,176],[240,2,127],[191,91,23],[102,102,102]])/255
    colorsetMayavi = [tuple(col) for col in colorset]
    
    # this has to do with how it's suggested that ndarray be subclassed...
    # https://docs.scipy.org/doc/numpy/user/basics.subclassing.html
    def __new__(cls, input_array, start=None, end=None, binSize=None, labels = {}, alignmentBins = None, units = None):
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
        obj.units = units

        if alignmentBins is list:
            obj.alignmentBins = np.stack(alignmentBins)
        else:
            obj.alignmentBins = alignmentBins # this is meant to tell us how this BinnedSpikeSet
                # was aligned when it was generated (i.e. what points in time correspond to an important 
                # stimulus-related change, say)
        obj._new_label_index = None
                
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
        self.units = getattr(obj, 'units', None)
        self.alignmentBins = copy(getattr(obj, 'alignmentBins', None))
#        print(self.alignmentBins)
        try:
            # print('size = ' + str(len(obj._new_label_index)))
            # print('first ' + str(obj._new_label_index))
            # print('last ' + str(obj._new_label_index) + '\n')
#            print('newlbind')
#            print(obj._new_label_index)
            if obj._new_label_index is not None: # we're gonna try and make None be the setting where no indexing has happened...
                for key, val in self.labels.items():
                    self.labels[key] = val[obj._new_label_index]
                
                self.alignmentBins = self.alignmentBins[obj._new_label_index]
#                print('albin')
#                print(self.alignmentBins)
#                print(obj._new_label_index)
                    
                self._new_label_index = getattr(obj, '_new_label_index', None)
                setattr(obj, '_new_label_index', None)
        except Exception as e:
#            print(e)
            if hasattr(obj, '_new_label_index'):
                # print('tried')
                # print(obj._new_label_index)
                # print('failed')
                pass
            else:
                pass;#print('no _new_label_index')
            pass
                # and also to be expansible if new/overlapping labels present
        self._new_label_index = getattr(self, '_new_label_index', None)
        # We do not need to return anything
        
    # this little tidbit of info inspired by
    # https://stackoverflow.com/questions/16343034/numpy-subclass-attribute-slicing
    # But for the moment... seems a bit too complex to want to deal with...
    def __getitem__(self,item):
        try:
            # print(type(self))
            if isinstance(item, (list,slice, int, np.ndarray)) or np.issubdtype(type(item), np.integer):
                # print('test')
                self._new_label_index = item
                #setattr(self,'_new_label_index', item)
            elif isinstance(item[0], (list,slice, int, np.ndarray)) or np.issubdtype(type(item[0]), np.integer):
                # print('test2')
                # print(str(item))
                # print(type(item))
                # print(str(self.shape))
                # print(str(self.labels))
                self._new_label_index = item[0]
                # setattr(self,'_new_label_index', item[0])
            # else:
                # print('test3')
                # print(type(item))
                # print(item)
                # print(str(self.shape))
                # self._new_label_index = item[0]
                # print('hello')
        except: 
            pass
        # print("0")
        return super().__getitem__(item)
    
    def __array_function__(self, func, types, args, kwargs):
        if func not in HANDLED_FUNCTIONS:
            if type(args) == tuple:
                newArgs = []
                for ar in args:
                    if type(ar) is BinnedSpikeSet:
                        newArgs.append(np.array(ar))
                    else:
                        newArgs.append(ar)
                newArgs = tuple(newArgs)
            else:
                newArgs = args

            return BinnedSpikeSet(func(*newArgs, **kwargs))
#                labels=self.labels.copy() if self.labels is not None else {},
#                alignmentBins = self.alignmentBins.copy if self.alignmentBins is not None else None,
#                units = self.units
#            )
        # Note: this allows subclasses that don't override
        # __array_function__ to handle BinnedSpikeSet objects
        if not all(issubclass(t, BinnedSpikeSet) for t in types):
            return NotImplemented
        return HANDLED_FUNCTIONS[func](*args, **kwargs)
        
    def copy(self, *args, **kwargs):
        out = BinnedSpikeSet(self.view(np.ndarray).copy(*args, **kwargs), start=self.start, end=self.end, binSize=self.binSize, labels=self.labels.copy() if self.labels is not None else {}, alignmentBins=self.alignmentBins.copy() if self.alignmentBins is not None else None, units=self.units)
        
        # # copy back the labels
        # for labelName, labels in self.labels.items():
        #     out.labels[labelName] = labels
        
        return out
#%% general methods
        
    def timeAverage(self):
        if self.dtype == 'object':
            out = self.copy()
            out[:] = [np.average(np.stack(trl), axis=1) for trl in out]
        else:
            out = np.average(self, axis=2)
        return out
    
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
    def computeTimestampFromFirstBin(self):
        trlTmStmp = []
        for trl in self:
            trlTmStmp.append([self.binSize*(np.where(trlChan)[0][None,:]+0.5) for trlChan in trl])
            
        return trlTmStmp
        
    def balancedTrialInds(self, labels, minCnt = None):
        unq, unqCnts = np.unique(labels, return_counts=True, axis=0)
        if minCnt is None:
            minCnt = min(unqCnts)
        idxUse = []
        
        for idx, cnt in enumerate(unqCnts):
            labHere = unq[idx]
            inds, = np.nonzero(np.all(labels==labHere, axis=labHere.ndim))
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
            unq, unqCnts = np.unique(labels, return_counts=True, axis=0)
            spikesUse = self.copy()
            overallBaseline = []
            for lbl in unq:
                overallBaseline.append(self[np.all(labels==lbl,axis=lbl.ndim), :, :].trialAverage())
                spikesUse[np.all(labels==lbl,axis=lbl.ndim), :, :] = self[np.all(labels==lbl,axis=lbl.ndim), :, :] - self[np.all(labels==lbl,axis=lbl.ndim), :, :].trialAverage()

        return spikesUse, overallBaseline

    # group by all unique values in label and/or group by explicit values in labelExtract
    def groupByLabel(self, labels, labelExtract=None):
        unq, unqCnts = np.unique(labels, return_counts=True, axis=0)
        if labelExtract is None:
            groupedSpikes = [self[np.all(labels==lbl,axis=lbl.ndim)] for lbl in unq]
        else:
            groupedSpikes = [self[np.all(labels==lbl,axis=lbl.ndim)] for lbl in labelExtract]
            unq = labelExtract
            
        
        return groupedSpikes, unq  

    def avgFiringRateByChannel(self):
        if self.units != 'Hz':
            hzBinned = self.convertUnitsTo(units='Hz')
        else:
            hzBinned = self

        if self.dtype == 'object':
            chanFirst = hzBinned.swapaxes((0,1))
            avgFiringChan = [np.mean(np.stack(chan), axis=1) for chan in chanFirst]
            breakpoint() # because I'm not sure this is correct...
        else:
            avgFiringChan = hzBinned.timeAverage().trialAverage()
        
        return avgFiringChan

    def stdFiringRateByChannel(self):
        if self.units != 'Hz':
            hzBinned = self.convertUnitsTo(units='Hz')
        else:
            hzBinned = self

        if self.dtype == 'object':
            chanFirst = hzBinned.swapaxes((0,1))
            avgFiringStdChan = [np.std(np.stack(chan), axis=1) for chan in chanFirst]
            breakpoint() # because I'm not sure this is correct... NOTE want mean before std?

        else:
            avgFiringStdChan = hzBinned.timeAverage().trialStd()
        
        return avgFiringStdChan

    def sumTrialCountByChannel(self):
        if self.units != 'count':
            cntBinned = self.convertUnitsTo(units='count')
        else:
            cntBinned = self

        if self.dtype == 'object':
            chanFirst = cntBinned.swapaxes((0,1))
            avgCountChan = np.stack([np.sum(np.stack(chan), axis=1) for chan in chanFirst])
            breakpoint() # because I'm not sure this is correct...
        else:
            avgCountChan = cntBinned.sum(axis=2).trialAverage()
        
        return avgCountChan

    def stdSumTrialCountByChannel(self):
        if self.units != 'count':
            cntBinned = self.convertUnitsTo(units='count')
        else:
            cntBinned = self

        if self.dtype == 'object':
            chanFirst = cntBinned.swapaxes((0,1))
            avgCountStdChan = np.stack([np.std(np.stack(chan), axis=1) for chan in chanFirst])
            breakpoint() # because I'm not sure this is correct... NOTE want sum before std?
        else:
            # we're taking a sum over the bins before taking the standard
            # deviation... so it's the standard devation of spike counts in a
            # trial no matter what... but that's also how the functions named so hah!
            avgCountStdChan = cntBinned.sum(axis=2).trialStd()
        
        return avgCountStdChan

    # this is different from chaining timeAverage and trialAverage together
    # when there are different numbers of bins in different trials
    def avgValByChannel(self):
        if self.dtype == 'object':
            chanFirst = self.swapaxes(0,1)
            avgValChan = np.stack([np.mean(np.stack(chan), axis=1) for chan in chanFirst])
            breakpoint() # because I'm not sure this is correct...
        else:
            avgValChan = self.timeAverage().trialAverage()
        
        return avgValChan

    def stdValByChannel(self):
        if self.dtype == 'object':
            chanFirst = self.swapaxes(0,1)
            stdValChan = np.stack([np.std(np.stack(chan), axis=1) for chan in chanFirst])
            breakpoint() # because I'm not sure this is correct...
        else:
            numChannels = self.shape[1]
            stdValChan = self.swapaxes(0,1).reshape(numChannels,-1).std(axis=1)
        
        return stdValChan

    # note: while this function will only reflect fano factor of 1 for poisson
    # distributed spike counts if self.units is counts, not Hz, the definition
    # of fano factor is just var/mean, so... I'm not gonna do that check here;
    # I expect other functions feeding into this one to check that the units
    # make sense
    def fanoFactorByChannel(self):
        avgValChan = self.avgValByChannel()
        stdValChan = self.stdValByChannel()

        # fano factor is variance/mean
        return stdValChan**2/avgValChan

    def convertUnitsTo(self, units='Hz'):
        if self.units is None:
            raise(Exception("Don't know what the original units were!"))
        elif self.units == units:
            return self.copy() # return a copy here...
        else:
            if self.units == 'Hz':
                if units == 'count':
                    outNewUnits = self.copy()
                    outNewUnits = outNewUnits*self.binSize/1000
                    outNewUnits.units = units
                else:
                    raise(Exception("Don't know how to convert between %s and %s") % (self.units, units))
            elif self.units == 'count':
                if units == 'Hz':
                    outNewUnits = self.copy()
                    outNewUnits = outNewUnits*1000/self.binSize
                    outNewUnits.units = units
                else:
                    raise(Exception("Don't know how to convert between %s and %s") % (self.units, units))
            else:
                raise(Exception("Don't know how to convert between %s and %s") % (self.units, units))

            return outNewUnits

#%% filtering mechanics
    def channelsAboveThresholdFiringRate(self, firingRateThresh=1): #low firing thresh in Hz
        avgFiringChan = self.avgFiringRateByChannel()
        
        highFiringChannels = avgFiringChan>firingRateThresh
        
        return self[:,highFiringChannels], np.where(highFiringChannels)[0]

    # note that this will always calculate the *spike count* fano factor
    def channelsBelowThresholdFanoFactor(self, fanoFactorThresh = 4):
        if self.units != 'count':
            cntBinned = self.convertUnitsTo(units='count')
        else:
            cntBinned = self

        fanoFactorChans = cntBinned.fanoFactorByChannel()

        lowFanoFactorChans = fanoFactorChans < fanoFactorThresh

        return self[:,lowFanoFactorChans], np.where(lowFanoFactorChans)[0]

    def channelsNotRespondingSparsely(self, zeroRate=0): 
        # note that 'sparsely' here has a very specific definition (of mine)
        # related to the responses of the channel having to occur often enough
        # to not exist only at >3 standard deviations from the mean
        #
        # the steps to get there are the following: first flatten the firing
        # rate of all the channels, then find the standard deviation of said
        # firing rate. Next, we mark the firing rates that are >3x the standard
        # deviation (this could be considered time points that have 'outlier'
        # firing rates. Then we check whether the only remaining unmarked
        # locations had zero firing rate to begin with. The flag for 'zero
        # rate' reflects that sometimes a trial-mean subtracted array will be
        # input here, where 0 firing rate gets mean subtracted to some
        # different baseline value

        if zeroRate.ndim == 3:
            # we're resizing zeroRate, which was given on a per-trial basis, to be the right size
            zeroRateFlattened = zeroRate.swapaxes(0, 1).reshape((zeroRate.shape[1], -1))

        
        flattenedResp = np.array(self.swapaxes(0,1).reshape((self.shape[1],-1)))
        stdFlat = np.std(flattenedResp, axis=1)

        maskGT3Std = np.abs(flattenedResp) > 3*stdFlat[:,None]


        # add the mean back in, effectively making no firing rate locations equal 0 again
        flattenedShiftedResp = flattenedResp + zeroRateFlattened

        zeroFiringRate = 0
        numRespsNotZeroRate = [np.sum(flattenedShiftedResp[i][~maskGT3Std[i]]!=zeroFiringRate) for i in range(flattenedResp.shape[0])]
        numRespsNotZeroRate = np.array(numRespsNotZeroRate)

        chansResponding = numRespsNotZeroRate != 0
        # this line only to remind myself of the double negative of 'not' and
        # 'sparsely' meaning actually responding heh
        chansNotRespondingSparsely = chansResponding 


        
        return self[:,chansNotRespondingSparsely,:], np.where(chansNotRespondingSparsely)[0]

    def removeInconsistentChannelsOverTrials(self, zeroRate=0): 
        # Alright instead of schmancy thigns I tried to do, all I'm going to do
        # now is check which channels had any firing during >90% of trials,
        # after removing firing rates that were >3*std of firing rates of that
        # channel.

        # remove firing rate bins where the neuron fired at >3*std for that
        # neuron
        flattenedResp = np.array(self.swapaxes(0,1).reshape((self.shape[1],-1)))
        stdFlat = np.std(flattenedResp, axis=1)

        selfShift = self+zeroRate
        maskGT3Std = np.abs(self) > 3*stdFlat[None,:,None]

        # count the time bins for which the neuron fired in each trial
        numRespsPerTrlAndChan = [[np.sum(chn[~chMsk]) for chn,chMsk in zip(trls,trlMsk)] for trls,trlMsk in zip(selfShift, maskGT3Std)]
        respSizePerTrlAndChan = np.stack(numRespsPerTrlAndChan)

        # find the number of trials for which the neuron responded *at all*
        numRespTrialsByChan = np.sum(respSizePerTrlAndChan!=0, axis=0)

        # only select neurons that responded in more than 90% of trials
        consistentChans = (numRespTrialsByChan/self.shape[0])>=0.9
        breakpoint()

        return self[:,consistentChans,:], np.where(consistentChans)[0]
#        # alright, this function aims to remove channels which don't respond to
#        # some large subset of trials. How is 'large subset' defined?
#        # Wyyeeelll... by making some assumptions.
#        # 
#        # Note that in all these cases, we are first removing firing rates that
#        # are >3*std of the normal firing rate.
#        #
#        # Now. First, I make the assumption that a channel responding to a
#        # trial at all (once those 3x firing rates are removed) is a Bernoulli
#        # distributed variable (with a 1 if there are any spikes and a 0 if
#        # there are none). As a result, the sum of those successes is a
#        # binomial distributed variable with the median probability, p, that a
#        # trial from a channel fired during all the trials being its p. The
#        # expectation of this sum is thus np
#        #
#        # Since this variable is (assumed to be) Binomial distributed, I can
#        # now compute its variance by var = np*(1-p). With this variance, I can
#        # find channels that responded less than p-6*sqrt(var) of the
#        # time--effectively removing channels responding less than 3 standard
#        # deviations from average. I generally don't mind high responding
#        # channels, so I won't remove them... but I will also warn that they
#        # could be removed.
#        flattenedResp = np.array(self.swapaxes(0,1).reshape((self.shape[1],-1)))
#        stdFlat = np.std(flattenedResp, axis=1)
#
#        selfShift = self+zeroRate
#        maskGT3Std = np.abs(self) > 3*stdFlat[None,:,None]
#        numRespsPerTrlAndChan = [[np.sum(chn[~chMsk]) for chn,chMsk in zip(trls,trlMsk)] for trls,trlMsk in zip(selfShift, maskGT3Std)]
#        respSizePerTrlAndChan = np.stack(numRespsPerTrlAndChan)
#
#        # find the rate of response per channel
#        numRespsByChan = np.sum(respSizePerTrlAndChan!=0, axis=0)
#
#        numTrials = self.shape[0]
#        binomialProbabilityP = np.mean(numRespsByChan)/numTrials
#        binomialExpectation = numTrials * binomialProbabilityP
#        binomialVarRespProp = numTrials*binomialProbabilityP * (1-binomialProbabilityP)
#        binomialStdRespProp = np.sqrt(binomialVarRespProp)
#
#        # note the multiplication by 6 here for 6*std
#        # also note that 6 is a... not... to... specific number?
#        consistentChans = (numRespsByChan - binomialExpectation) >= -6*binomialStdRespProp
#
#        if np.any((numRespsByChan - binomialExpectation) > 3*binomialStdRespProp):
#            print("Seems like some high FR channels have too... much... inconsistency?")
#
#        breakpoint()
#
#        return self[:,consistentChans,:], np.where(consistentChans)[0]
#
    # coincidenceTime is how close spikes need to be, in ms, to be considered coincicdent, 
    # coincidentThresh in percent/100 (i.e. 20% = 0.2)
    def removeCoincidentSpikes(self, coincidenceTime=1, coincidentThresh=0.2): 
        spikeCountOverallPerChan = self.timeSpikeCount().trialSpikeCount()
        
        timestampSpikes = self.computeTimestampFromFirstBin()
        
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
        if newBinSize/self.binSize > self.shape[2]:
            raise Exception('BinnedSpikeSet:BadBinSize', 'New bin size larger than the length of each trials!')
            return None
        if (newBinSize/self.binSize) % self.shape[2]:
            raise Exception('BinnedSpikeSet:BadBinSize', "New bin size doesn't evenly divide trajectory. Avoiding splitting to not have odd last bin.")
        
        binsPerTrial = [np.arange(st, en+newBinSize/20, newBinSize)-st for st, en in zip(self.start,self.end)]
        spikeTimestamps = self.computeTimestampFromFirstBin()
        
        spikeDatBinnedList = []
        alignmentBins = []
        for trlNum, (trl, bins) in enumerate(zip(self, binsPerTrial)):
            spkDatBinnedTrl = []
            for chanNum, chanSpks in enumerate(trl):
                if self.units == 'count':
                    spkDatBinnedTrl.append(sp.stats.binned_statistic(spikeTimestamps[trlNum][chanNum].flatten(), chanSpks[np.where(chanSpks)], statistic='sum', bins=bins)[0] if spikeTimestamps[trlNum][chanNum].size else np.zeros_like(bins)[:-1])
                elif self.units == 'Hz':
                    spkDatBinnedTrl.append(sp.stats.binned_statistic(spikeTimestamps[trlNum][chanNum].flatten(), chanSpks[np.where(chanSpks)], statistic='mean', bins=bins)[0] if spikeTimestamps[trlNum][chanNum].size else np.zeros_like(bins)[:-1])
                else:
                    raise(Exception("Don't know how to rebin with units: %s!" % self.units))
            spikeDatBinnedList.append(spkDatBinnedTrl)
            alignmentBins.append(np.digitize(self.alignmentBins[trlNum], bins=bins) - 1)

        newBinnedSpikeSet = np.stack(spikeDatBinnedList).view(BinnedSpikeSet)

        newBinnedSpikeSet.binSize = newBinSize
        newBinnedSpikeSet.start = self.start
        newBinnedSpikeSet.end = self.end
        newBinnedSpikeSet.units = self.units
        newBinnedSpikeSet.alignmentBins = np.stack(alignmentBins)
        newBinnedSpikeSet.labels = self.labels
        
        return newBinnedSpikeSet
        
#%% Analysis methods
    def pca(self, baselineSubtract = False, labels = None, crossvalid=True, n_components = None, plot = False):
        if n_components is None:
            if self.shape[0] < self.shape[1]:
                print("PCA not doable with so few trials")
                return
            n_components =  self.shape[1]
            
        if baselineSubtract and labels is not None:
            chanSub, _ = self.baselineSubtract(labels=labels)
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
            chanSub, _ = self.baselineSubtract(labels=labels)
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
            spikesUse, _ = self.baselineSubtract(labels)
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
        residualSpikes, overallBaseline = self.baselineSubtract(labels=labels)
        unq, unqInv = np.unique(labels, return_inverse=True, axis=0)
        residualSpikesGoodChans, chansGood = residualSpikes.channelsNotRespondingSparsely(zeroRate = np.array(overallBaseline)[unqInv])
        residualSpikesGoodChans, chansGood = residualSpikesGoodChans.removeInconsistentChannelsOverTrials(zeroRate = np.array(overallBaseline)[unqInv])
        
        
        chanFirst = residualSpikesGoodChans.swapaxes(0, 1) # channels are now main axis, for sampling in reshape below
        chanFlat = chanFirst.reshape((residualSpikesGoodChans.shape[1], -1)) # chan are now rows, time is cols
        
        # let's remove trials that are larger than 3*std
        stdChanResp = np.std(chanFlat, axis=1)
        chanMask = np.abs(chanFlat) > (3*stdChanResp[:,None]) # < 0
        maskChanFlat = np.ma.array(np.array(chanFlat), mask = np.array(chanMask))
        
        # do the same as above for the baseline subtracted values, without baseline
        # subtracting--now we can find the true geometric mean firing rate
        flatCnt = np.array(self[:,chansGood,:].swapaxes(0,1).reshape((chansGood.size,-1)))
        flatCntMn = flatCnt.mean(axis=1)
        flatCntMn = np.expand_dims(flatCntMn, axis=1) # need to add back a lost dimension
        geoMeanCnt = np.sqrt(flatCntMn @ flatCntMn.T)
        
        uniqueLabel, labelPresented = np.unique(labels, axis=0, return_inverse=True)

        kla = [np.sum(flatCnt[i][~chanMask[i]]) for i in range(flatCnt.shape[0])]
        kla = np.array(kla)
        maskChanSpksLeft = maskChanFlat[kla!=0]
        maskChanNoSpksLeft = maskChanFlat[kla==0]

#        _, t = self.channelsNotRespondingSparsely()
        breakpoint()

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
        
        corrSpksPerCond = np.empty(geoMeanCnt.shape + (0,))
        
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
                
                corrSpksPerCond = np.concatenate((corrSpksPerCond, corrLblSpks[:,:,None]), axis=2) 
                    
            corrSpksCondMn = corrSpksPerCond.mean(axis=2)
        else:
            covSpks = np.ma.cov(maskChanFlat)
            stdSpks = np.ma.std(maskChanFlat, axis=1)
            stdMult = np.outer(stdSpks, stdSpks)
            
            corrSpksPerCond = covSpks.data/stdMult
            if normalize:
                corrSpksPerCond = corrSpksPerCond/geoMeanCnt
                
            np.fill_diagonal(corrSpksPerCond, 0)
            
            if plot:
                imgs.append(axsCorr.imshow(corrSpksPerCond))
                minCorr = np.min(corrSpksPerCond.flat)
                maxCorr = np.max(corrSpksPerCond.flat)
                
                axsHist.hist(corrSpksPerCond.flat)

            corrSpksCondMn = corrSpksPerCond
                
                
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
        return residualSpikesGoodChans, corrSpksCondMn, corrSpksPerCond, geoMeanCnt
    
    # note that timeBeforeAndAfterStart asks about what the first and last points of
    # the binned spikes represent as it relates to the first and last alignment point
    # i.e. the default (0,250) tells us that the first bin is aligned with whatever
    # alignment was used to generate this spike set (say, the delay start), and should
    # be plotted out to 250ms from that point. The default timeBeforeAndAfterEnd
    # of (-250,0) tells us that the *last* bin is aligned at the alignment point
    # for the end (whatever that may be--say, the delay end), and should be plotted
    # starting -250 ms before
    def gpfa(self, eng, description, outputPath, signalDescriptor = "", xDimTest = [2,5,8], sqrtSpikes = False,
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
        
        if type(self) is list:
            bnSpksCheck = np.concatenate(self, axis=2).view(BinnedSpikeSet)
            bnSpksCheck.labels = {}
            bnSpksCheck.labels[labelUse] = np.stack([bnSp.labels[labelUse] for bnSp in self])
            binSize = self[0].binSize
            colorset = self[0].colorset
        elif self.dtype == 'object':
            bnSpksCheck = np.concatenate(self, axis=2).view(BinnedSpikeSet)
            bnSpksCheck.labels = {}
            # bastardization of how BinnedSpikeSet should be used, give that labels
            # should refer to trials, and bnSpksCheck is flattened to one 'trial'
            bnSpksCheck.labels[labelUse] = np.stack([bnSp.labels[labelUse] for bnSp in self])
            binSize = self[0].binSize
            colorset = self[0].colorset
        else:
            bnSpksCheck = self
            binSize = self.binSize
            colorset = self.colorset
        
        if timeBeforeAndAfterStart is not None:
            tmValsStartBest = np.arange(timeBeforeAndAfterStart[0], timeBeforeAndAfterStart[1], binSize)
        else:
            tmValsStartBest = np.ndarray((0,0))
            
        if timeBeforeAndAfterEnd is not None:
            tmValsEndBest = np.arange(timeBeforeAndAfterEnd[0], timeBeforeAndAfterEnd[1], binSize)  
        else:
            tmValsEndBest = np.ndarray((0,0))
        
        _, chIndsUseFR = bnSpksCheck.channelsAboveThresholdFiringRate(firingRateThresh=firingRateThresh)
        # binnedSpikeHighFR = self[:,chIndsUseFR,:]
        
        if balanceDirs:
            trlIndsUseLabel = self.balancedTrialInds(labels=self.labels[labelUse])
            # binnedSpikesUse = binnedSpikeHighFR[trlIndsUseLabel]#.view(np.ndarray)
        else:
            trlIndsUseLabel = range(len(self))
            # binnedSpikesUse = binnedSpikeHighFR
        # binnedSpikesBalanced = binnedSpikesBalanced.view(BinnedSpikeSet)
        # binnedSpikesBalanced = [self[trl] for trl in trlIndsUse] # FOR LATER
        
        if type(self) is list:
            binnedSpikeHighFR = [bnSp[:,chIndsUseFR, :] for bnSp in self]
            binnedSpikesUse = np.empty(len(binnedSpikeHighFR), dtype='object')
            binnedSpikesUse[:] = binnedSpikeHighFR
            binnedSpikesUse = binnedSpikesUse[trlIndsUseLabel]
            binnedSpikesUse = BinnedSpikeSet(binnedSpikesUse, binSize = binSize)
            newLabels = np.stack([bnSp.labels[labelUse] for bnSp in binnedSpikesUse])
            binnedSpikesUse.labels[labelUse] = newLabels
            if baselineSubtract:
                raise(Exception("Can't baseline subtract trials of unequal size!"))
        elif self.dtype == 'object':
            binnedSpikeHighFR = self[:,chIndsUseFR]
            binnedSpikesUse = binnedSpikeHighFR[trlIndsUseLabel]
            newLabels = binnedSpikesUse.labels[labelUse]
            if baselineSubtract:
                if sqrtSpikes:
                    # we want to square root before negative numbers appear because of baseline subtracting
                    aB = binnedSpikesUse.alignmentBins
                    binnedSpikesUse = np.sqrt(binnedSpikesUse)
                    sqrtSpikes = False
                    binnedSpikesUse.labels[labelUse] = newLabels
                    binnedSpikesUse.alignmentBins = aB
                binnedSpikesUse, labelMeans = binnedSpikesUse.baselineSubtract(labels = newLabels)

                # ROMP this might break stuff because I haven't accounted for
                # non equal trial sizes when aligning objects for
                # non-sparse-responses
                _, unqInv = np.unique(newLabels, return_inverse=True, axis=0)
                binnedSpikesUse, chansGood = binnedSpikesUse.channelsNotRespondingSparsely(zeroRate = np.array(labelMeans)[unqInv])
                labelMeans = [lM[chansGood,:] for lM in labelMeans]
                binnedSpikesUse, chansGood = binnedSpikesUse.removeInconsistentChannelsOverTrials(zeroRate = np.array(overallBaseline)[unqInv])
                labelMeans = [lM[chansGood,:] for lM in labelMeans]
                firingRateThresh = -1
        else:
            binnedSpikeHighFR = self[:,chIndsUseFR,:]
            binnedSpikesUse = binnedSpikeHighFR[trlIndsUseLabel]
            newLabels = binnedSpikesUse.labels[labelUse]
            if baselineSubtract:
                if sqrtSpikes:
                    # we want to square root before negative numbers appear because of baseline subtracting
                    aB = binnedSpikesUse.alignmentBins
                    binnedSpikesUse = np.sqrt(binnedSpikesUse)
                    sqrtSpikes = False
                    binnedSpikesUse.labels[labelUse] = newLabels
                    binnedSpikesUse.alignmentBins = aB
                binnedSpikesUse, labelMeans = binnedSpikesUse.baselineSubtract(labels = newLabels)

                _, unqInv = np.unique(newLabels, return_inverse=True, axis=0)
                binnedSpikesUse, chansGood = binnedSpikesUse.channelsNotRespondingSparsely(zeroRate = np.array(labelMeans)[unqInv])
                labelMeans = [lM[chansGood,:] for lM in labelMeans]
                binnedSpikesUse, chansGood = binnedSpikesUse.removeInconsistentChannelsOverTrials(zeroRate = np.array(labelMeans)[unqInv])
                labelMeans = [lM[chansGood,:] for lM in labelMeans]
                firingRateThresh = -1
        
        
        uniqueTargAngle, trialsPresented = np.unique(newLabels, return_inverse=True, axis=0)
        
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
        
        
        
        grpSpksNpArr, _ = binnedSpikesUse.groupByLabel(newLabels, labelExtract=uniqueTargAngle[stimsUse]) # extract one label...
        if combineConds and (numConds is None or numConds>1):
            groupedBalancedSpikes = [BinnedSpikeSet(np.concatenate(grpSpksNpArr, axis=0), binSize = grpSpksNpArr[0].binSize)] # grpSpksNpArr
            condDescriptors = ['s' + '-'.join(['%d' % stN for stN in stimsUse]) + 'Grpd']
        else:
            groupedBalancedSpikes = grpSpksNpArr
            condDescriptors = ['%d' % stN for stN in stimsUse]
            

        ## Start here
        xDimBestAll = []
        xDimScoreBestAll = []
        gpfaPrepAll = []
        normalGpfaScoreAndErrAll = []
        loadedSaved = []
        from time import time
        tSt = time()
        for idx, (grpSpks, condDesc) in enumerate(zip(groupedBalancedSpikes,condDescriptors)):
            print("** Training GPFA for condition %d/%d **" % (idx+1, len(groupedBalancedSpikes)))
            gpfaPrep = GPFA(grpSpks, firingRateThresh=firingRateThresh,sqrtSpks=sqrtSpikes)
            gpfaPrepAll.append(gpfaPrep)

            # we first want to check if the variables of interest have been presaved...
            fullOutputPath = outputPath / "gpfa" / ("run" + str(signalDescriptor)) / ("cond" + str(condDesc))
            fullOutputPath.mkdir(parents=True, exist_ok = True)
            preSavedDataPath = fullOutputPath / "procRes.npz"
            if preSavedDataPath.exists():
                try:
                    gpfaSaved = np.load(preSavedDataPath, allow_pickle=True)
                    xDimBestAll.append(gpfaSaved['xDimBest'])
                    xDimScoreBestAll.append(gpfaSaved['xDimScoreBest'])
                    normalGpfaScoreAndErrAll.append(gpfaSaved['normalGpfaScoreAndErr'])
                    gpfaPrep.dimOutput = gpfaSaved['dimOutput'][()]
                    gpfaPrep.testInds = gpfaSaved['testInds']
                    gpfaPrep.trainInds = gpfaSaved['trainInds']
                except KeyError as e:
                    normalGpfaScoreAndErrAll.append([])
                    xDimBestAll.append([])
                    xDimScoreBestAll.append([])
                    loadedSaved.append(False)
                else:
                    loadedSaved.append(True)
            else:
                normalGpfaScoreAndErrAll.append([])
                xDimBestAll.append([])
                xDimScoreBestAll.append([])
                loadedSaved.append(False)

            if not loadedSaved[-1]:

                for idxXdim, xDim in enumerate(xDimTest):
                    print("Testing dimensionality %d. Left to test: " % xDim + (str(xDimTest[idxXdim+1:]) if idxXdim+1<len(xDimTest) else "none"))


                    try:
                        

                        estParams, seqTrainNew, seqTestNew = gpfaPrep.runGpfaInMatlab(eng=eng, fname=fullOutputPath, runDescriptor = signalDescriptor, condDescriptor = condDesc, crossvalidateNum=crossvalidateNum, xDim=xDim)
                    except Exception as e:
                        if type(e) is engine.MatlabExecutionError:
                            print(e)
                            continue
                        else:
                            raise(e)

            print("GPFA training for condition %d/%d done" % (idx+1, len(groupedBalancedSpikes)))
        print("All GPFA training done in %d seconds" % (time()-tSt))
                        
        lblLLErr = 'LL err over folds'
        lblLL= 'LL mean over folds'
        cvApproach = "logLikelihood"
        shCovThresh = 0.95
        for idx, gpfaPrep in enumerate(gpfaPrepAll):
            if loadedSaved[idx]:
                continue # avoid wrapping everything in if statement...
            print("** Crossvalidating and plotting GPFA for condition %d/%d **" % (idx+1, len(gpfaPrepAll)))
            normalGpfaScore, normalGpfaScoreErr, reducedGpfaScore = gpfaPrep.crossvalidatedGpfaError(eng=eng, approach = cvApproach)
            normalGpfaScoreAndErr = [normalGpfaScore, normalGpfaScoreErr]
            normalGpfaScoreAndErrAll[idx] = normalGpfaScoreAndErr
            # best xDim is our lowest error from the normalGpfaScore... for now...
            if cvApproach is "logLikelihood":
                xDimScoreBest = xDimTest[np.argmax(normalGpfaScore)]
            elif cvApproach is "squaredError":
                xDimScoreBest = xDimTest[np.argmin(normalGpfaScore)]
            
            Cparams = [prm['C'] for prm in gpfaPrep.dimOutput[xDimScoreBest]['allEstParams']]
            shEigs = [np.flip(np.sort(np.linalg.eig(C.T @ C)[0])) for C in Cparams]
            percAcc = np.stack([np.cumsum(eVals)/np.sum(eVals) for eVals in shEigs])
            
            meanPercAcc = np.mean(percAcc, axis=0)
            stdPercAcc = np.std(percAcc, axis = 0)
            xDimBest = np.where(meanPercAcc>shCovThresh)[0][0]+1
            xDimBestAll[idx] = xDimBest
            xDimScoreBestAll[idx] = xDimScoreBest

            # save the computation!
            fullOutputPath = outputPath / "gpfa" / ("run" + str(signalDescriptor)) / ("cond" + str(condDescriptors[idx]))
            preSavedDataPath = fullOutputPath / "procRes.npz"
            print("Saving output...")
            t = time()
            np.savez(preSavedDataPath, xDimScoreBest=xDimScoreBest, xDimBest = xDimBest, dimOutput=gpfaPrep.dimOutput,
                testInds = gpfaPrep.testInds, trainInds=gpfaPrep.trainInds, normalGpfaScoreAndErr=normalGpfaScoreAndErr)

            tElapse = time()-t
            print("Output saved in %d seconds" % tElapse)
            
        if plotInfo is not None:
            for idx, gpfaPrep in enumerate(gpfaPrepAll):
                normalGpfaScore = normalGpfaScoreAndErrAll[idx][0]
                normalGpfaScoreErr = normalGpfaScoreAndErrAll[idx][1]

                if cvApproach is "logLikelihood":
                    xDimScoreBest = xDimTest[np.argmax(normalGpfaScore)]
                elif cvApproach is "squaredError":
                    xDimScoreBest = xDimTest[np.argmin(normalGpfaScore)]
                
                Cparams = [prm['C'] for prm in gpfaPrep.dimOutput[xDimScoreBest]['allEstParams']]
                shEigs = [np.flip(np.sort(np.linalg.eig(C.T @ C)[0])) for C in Cparams]
                percAcc = np.stack([np.cumsum(eVals)/np.sum(eVals) for eVals in shEigs])
                meanPercAcc = np.mean(percAcc, axis=0)
                stdPercAcc = np.std(percAcc, axis = 0)
                xDimBest = np.where(meanPercAcc>shCovThresh)[0][0]+1
                
                axScore = plotInfo['axScore']
                
                if cvApproach is "logLikelihood":
                    axScore.plot(xDimTest,normalGpfaScore, label = lblLL)
                    axScore.fill_between(xDimTest, normalGpfaScore-normalGpfaScoreErr, normalGpfaScore+normalGpfaScoreErr, alpha=0.2, label =lblLLErr)
                    axScore.set_title(description)
                elif cvApproach is "squaredError":
                    axScore.plot(xDimTest,normalGpfaScore, label = 'Summed GPFA Error Over Folds')
                    axScore.plot(np.arange(len(reducedGpfaScore))+1, reducedGpfaScore, label='Summed Reduced GPFA Error Over Folds')
                    axScore.set_title(description)
                # axScore.legend(loc='upper right')
                
                axDim = plotInfo['axDim']
                axDim.set_xlabel("num dims")
                
                axDim.plot(np.arange(len(meanPercAcc))+1,meanPercAcc)
                axDim.fill_between(np.arange(len(meanPercAcc))+1, meanPercAcc-stdPercAcc,meanPercAcc+stdPercAcc,alpha=0.2)
                
                axDim.axvline(xDimBest, linestyle='--')
                axDim.axhline(shCovThresh, linestyle='--')
                
                axScore.axvline(xDimBest, linestyle='--')

                xlD = axDim.get_xlim()
                xlS = axScore.get_xlim()
                axScore.set_xlim(xmin=np.min([xlD[0],xlS[0]]), xmax=np.max([xlD[1],xlS[1]]))

                lblLL = None
                lblLLErr = None
                grpSpks = groupedBalancedSpikes[idx]
                

                for cValUse in [0]:# range(crossvalidateNum):
#                    meanTraj = gpfaPrep.projectTrajectory(gpfaPrep.dimOutput[xDimScoreBest]['allEstParams'][cValUse], labelMeans[idx][None,:,:], sqrtSpks = sqrtSpikes)
                    meanTraj = gpfaPrep.projectTrajectory(gpfaPrep.dimOutput[xDimScoreBest]['allEstParams'][cValUse], np.stack(labelMeans), sqrtSpks = sqrtSpikes)
                    shuffTraj = gpfaPrep.shuffleGpfaControl(gpfaPrep.dimOutput[xDimScoreBest]['allEstParams'][cValUse], cvalTest = cValUse, sqrtSpks = sqrtSpikes)

                    rowsPlt = 2
                    if tmValsStartBest.size and tmValsEndBest.size:
                        colsPlt = np.ceil(xDimBest/rowsPlt)*2 # align to start and to end...
                    else:
                        colsPlt = np.ceil(xDimBest/rowsPlt) # aligning to only one of them...
                    axesStart = []
                    axesEnd = []
                    axVals = np.empty((0,4))
                    figSep = plt.figure()
                    figSep.suptitle(description + " cond " + str(uniqueTargAngleDeg[idx].tolist()) + "")
                    if xDimBest>2:
                        figTraj = plt.figure()
                        axStartTraj = plt.subplot(1,3,1,projection='3d')
                        axEndTraj = plt.subplot(1,3,2,projection='3d')
                        axAllTraj = plt.subplot(1,3,3,projection='3d')
                        figTraj.suptitle(description + " cond " + str(uniqueTargAngleDeg[idx].tolist()) + "")
                    elif xDimBest>1:
                        figTraj = plt.figure()
                        axStartTraj = plt.subplot(1,3,1)
                        axEndTraj = plt.subplot(1,3,2)
                        axAllTraj = plt.subplot(1,3,3)
                        figTraj.suptitle(description + " cond " + str(uniqueTargAngleDeg[idx].tolist()) + "")
                
                    plt.figure()
                    plt.imshow(np.abs(gpfaPrep.dimOutput[xDimScoreBest]['allEstParams'][cValUse]['C']),aspect="auto")
                    plt.title('C matrix (not orth)')
                    plt.colorbar()


                    # seqTrainNewAll = gpfaPrep.dimOutput[xDimScoreBest]['seqsTrainNew'][cValUse]
                    # seqTrainOrthAll = [sq['xorth'] for sq in seqTrainNewAll]
                    # seqTrainConcat = np.concatenate(seqTrainOrthAll, axis = 1)
                    # plt.figure()
                    # plt.plot(seqTrainConcat.T)



                    seqTestUse = gpfaPrep.dimOutput[xDimScoreBest]['seqsTestNew'][cValUse] # just use the first one...
                    lblStartTraj = 'traj start'
                    lblEndTraj = 'traj end'
                    lblDelayStart = 'delay start'
                    lblDelayEnd = 'delay end'
                    lblNoDelayStart = 'delay start outside traj'
                    lblNoDelayEnd = 'delay end outside traj'
                    lblNeuralTraj = 'neural trajectories'
                    for k, (sq, tstInd) in enumerate(zip(seqTestUse,gpfaPrep.testInds[cValUse])):
                        # if k>5:
                            # break
                        gSp = grpSpks[tstInd]
                        # sq = {}
                        # sq['xorth'] = np.concatenate((sq2['xorth'][1:], sq2['xorth'][:1]), axis=0)
                        # gSp = grpSpks[tstInd]
                        # print(gSp.alignmentBins.shape)
                        if tmValsStartBest.size:
                            startZeroBin = gSp.alignmentBins[0]
                            fstBin = 0
                            tmBeforeStartZero = (fstBin-startZeroBin)*binSize
                            tmValsStart = tmValsStartBest[tmValsStartBest>=tmBeforeStartZero]
                        else:
                            tmValsStart = np.ndarray((0,0))
                            
                        if tmValsEndBest.size:
                            # Only plot what we have data for...
                            endZeroBin = gSp.alignmentBins[1]
                            # going into gSp[0] because gSp might be dtype object instead of the float64,
                            # so might have trials within the array, not accessible without
                            lastBin = gSp[0].shape[0] # note: same as sq['xorth'].shape[1]
                            timeAfterEndZero = (lastBin-endZeroBin)*binSize
                            tmValsEnd = tmValsEndBest[tmValsEndBest<timeAfterEndZero]
                        else:
                            tmValsEnd = np.ndarray((0,0))
                            
                        if xDimBest>2:
                            plt.figure(figTraj.number)
                            if tmValsStart.size:
                                axStartTraj.plot(sq['xorth'][0,:tmValsStart.shape[0]], sq['xorth'][1,:tmValsStart.shape[0]], sq['xorth'][2,:tmValsStart.shape[0]],
                                       color=colorset[idx,:], linewidth=0.4)                            
                                axStartTraj.plot([sq['xorth'][0,0]], [sq['xorth'][1,0]], [sq['xorth'][2,0]], 'o',
                                       color=colorset[idx,:], linewidth=0.4, label=lblStartTraj, markeredgecolor='black')
                                axStartTraj.plot([sq['xorth'][0,tmValsStart.shape[0]-1]], [sq['xorth'][1,tmValsStart.shape[0]-1]], [sq['xorth'][2,tmValsStart.shape[0]-1]], '>',
                                       color=colorset[idx,:], linewidth=0.4, label=lblEndTraj, markeredgecolor='black')
                                
                                # marking the alginment point here
                                if np.min(tmValsStart) <= 0 and np.max(tmValsStart) >= 0:
                                    alignX = np.interp(0, tmValsStart, sq['xorth'][0,:tmValsStart.shape[0]])
                                    alignY = np.interp(0, tmValsStart, sq['xorth'][1,:tmValsStart.shape[0]])
                                    alignZ = np.interp(0, tmValsStart, sq['xorth'][2,:tmValsStart.shape[0]])
                                    axStartTraj.plot([alignX], [alignY], [alignZ], '*', color='green', label = 'delay start alignment')
                                else:
                                    axStartTraj.plot([np.nan], [np.nan], '*', color='green', label =lblNoDelayStart)
                                
                                axStartTraj.set_title('Start')
                                axStartTraj.set_xlabel('gpfa 1')
                                axStartTraj.set_ylabel('gpfa 2')
                                axStartTraj.set_zlabel('gpfa 3')
                                axStartTraj.legend()
                            
                            if tmValsEnd.size:
                                axEndTraj.plot(sq['xorth'][0,-tmValsEnd.shape[0]:], sq['xorth'][1,-tmValsEnd.shape[0]:], sq['xorth'][2,-tmValsEnd.shape[0]:],
                                           color=colorset[idx,:], linewidth=0.4)
                                axEndTraj.plot([sq['xorth'][0,-tmValsEnd.shape[0]]], [sq['xorth'][1,-tmValsEnd.shape[0]]], [sq['xorth'][2,-tmValsEnd.shape[0]]], 'o',
                                       color=colorset[idx,:], linewidth=0.4, label=lblEndTraj, markeredgecolor='black')
                                axEndTraj.plot([sq['xorth'][0,-1]], [sq['xorth'][1,-1]], [sq['xorth'][2,-1]], '>',
                                       color=colorset[idx,:], linewidth=0.4, label=lblEndTraj, markeredgecolor='black')
                                
                                # marking the alginment point here
                                if np.min(tmValsEnd) <= 0 and np.max(tmValsEnd) >= 0:
                                    alignX = np.interp(0, tmValsEnd, sq['xorth'][0,-tmValsEnd.shape[0]:])
                                    alignY = np.interp(0, tmValsEnd, sq['xorth'][1,-tmValsEnd.shape[0]:])
                                    alignZ = np.interp(0, tmValsEnd, sq['xorth'][2,-tmValsEnd.shape[0]:])
                                    axEndTraj.plot([alignX], [alignY], [alignZ], '*', color='red', label = 'delay end alignment')
                                else:
                                    axEndTraj.plot([np.nan], [np.nan], '*', color='red', label =lblNoDelayEnd)
                                
                                axEndTraj.set_title('End')
                                axEndTraj.set_xlabel('gpfa 1')
                                axEndTraj.set_ylabel('gpfa 2')
                                axEndTraj.set_zlabel('gpfa 3')
                                axEndTraj.legend()
                                
                            if tmValsStart.size and tmValsEnd.size:
                                axAllTraj.plot(sq['xorth'][0,:], sq['xorth'][1,:], sq['xorth'][2,:],
                                           color=colorset[idx,:], linewidth=0.4)
                                axAllTraj.plot([sq['xorth'][0,0]], [sq['xorth'][1,0]], [sq['xorth'][2,0]], 'o',
                                           color=colorset[idx,:], linewidth=0.4, label=lblStartTraj, markeredgecolor='black')
                                axAllTraj.plot([sq['xorth'][0,-1]], [sq['xorth'][1,-1]], [sq['xorth'][2,-1]], '>',
                                           color=colorset[idx,:], linewidth=0.4, label=lblEndTraj, markeredgecolor='black')
                                
                                
                                # marking the alginment points here
                                # the binSize and start are detailed here, so we can find the rest of time
                                lenT = sq['xorth'].shape[1]
                                allTimesAlignStart = np.arange(tmValsStart[0], tmValsStart[0]+binSize*lenT, binSize)
                                if np.min(allTimesAlignStart) <= 0 and np.max(allTimesAlignStart) >= 0:
                                    alignXStart = np.interp(0, allTimesAlignStart, sq['xorth'][0,:])
                                    alignYStart = np.interp(0, allTimesAlignStart, sq['xorth'][1,:])
                                    alignZStart = np.interp(0, allTimesAlignStart, sq['xorth'][2,:])
                                    axAllTraj.plot([alignXStart], [alignYStart], [alignZStart], '*', color='green', label =lblDelayStart)
                                else:
                                    axAllTraj.plot([np.nan], [np.nan], '*', color='green', label =lblNoDelayStart)
                                    
                                # gotta play some with the end to make sure 0 is aligned from the end as in tmValsEnd
                                allTimesAlignEnd = np.arange(tmValsEnd[-1]-binSize*(lenT-1), tmValsEnd[-1]+binSize/10, binSize)
                                if np.min(allTimesAlignEnd) <= 0 and np.max(allTimesAlignEnd) >= 0:
                                    alignXEnd = np.interp(0, allTimesAlignEnd, sq['xorth'][0,:])
                                    alignYEnd = np.interp(0, allTimesAlignEnd, sq['xorth'][1,:])
                                    alignZEnd = np.interp(0, allTimesAlignEnd, sq['xorth'][2,:])
                                    axAllTraj.plot([alignXEnd], [alignYEnd], [alignZEnd], '*', color='red', label = lblDelayEnd)
                                else:
                                    axAllTraj.plot([np.nan], [np.nan], '*', color='red', label =lblNoDelayEnd)
                                    
                                        
                                
                                axAllTraj.set_title('All')
                                axAllTraj.set_xlabel('gpfa 1')
                                axAllTraj.set_ylabel('gpfa 2')
                                axAllTraj.set_zlabel('gpfa 3')
                                axAllTraj.legend()

                        elif xDimBest>1:
                            plt.figure(figTraj.number)
                            if tmValsStart.size:
                                axStartTraj.plot(sq['xorth'][0,:tmValsStart.shape[0]], sq['xorth'][1,:tmValsStart.shape[0]],
                                       color=colorset[idx,:], linewidth=0.4)
                                axStartTraj.plot(sq['xorth'][0,0], sq['xorth'][1,0], 'o',
                                       color=colorset[idx,:], linewidth=0.4, label='traj start', markeredgecolor='black')
                                axStartTraj.plot(sq['xorth'][0,tmValsStart.shape[0]-1], sq['xorth'][1,tmValsStart.shape[0]-1], '>',
                                       color=colorset[idx,:], linewidth=0.4, label='traj end', markeredgecolor='black')
                                
                                # marking the alginment point here
                                if np.min(tmValsStart) <= 0 and np.max(tmValsStart) >= 0:
                                    alignX = np.interp(0, tmValsStart, sq['xorth'][0,:tmValsStart.shape[0]])
                                    alignY = np.interp(0, tmValsStart, sq['xorth'][1,:tmValsStart.shape[0]])
                                    axStartTraj.plot(alignX, alignY, '*', color='green', label =lblDelayStart)
                                else:
                                    axStartTraj.plot(np.nan, '*', color='green', label =lblNoDelayStart)
                                axStartTraj.set_title('Start')
                                axStartTraj.set_xlabel('gpfa 1')
                                axStartTraj.set_ylabel('gpfa 2')
                                axStartTraj.legend()
                            
                            if tmValsEnd.size:
                                axEndTraj.plot(sq['xorth'][0,-tmValsEnd.shape[0]:], sq['xorth'][1,-tmValsEnd.shape[0]:],
                                           color=colorset[idx,:], linewidth=0.4)
                                axEndTraj.plot(sq['xorth'][0,-tmValsEnd.shape[0]], sq['xorth'][1,-tmValsEnd.shape[0]], 'o',
                                       color=colorset[idx,:], linewidth=0.4, label='traj start', markeredgecolor='black')
                                axEndTraj.plot(sq['xorth'][0,-1], sq['xorth'][1,-1], '>',
                                       color=colorset[idx,:], linewidth=0.4, label='traj end', markeredgecolor='black')
                                
                                # marking the alginment point here
                                if np.min(tmValsEnd) <= 0 and np.max(tmValsEnd) >= 0:
                                    alignX = np.interp(0, tmValsEnd, sq['xorth'][0,-tmValsEnd.shape[0]:])
                                    alignY = np.interp(0, tmValsEnd, sq['xorth'][1,-tmValsEnd.shape[0]:])
                                    axEndTraj.plot(alignX, alignY, '*', color='red', label =lblDelayEnd)
                                else:
                                    axEndTraj.plot(np.nan, '*', color='red', label =lblNoDelayEnd)

                                axEndTraj.set_title('End')
                                axEndTraj.set_xlabel('gpfa 1')
                                axEndTraj.set_ylabel('gpfa 2')
                                axEndTraj.legend()
                            
                            if tmValsStart.size and tmValsEnd.size:
                                axAllTraj.plot(sq['xorth'][0,:], sq['xorth'][1,:],
                                           color=colorset[idx,:], linewidth=0.4)
                                axAllTraj.plot(sq['xorth'][0,0], sq['xorth'][1,0], 'o',
                                           color=colorset[idx,:], linewidth=0.4, label=lblStartTraj, markeredgecolor='black')
                                axAllTraj.plot(sq['xorth'][0,-1], sq['xorth'][1,-1], '>',
                                           color=colorset[idx,:], linewidth=0.4, label=lblEndTraj, markeredgecolor='black')
                                
                                # marking the alginment points here
                                # the binSize and start are detailed here, so we can find the rest of time
                                lenT = sq['xorth'].shape[1]
                                allTimesAlignStart = np.arange(tmValsStart[0], tmValsStart[0]+binSize*lenT, binSize)
                                if np.min(allTimesAlignStart) <= 0 and np.max(allTimesAlignStart) >= 0:
                                    alignXStart = np.interp(0, allTimesAlignStart, sq['xorth'][0,:])
                                    alignYStart = np.interp(0, allTimesAlignStart, sq['xorth'][1,:])
                                    axAllTraj.plot(alignXStart, alignYStart, '*', color='green', label =lblDelayStart)
                                else:
                                    axAllTraj.plot(np.nan, '*', color='green', label =lblNoDelayStart)
                                    
                                # gotta play some with the end to make sure 0 is aligned from the end as in tmValsEnd
                                allTimesAlignEnd = np.arange(tmValsEnd[-1]-binSize*(lenT-1), tmValsEnd[-1]+binSize/10, binSize)
                                if np.min(allTimesAlignEnd) <= 0 and np.max(allTimesAlignEnd) >= 0:
                                    alignXEnd = np.interp(0, allTimesAlignEnd, sq['xorth'][0,:])
                                    alignYEnd = np.interp(0, allTimesAlignEnd, sq['xorth'][1,:])
                                    axAllTraj.plot(alignXEnd, alignYEnd, '*', color='red', label =lblDelayEnd)
                                else:
                                    axAllTraj.plot(np.nan, '*', color='red', label =lblNoDelayEnd)
                                    
                                        
                                
                                axAllTraj.set_title('All')
                                axAllTraj.set_xlabel('gpfa 1')
                                axAllTraj.set_ylabel('gpfa 2')
                                axAllTraj.legend()

                        
                        if True:
                            pltNum = 1
                            plt.figure(figSep.number)
                            plt.suptitle(description + " cond " + str(uniqueTargAngleDeg[idx].tolist()) + "")
                            for dimNum, dim in enumerate(sq['xorth']):
                                dimNum = dimNum+1 # first is 1-dimensional, not zero-dimensinoal
                                #we'll only plot the xDimBest dims...
                                if dimNum > xDimBest:
                                    break
                                
                                if tmValsStart.size:
                                    if len(axesStart) + len(axesEnd) <rowsPlt*colsPlt:
                                        axesHere = figSep.add_subplot(rowsPlt, colsPlt, pltNum)
                                        axesStart.append(axesHere)
                                        if pltNum == 1: 
                                            axesHere.set_title("dim " + str(dimNum) + " periStart")
                                        else:
                                            axesHere.set_title("d"+str(dimNum)+"S")

                                        if pltNum <= (xDimBest - colsPlt):
                                            axesHere.set_xticklabels('')
                                            axesHere.xaxis.set_visible(False)
                                            axesHere.spines['bottom'].set_visible(False)
                                        else:  
                                            axesHere.set_xlabel('time (ms)')

                                        if (pltNum % colsPlt) != 1:
                                            axesHere.set_yticklabels('')
                                            axesHere.yaxis.set_visible(False)
                                            axesHere.spines['left'].set_visible(False)
                                    else:
                                        if tmValsEnd.size:
                                            axesHere = axesStart[int((pltNum-1)/2)]
                                        else:
                                            axesHere = axesStart[int(pltNum-1)]    
                                        plt.axes(axesHere)
                                
                                    plt.plot(tmValsStart, dim[:tmValsStart.shape[0]], color=colorset[idx,:], linewidth=0.4, label=lblNeuralTraj)
                                
                                
                                
                                    axVals = np.append(axVals, np.array(axesHere.axis())[None, :], axis=0)
                                    pltNum += 1
                                
                                if tmValsEnd.size:
                                    if len(axesStart) + len(axesEnd) <rowsPlt*colsPlt:
                                        axesHere = figSep.add_subplot(rowsPlt, colsPlt, pltNum)
                                        axesEnd.append(axesHere)
                                        if pltNum == colsPlt:
                                            axesHere.set_title("dim " + str(dimNum) + " periEnd")
                                        else:
                                            axesHere.set_title("d"+str(dimNum)+"E")

                                        if pltNum <= (xDimBest - colsPlt):
                                            axesHere.set_xticklabels('')
                                            axesHere.xaxis.set_visible(False)
                                            axesHere.spines['bottom'].set_visible(False)
                                        else:  
                                            axesHere.set_xlabel('time (ms)')

                                        if (pltNum % colsPlt) != 1:
                                            axesHere.set_yticklabels('')
                                            axesHere.yaxis.set_visible(False)
                                            axesHere.spines['left'].set_visible(False)
                                    else:
                                        if tmValsStart.size:
                                            axesHere = axesEnd[int(pltNum/2-1)]
                                        else:
                                            axesHere = axesEnd[int(pltNum-1)]
                                        plt.axes(axesHere)
                            
                                    plt.plot(tmValsEnd, dim[-tmValsEnd.shape[0]:], color=colorset[idx,:], linewidth=0.4, label=lblNeuralTraj)

                                    axVals = np.append(axVals, np.array(axesHere.axis())[None, :], axis=0)
                                    pltNum += 1

                            axesHere.legend()

                        lblStartTraj = None
                        lblEndTraj = None
                        lblDelayStart = None
                        lblDelayEnd = None
                        lblNoDelayStart = None
                        lblNoDelayEnd = None
                        lblNeuralTraj = None


                    # wow is this ugly, but now I'm gonna plot the shuffles
                    lblShuffle = 'shuffles'
                    for sq in shuffTraj:
                        pltNum = 1
                        plt.figure(figSep.number)
                        for dimNum, dim in enumerate(sq['xorth']):
                            dimNum = dimNum+1 # first is 1-dimensional, not zero-dimensinoal
                            #we'll only plot the xDimBest dims...
                            if dimNum > xDimBest:
                                break
                            
                            if tmValsStart.size:
                                if len(axesStart) + len(axesEnd) <rowsPlt*colsPlt:
                                    axesHere = figSep.add_subplot(rowsPlt, colsPlt, pltNum)
                                    axesStart.append(axesHere)
                                    if pltNum == 1:
                                        axesHere.set_title("dim " + str(dimNum) + " periStart")
                                    else:
                                        axesHere.set_title("d"+str(dimNum)+"S")

                                    if pltNum <= (xDimBest - colsPlt):
                                        axesHere.set_xticklabels('')
                                        axesHere.xaxis.set_visible(False)
                                        axesHere.spines['bottom'].set_visible(False)
                                    else:  
                                        axesHere.set_xlabel('time (ms)')

                                    if (pltNum % colsPlt) != 1:
                                        axesHere.set_yticklabels('')
                                        axesHere.yaxis.set_visible(False)
                                        axesHere.spines['left'].set_visible(False)
                                else:
                                    if tmValsEnd.size:
                                        axesHere = axesStart[int((pltNum-1)/2)]
                                    else:
                                        axesHere = axesStart[int(pltNum-1)]    
                                    plt.axes(axesHere)
                            
                                plt.plot(tmValsStart, dim[:tmValsStart.shape[0]], color=[0.5,0.5,0.5], linewidth=0.2, alpha=0.5, label=lblShuffle)
                            
                            
                            
                                axVals = np.append(axVals, np.array(axesHere.axis())[None, :], axis=0)
                                pltNum += 1
                            
                            if tmValsEnd.size:
                                if len(axesStart) + len(axesEnd) <rowsPlt*colsPlt:
                                    axesHere = figSep.add_subplot(rowsPlt, colsPlt, pltNum)
                                    axesEnd.append(axesHere)
                                    if pltNum == colsPlt:
                                        axesHere.set_title("dim " + str(dimNum) + " periEnd")
                                    else:
                                        axesHere.set_title("d"+str(dimNum)+"E")

                                    if pltNum <= (xDimBest - colsPlt):
                                        axesHere.set_xticklabels('')
                                        axesHere.xaxis.set_visible(False)
                                        axesHere.spines['bottom'].set_visible(False)
                                    else:  
                                        axesHere.set_xlabel('time (ms)')

                                    if (pltNum % colsPlt) != 1:
                                        axesHere.set_yticklabels('')
                                        axesHere.yaxis.set_visible(False)
                                        axesHere.spines['left'].set_visible(False)
                                else:
                                    if tmValsStart.size:
                                        axesHere = axesEnd[int(pltNum/2-1)]
                                    else:
                                        axesHere = axesEnd[int(pltNum-1)]
                                    plt.axes(axesHere)
                        
                                plt.plot(tmValsEnd, dim[-tmValsEnd.shape[0]:], color=[0.5,0.5,0.5], linewidth=0.2, alpha=0.5, label=lblShuffle)                 
                                axVals = np.append(axVals, np.array(axesHere.axis())[None, :], axis=0)
                                pltNum += 1

                        axesHere.legend()
                        lblShuffle = None


                    # just keeps getting uglier, but this is for the mean trajectory...
                    lblMn = 'mean traj per cond'
                    for condNum, mnTraj in enumerate(meanTraj):
                        pltNum = 1
                        plt.figure(figSep.number)
                        for dimNum, dim in enumerate(mnTraj['xorth']):
                            dimNum = dimNum+1 # first is 1-dimensional, not zero-dimensinoal
                            #we'll only plot the xDimBest dims...
                            if dimNum > xDimBest:
                                break

                            
                            if tmValsStart.size:
                                if len(axesStart) + len(axesEnd) <rowsPlt*colsPlt:
                                    axesHere = figSep.add_subplot(rowsPlt, colsPlt, pltNum)
                                    axesStart.append(axesHere)
                                    if pltNum == 1:
                                        axesHere.set_title("dim " + str(dimNum) + " periStart")
                                    else:
                                        axesHere.set_title("d"+str(dimNum)+"S")

                                    if pltNum <= (xDimBest - colsPlt):
                                        axesHere.set_xticklabels('')
                                        axesHere.xaxis.set_visible(False)
                                        axesHere.spines['bottom'].set_visible(False)
                                    else:  
                                        axesHere.set_xlabel('time (ms)')

                                    if (pltNum % colsPlt) != 1:
                                        axesHere.set_yticklabels('')
                                        axesHere.yaxis.set_visible(False)
                                        axesHere.spines['left'].set_visible(False)
                                else:
                                    if tmValsEnd.size:
                                        axesHere = axesStart[int((pltNum-1)/2)]
                                    else:
                                        axesHere = axesStart[int(pltNum-1)]    
                                    plt.axes(axesHere)
                            
#                                plt.plot(tmValsStart, dim[:tmValsStart.shape[0]], color=colorset[idx,:][0,0,0], linewidth=1,label=lblMn)
                                plt.plot(tmValsStart, dim[:tmValsStart.shape[0]], color=colorset[condNum,:], linewidth=1,label=lblMn)
                            
                            
                            
                                axVals = np.append(axVals, np.array(axesHere.axis())[None, :], axis=0)
                                pltNum += 1
                            
                            if tmValsEnd.size:
                                if len(axesStart) + len(axesEnd) <rowsPlt*colsPlt:
                                    axesHere = figSep.add_subplot(rowsPlt, colsPlt, pltNum)
                                    axesEnd.append(axesHere)
                                    if pltNum == colsPlt:
                                        axesHere.set_title("dim " + str(dimNum) + " periEnd")
                                    else:
                                        axesHere.set_title("d"+str(dimNum)+"E")

                                    if pltNum <= (xDimBest - colsPlt):
                                        axesHere.set_xticklabels('')
                                        axesHere.xaxis.set_visible(False)
                                        axesHere.spines['bottom'].set_visible(False)
                                    else:  
                                        axesHere.set_xlabel('time (ms)')

                                    if (pltNum % colsPlt) != 1:
                                        axesHere.set_yticklabels('')
                                        axesHere.yaxis.set_visible(False)
                                        axesHere.spines['left'].set_visible(False)

                                else:
                                    if tmValsStart.size:
                                        axesHere = axesEnd[int(pltNum/2-1)]
                                    else:
                                        axesHere = axesEnd[int(pltNum-1)]
                                    plt.axes(axesHere)
                        
                                plt.plot(tmValsEnd, dim[-tmValsEnd.shape[0]:], color=[0,0,0], linewidth=1, label=lblMn)                 
                                axVals = np.append(axVals, np.array(axesHere.axis())[None, :], axis=0)
                                pltNum += 1

                        axesHere.legend() #legend on the last plot...
                        lblMn = None
                    
                    ymin = np.min(np.append(0, np.min(axVals, axis=0)[2]))
                    ymax = np.max(axVals, axis=0)[3]
                    for ax in axesStart:
                        ax.set_ylim(bottom = ymin, top = ymax )
                        plt.axes(ax)
                        plt.axvline(x=0, linestyle=':', color='black')
                        plt.axhline(y=0, linestyle=':', color='black')
#                        xl = ax.get_xlim()
#                        yl = ax.get_ylim()
#                        ax.axhline(y=0,xmin=xl[0],xmax=xl[1], linestyle = ':', color='black')
#                        ax.axvline(x=0,ymin=xl[0],ymax=xl[1], linestyle = ':', color='black')
                        ax.spines['right'].set_visible(False)
                        ax.spines['top'].set_visible(False)
                        
                    for ax in axesEnd:
                        ax.set_ylim(bottom = ymin, top = ymax )
                        plt.axes(ax)
                        plt.axvline(x=0, linestyle=':', color='black')
                        plt.axhline(y=0, linestyle=':', color='black')
#                        xl = ax.get_xlim()
#                        yl = ax.get_ylim()
#                        ax.axhline(y=0,xmin=xl[0],xmax=xl[1], linestyle = ':', color='black')
#                        ax.axvline(x=0,ymin=xl[0],ymax=xl[1], linestyle = ':', color='black')
                        ax.spines['right'].set_visible(False)
                        ax.spines['top'].set_visible(False)
        
            

        gpfaPrepDimOutputAll = [gpfa.dimOutput for gpfa in gpfaPrepAll]
        gpfaTestIndsAll = [gpfa.testInds for gpfa in gpfaPrepAll]
        gpfaTrainIndsAll = [gpfa.trainInds for gpfa in gpfaPrepAll]

        return xDimBestAll, xDimScoreBestAll, gpfaPrepDimOutputAll, gpfaTestIndsAll, gpfaTrainIndsAll

#%% implementation of some np functions

def implements(numpy_function):
    """Register an __array_function__ implementation for MyArray objects."""
    def decorator(func):
        HANDLED_FUNCTIONS[numpy_function] = func
        return func
    return decorator

# implementation of concatenate for BinnedSpikeSet objects
@implements(np.concatenate)
def concatenate(arrays, axis=0, out=None):
    binSizes = np.stack([arr.binSize for arr in arrays])
    if not np.all(binSizes == binSizes[0]):
        raise(Exception("Can't (shouldn't?) concatenate BinnedSpikeSets with different bin sizes!"))
    from itertools import chain
    if type(arrays) is list:
        arrayNd = [arr.view(np.ndarray) for arr in arrays]
    elif  type(arrays) is BinnedSpikeSet and arrays.dtype=='object':
        # we're converting from a BinnedSpikeSet trl x chan object whose elements
        # are spike trains, into a # trials length list of single trials 1 x trl x chan
        arrayNd = [np.stack(arr.tolist())[None,:,:] for arr in arrays]
    elif type(arrays) is BinnedSpikeSet: # and it's not an object
        arrayNd = arrays.view(np.ndarray)
    else:
        raise(Exception("Concatenating anything but lists of BinnedSpikeSet not implemented yet!"))
        
    concatTrad = np.concatenate(arrayNd, axis = axis)
    if axis == 0:
        concatStartInit = [arr.start for arr in arrays]
        concatStart = list(chain(*concatStartInit))
        concatEndInit = [arr.end for arr in arrays]
        concatEnd = list(chain(*concatEndInit))
        
        unLabelKeys = np.unique([arr.labels.keys() for arr in arrays])
        for key in unLabelKeys:
            # might need to return to the below if keys start having to be lists...
            keyVals = np.stack([arr.labels[key] for arr in arrays])
            newLabels[key] = keyVals # list(chain(*keyVals))
            
        concatAlBinsInit = np.stack([arr.alignmentBins for arr in arrays])
        concatAlBins = list(chain(*concatAlBinsInit))
        return BinnedSpikeSet(concatTrad, start=concatStart, end=concatEnd, binSize=binSizes[0], labels=newLabels, alignmentBins=concatAlBins)
    elif axis < 3:
        return BinnedSpikeSet(concatTrad, start=None, end=None, binSize=binSizes[0], labels=None, alignmentBins=None)
    else:
        raise(Exception("Not really sure what concatenating in more than 3 dims means... let's talk later"))
    
@implements(np.copyto)
def copyto(arr, fill_val, **kwargs):
    return np.copyto(np.array(arr), fill_val, **kwargs)

@implements(np.where)
def where(condition, x=None, y=None):
    
    if x is None and y is None:
        return np.where(np.array(condition))
    else:
        return np.where(np.array(condition), np.array(x), np.array(y))

