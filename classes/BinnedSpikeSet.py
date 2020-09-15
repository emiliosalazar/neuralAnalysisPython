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
from methods.plotUtils.GpfaPlotMethods import visualizeGpfaResults
#from mayavi import mlab


from copy import copy
import itertools

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
                
                self.alignmentBins = self.alignmentBins[obj._new_label_index] if self.alignmentBins is not None else None
                self.start = self.start[obj._new_label_index] if self.start is not None else None
                self.end = self.end[obj._new_label_index] if self.end is not None else None
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
        try:
            return super().__getitem__(item)
        except IndexError as e:
#            print(item)
            if self.dtype == 'object' and type(item) is tuple and len(item)==3:
#                breakpoint()
                trlUse = item[0]
                trl = self[trlUse]
                if len(trl.shape) < len(self.shape): # we lost a dimension
                    npArr = np.stack(trl)[item[1],item[2]]
                else:
                    npArr = np.stack([np.stack(bT)[item[1],item[2]] for bT in trl])

                newBinSet = BinnedSpikeSet(
                                npArr,
                                start=self[trlUse].start.copy(),
                                end=self[trlUse].end.copy(),
                                binSize = self.binSize,
                                labels = self[trlUse].labels.copy(),
                                alignmentBins = self[trlUse].alignmentBins.copy(),
                                units=self.units
                                )
                return newBinSet
            raise(e)
    
    def __array_function__(self, func, types, args, kwargs):
        if func not in HANDLED_FUNCTIONS:
            def unwrapArgs(args):
                argOut = []
                for ar in args:
                    if type(ar) is BinnedSpikeSet:
                        argOut.append(np.array(ar))
                    elif type(ar) is list:
                        argOut.append(list(unwrapArgs(ar)))
                    elif type(ar) is tuple:
                        argOut.append(tuple(unwrapArgs(ar)))
                    else:
                        # we're not going to take care of any other type of iterators for now...
                        argOut.append(ar)

                return argOut

            newArgs = unwrapArgs(args)

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
        if self.size == 0:
            # the average of nothing is nothing
            if self.shape[2] == 0:
                print('no timepoints, huh?')
                return np.empty((self.shape[0], 0)).view(BinnedSpikeSet)
            return np.empty(self.shape[:2]).view(BinnedSpikeSet)
        else:
            if self.dtype == 'object':
                out = self.copy()
                out[:] = [np.average(np.stack(trl), axis=1) for trl in out]
                # at this point it's no longer an object, so take on the type
                # it shoulda been
                out = out.astype(self[0,0].dtype)
            else:
                out = np.average(self, axis=2)
            return out

    def timeSum(self):
        if self.size == 0:
            # the sum of nothing is nothing
            if self.shape[2] == 0:
                print('no timepoints, huh?')
                return np.empty((self.shape[0], 0)).view(BinnedSpikeSet)
            return np.empty(self.shape[:2]).view(BinnedSpikeSet)
        else:
            if self.dtype == 'object':
                out = self.copy()
                out[:] = [np.sum(np.stack(trl), axis=1) for trl in out]
                # at this point it's no longer an object, so take on the type
                # it shoulda been
                out = out.astype(self[0,0].dtype)
            else:
                out = np.sum(self, axis=2)
            return out
    
    def timeStd(self):
        return np.std(self, axis=2, ddof=1) # ddof=1 makes this sample standard deviation #np.sqrt(self.timeAverage()) #
    
    def timeSpikeCount(self):
        return np.sum(self, axis=2)
    
    def trialAverage(self):
        if self.size > 0:
            if self.dtype == 'object':
                if np.unique(self.alignmentBins, axis=0).shape[0] == 1:
                    trlsAsObj = [np.stack(trl) for trl in self]
                    mxLen = np.max([trl.shape[1] for trl in trlsAsObj])
                    padSelf = np.stack([np.pad(trl, ((0,0),(0,mxLen-trl.shape[1])), constant_values=np.nan) for trl in trlsAsObj])
                    #NOTE check about labels field...
                    return np.nanmean(padSelf,axis=0)
                else:
                    raise Exception('Can only trial average trials of different lengths if they are all aligned!')
            else:
                return np.average(self, axis=0)
        else:
            return np.empty_like(self.shape[1:]).view(BinnedSpikeSet)
        
    def trialStd(self):
        if self.size > 0:
            return np.std(self, axis=0, ddof=1)
        else:
            return None

    def trialSem(self):
        if self.size > 0:
            return np.std(self, axis=0, ddof=1)/np.sqrt(self.shape[0])
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
#            print("baseline subtracting time trace with no groups")
            overallBaseline = self.trialAverage()
            spikesUse = self.copy()
            if self.dtype == 'object':
                # though the below check would've broken trialAverage() anyway
                if np.unique(self.alignmentBins, axis=0).shape[0] == 1:
                    for trl in spikesUse:
                        for ch, bsCh in zip(trl, overallBaseline):
                            ch[:] = ch[:] - bsCh[:ch.shape[0]]
            else:
                spikesUse = spikesUse[:,:] - overallBaseline[:,:] # this indexing is important for the moment to keep the labels field correct...
        else:
            unq, unqCnts = np.unique(labels, return_counts=True, axis=0)
            spikesUse = self.copy()
            overallBaseline = []
            if self.dtype == 'object':
                if np.unique(self.alignmentBins, axis=0).shape[0] == 1:
                    for lbl in unq:
                        lblTrls = np.all(labels==lbl,axis=lbl.ndim)
                        lblBaseline = self[lblTrls].trialAverage()
                        overallBaseline.append(lblBaseline)
                        # can't quite get broadcasting to work here, but that's fine
                        for trl in spikesUse[lblTrls]:
                            for ch, bsCh in zip(trl, lblBaseline):
                                ch[:] = ch[:] - bsCh[:ch.shape[0]]
                else:
                    raise Exception("Can only baseline subtract when trials have dissimilar lengths if they are identically aligned!")
            else:
                for lbl in unq:
                    overallBaseline.append(self[np.all(labels==lbl,axis=lbl.ndim)].trialAverage())
                    spikesUse[np.all(labels==lbl,axis=lbl.ndim), :] = self[np.all(labels==lbl,axis=lbl.ndim), :] - self[np.all(labels==lbl,axis=lbl.ndim), :].trialAverage()
            # had this before and not sure if the extra : is necessary...
#                spikesUse[np.all(labels==lbl,axis=lbl.ndim), :, :] = self[np.all(labels==lbl,axis=lbl.ndim), :, :] - self[np.all(labels==lbl,axis=lbl.ndim), :, :].trialAverage()

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
            chanFirst = hzBinned.swapaxes(0,1)
            avgFiringChan = np.array([np.mean(np.hstack(chan), axis=0) for chan in chanFirst])
        else:
            avgFiringChan = hzBinned.timeAverage().trialAverage()
        
        return avgFiringChan

    def stdFiringRateByChannel(self):
        if self.units != 'Hz':
            hzBinned = self.convertUnitsTo(units='Hz')
        else:
            hzBinned = self

        if self.dtype == 'object':
            chanFirst = hzBinned.swapaxes(0,1)
            avgFiringStdChan = [np.std(np.hstack(chan), axis=1) for chan in chanFirst]
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
            chanFirst = cntBinned.swapaxes(0,1)
            avgCountChan = np.stack([np.sum(np.hstack(chan), axis=1) for chan in chanFirst])
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
            chanFirst = cntBinned.swapaxes(0,1)
            avgCountStdChan = np.stack([np.std(np.hstack(chan), axis=1) for chan in chanFirst])
            breakpoint() # because I'm not sure this is correct... NOTE want sum before std?
        else:
            # we're taking a sum over the bins before taking the standard
            # deviation... so it's the standard devation of spike counts in a
            # trial no matter what... but that's also how the functions named so hah!
            avgCountStdChan = cntBinned.sum(axis=2).trialStd()
        
        return avgCountStdChan

    # this is different from chaining timeAverage and trialAverage together
    # when there are different numbers of bins in different trials
    def avgValByChannelOverBins(self):
        if self.dtype == 'object':
            chanFirst = self.swapaxes(0,1)
            avgValChan = np.stack([np.mean(np.hstack(chan), axis=1) for chan in chanFirst])
            breakpoint() # because I'm not sure this is correct...
        else:
            avgValChan = self.timeAverage().trialAverage()
        
        return avgValChan

    def stdValByChannelOverBins(self):
        if self.size == 0:
            stdValChan = np.empty(self.shape[1])
        elif self.dtype == 'object':
            chanFirst = self.swapaxes(0,1)
            stdValChan = np.stack([np.std(np.hstack(chan), axis=1) for chan in chanFirst])
            breakpoint() # because I'm not sure this is correct...
        else:
            numChannels = self.shape[1]
            stdValChan = self.swapaxes(0,1).reshape(numChannels,-1).std(axis=1)
        
        return stdValChan

    def stdValByChannelOverTrials(self):
        if self.size == 0:
            stdValChan = np.empty(self.shape[1])
        elif self.dtype == 'object':
            chanFirst = self.swapaxes(0,1)
            stdValChan = np.stack([np.std(np.hstack(chan), axis=1) for chan in chanFirst])
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
        avgValChan = self.avgValByChannelOverBins()
        stdValChan = self.stdValByChannelOverBins()

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
    def channelsAboveThresholdFiringRate(self, firingRateThresh, asLogical = False): #low firing thresh in Hz
        avgFiringChan = self.avgFiringRateByChannel()
        
        highFiringChannels = avgFiringChan>firingRateThresh
        if not asLogical:
            highFiringChannels = np.where(highFiringChannels)[0]
        
        return self[:,highFiringChannels], highFiringChannels

    # note that this will always calculate the *spike count* fano factor
    def channelsBelowThresholdFanoFactor(self, fanoFactorThresh, asLogical = False):
        if self.units != 'count':
            cntBinned = self.convertUnitsTo(units='count')
        else:
            cntBinned = self

        fanoFactorChans = cntBinned.fanoFactorByChannel()

        lowFanoFactorChans = fanoFactorChans < fanoFactorThresh
        if not asLogical:
            lowFanoFactorChans = np.where(lowFanoFactorChans)[0]

        return self[:,lowFanoFactorChans], lowFanoFactorChans

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
        # now is check which channels had any firing during >meanPerc-3*stdPerc
        # of trials, after removing firing rates that were >3*std of firing
        # rates of that channel.

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

        # only select neurons that responded in more than meanPerc-stdPerc of trials
        respPerc = numRespTrialsByChan/self.shape[0]
        respPercMean = respPerc.mean(axis=0)
        respPercStd = respPerc.std(axis=0)
        consistentChans = respPerc>=(respPercMean-3*respPercStd) # must be closer than three std

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
        # this syntax is for when newBinSize is an array because each trial is changed to a different length
        if np.any((newBinSize/self.binSize) % 1): 
            raise Exception('BinnedSpikeSet:BadBinSize', 'New bin size must be whole multiple of previous bin size!')
            return None

        if len(self.shape) == 3:
            trlLen = self.shape[2]
        elif len(self.shape) == 2:
            trlLen = [bnSpTrl[0].shape[0] for bnSpTrl in self] 

        if np.any((newBinSize/self.binSize) > trlLen):
            raise Exception('BinnedSpikeSet:BadBinSize', 'New bin size larger than the length of each trials!')
            return None
        if np.any((newBinSize/self.binSize) % trlLen):
            raise Exception('BinnedSpikeSet:BadBinSize', "New bin size doesn't evenly divide trajectory. Avoiding splitting to not have odd last bin.")

        # we're assuming no tuples are in here...
        if np.asarray(newBinSize).size == 1:
            newBinSize = np.repeat(newBinSize, len(self.start))

        # this really shouldn't happen, but it checks for bad data in some sense...
        if np.asarray(newBinSize).size != self.start.size: 
            raise Exception('BinnedSpikeSet:BadNumberOfNewBins', "newBinSize should have either 1 element or as many elements as the number of trials")
        
        # allows for both one new bin size and multiple new bin sizes (one per trial, or this will fail earlier when doing the size checks)
        binTrialIterator = zip(self.start, self.end, newBinSize)
        binsPerTrial = [np.arange(st, en+nbs/20, nbs)-st for st, en, nbs in binTrialIterator]
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
        residualSpikesGoodChans = residualSpikes
        chansGood = np.arange(residualSpikesGoodChans.shape[1])
#        residualSpikesGoodChans, chansGood = residualSpikes.channelsNotRespondingSparsely(zeroRate = np.array(overallBaseline)[unqInv])
#        residualSpikesGoodChans, chansGood = residualSpikesGoodChans.removeInconsistentChannelsOverTrials(zeroRate = np.array(overallBaseline)[unqInv])
        
        
        chanFirst = residualSpikesGoodChans.swapaxes(0, 1) # channels are now main axis, for sampling in reshape below
        chanFlat = chanFirst.reshape((residualSpikesGoodChans.shape[1], -1)) # chan are now rows, time is cols
        
        # let's remove trials that are larger than 3*std
        stdChanResp = np.std(chanFlat, axis=1)
#        chanMask = np.abs(chanFlat) > (3*stdChanResp[:,None]) # < 0
        chanMask = np.abs(chanFlat) < 0 # mask nothing... for now
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
                    minCorr = np.minimum(np.nanmin(corrLblSpks.flat), minCorr)
                    maxCorr = np.maximum(np.nanmax(corrLblSpks.flat), maxCorr)
                    
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
                minCorr = np.nanmin(corrSpksPerCond.flat)
                maxCorr = np.nanmax(corrSpksPerCond.flat)
                
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
    
    def prepareGpfaOrFa(self, sqrtSpikes = False,
             labelUse = 'stimulusMainLabel', condNums=1, combineConds = False, overallFiringRateThresh = 1, perConditionGroupFiringRateThresh = 1, balanceConds = True, 
             computeResiduals = True):
              

        if type(self) is list:
            bnSpksCheck = np.concatenate(self, axis=2).view(BinnedSpikeSet)
            bnSpksCheck.labels = {}
            bnSpksCheck.labels[labelUse] = np.stack([bnSp.labels[labelUse] for bnSp in self])
            binSize = self[0].binSize
            colorset = self[0].colorset
        elif self.dtype == 'object':
            bnSpksCheck = np.concatenate(self, axis=2)
            bnSpksCheck.labels = {}
            binSize = self.binSize
            colorset = self.colorset
        else:
            bnSpksCheck = self
            binSize = self.binSize
            colorset = self.colorset
        
        # note that overallFiringRateThresh should be a negative number if we're inputing residuals...
        _, chIndsUseFR = bnSpksCheck.channelsAboveThresholdFiringRate(firingRateThresh=overallFiringRateThresh)
        
        if balanceConds:
            # This seed-setting is important for knowing we're getting the same
            # inds even when reloading/re-crossvaling data
            initSt = np.random.get_state()
            np.random.seed(seed=0)
            trlIndsUseLabel = self.balancedTrialInds(labels=self.labels[labelUse])
            np.random.set_state(initSt)
        else:
            trlIndsUseLabel = range(len(self))
        
        if type(self) is list:
            binnedSpikeHighFR = [bnSp[:,chIndsUseFR, :] for bnSp in self]
            binnedSpikesUse = np.empty(len(binnedSpikeHighFR), dtype='object')
            binnedSpikesUse[:] = binnedSpikeHighFR
            binnedSpikesUse = binnedSpikesUse[trlIndsUseLabel]
            binnedSpikesUse = BinnedSpikeSet(binnedSpikesUse, binSize = binSize)
            newLabels = np.stack([bnSp.labels[labelUse] for bnSp in binnedSpikesUse])
            binnedSpikesUse.labels[labelUse] = newLabels
        elif self.dtype == 'object':
            binnedSpikeHighFR = self[:,chIndsUseFR]
            binnedSpikesUse = binnedSpikeHighFR[trlIndsUseLabel]
            newLabels = binnedSpikesUse.labels[labelUse]
            if sqrtSpikes:
                binnedSpikesUse = np.sqrt(binnedSpikesUse)
        else:
            binnedSpikeHighFR = self[:,chIndsUseFR,:]
            binnedSpikesUse = binnedSpikeHighFR[trlIndsUseLabel]
            newLabels = binnedSpikesUse.labels[labelUse]
            if sqrtSpikes:
                binnedSpikesUse = np.sqrt(binnedSpikesUse)
        
        uniqueTargAngle, trialsPresented = np.unique(newLabels, return_inverse=True, axis=0)
        
        if condNums is None:
            condsUse = np.arange(uniqueTargAngle.shape[0])
        else:
            # we allow the top level to choose random condNums if it wants them
            condsUse = np.stack(condNums)

        
        uniqueTargAngleDeg = uniqueTargAngle*180/np.pi
        
        
        if perConditionGroupFiringRateThresh > 0 and combineConds:
            breakpoint()
            # so there's a confusion if both of these are true... especially
            # when computeResiduals is in the mix--if computeResiduals is True,
            # we want to compute the residuals on a per-condition basis
            # (right?) and then combine the conditions... but in order to
            # compute the residuals, we first want to remove the low firing
            # rate channels... but if we do that the conditions might not be
            # combined correctly because some channels for some conditions
            # might be kicked off. In any case, I'm not sure when we'll run
            # into this, but when we do I will revisit.
            raise Exception("Not sure how to order these operations, read the comment about revisiting")
        
        grpSpksNpArr, _ = binnedSpikesUse.groupByLabel(newLabels, labelExtract=uniqueTargAngle[condsUse]) # extract one label...

        if perConditionGroupFiringRateThresh > 0:
            for idx, grpSpks in enumerate(grpSpksNpArr):
                grpSpks, _ = grpSpks.channelsAboveThresholdFiringRate(firingRateThresh=perConditionGroupFiringRateThresh)
                grpSpksNpArr[idx] = grpSpks

            perConditionGroupFiringRateThresh = -1

        if computeResiduals:
            for idx, grpSpks in enumerate(grpSpksNpArr):
                residGrpSpks, grpMn = grpSpks.baselineSubtract()

                grpSpksNpArr[idx] = residGrpSpks
                


        if combineConds and (condNums is None or len(condNums)>1):
            grpSpksNpArr = [BinnedSpikeSet(np.concatenate(grpSpksNpArr, axis=0), binSize = grpSpksNpArr[0].binSize)] # grpSpksNpArr
            condDescriptors = ['s' + '-'.join(['%d' % stN for stN in condsUse]) + 'Grpd']
        else:
            grpSpksNpArr = grpSpksNpArr
            condDescriptors = ['%d' % stN for stN in condsUse]


        # simple change of variable name
        groupedBalancedSpikes = grpSpksNpArr
            

        ## Start here
        # This is no longer a for loop for below because there's only one
        # xDimTest, but I'm keeping these variables to avoid changing much
        # of the downstream code for now heh...
#        idxXdim = 0
#        xDim = xDimTest[idxXdim]

        return groupedBalancedSpikes, condDescriptors, condsUse

    def fa(self, groupedBalancedSpikes, outputPathToConditions, condDescriptors, xDim, labelUse, crossvalidateNum = 4):
        assert isinstance(xDim, int), "Must only provide one integer xDim at a time"
        from classes.FA import FA
        from time import time
        tSt = time()

        condSavePaths = []
        xDimBestAll = []
        xDimScoreBestAll = []
        gpfaPrepAll = []
        loadedSaved = []
        faScoreAll = []
        faPrepAll = []

        cvApproach = "logLikelihood"
        # use Williamson 2016 method with svCovThresh (should be 0.95 or
        # 95%...) threshold to find dimensionality
        shCovThresh = 0.95
        for idx, grpSpks  in enumerate(groupedBalancedSpikes):
            # note that this is a conservative check, as trl < xDim might be
            # True, but trl*tmPt > xDim (where tmPt is the third dimension for
            # grpSpks. I'll get to this problem when it arrives, as it could
            # require dealing with object arrays...
            if np.any(np.array(grpSpks.shape[:2]) < xDim): 
                breakpoint()
                grpSpkShape = np.array(grpSpks.shape)
                print("Either not enough trials (%d) or not enough channels (%d) to train a %d-dimensional FA. Maximum possible dimensionality: %d" % (grpSpkShape[0], grpSpkShape[1], xDim, np.min(np.array(grpSpkShape))))
                continue
            else:
                print("** Training and crossvalidating FA for condition %d/%d **" % (idx+1, len(groupedBalancedSpikes)))
                faPrep = FA(grpSpks, crossvalidateNum=crossvalidateNum)
                faPrepAll.append(faPrep)

                faScoreCond = np.empty((1,crossvalidateNum))


                fullOutputPath = outputPathToConditions /  ("cond" + str(condDescriptors[idx]))
#                try:
                faScoreCond[0, :] = faPrep.runFa( numDim=xDim, gpfaResultsPath = fullOutputPath )[0]
#                except Exception as e:
#                    if e.args[0] == "FA:NumObs":
#                        print(e.args[1])
#                        faScoreCond[0, :] = np.repeat(np.nan, crossvalidateNum)
#                    else:
#                        raise(e)

                faScoreAll.append(faScoreCond)

                print("FA training for condition %d/%d done" % (idx+1, len(groupedBalancedSpikes)))
                        
                preSavedDataPath = fullOutputPath / ("faResultsDim%d.npz" % xDim)
                np.savez(preSavedDataPath, dimOutput=faPrep.dimOutput[xDim], testInds = faPrep.testInds, trainInds=faPrep.trainInds, score=faScoreCond[0,:], alignmentBins = grpSpks.alignmentBins, condLabel = grpSpks[0].labels[labelUse], binSize = grpSpks.binSize  )

#            normalFaScore = faPrep.crossvalidatedFaError(approach = cvApproach, dimsCrossvalidate = xDim)
#            normalFaScoreCurr = normalFaScoreAll[idx]
#            normalFaScoreCurr[indsCVal, :] = normalFaScore
#            normalFaScoreAll[idx] = normalFaScoreCurr

            # find the best xdim based on the crossvalidation approach
#            faScoreMn = faScoreCond.mean(axis=1)
#            if cvApproach is "logLikelihood":
#                xDimScoreBest = xDimTest[np.argmax(faScoreMn)]
#            elif cvApproach is "squaredError":
#                xDimScoreBest = xDimTest[np.argmin(faScoreMn)]
#
#            
#            Cparams = [prm['C'] for prm in faPrep.dimOutput[xDimScoreBest]['allEstParams']]
#            shEigs = [np.flip(np.sort(np.linalg.eig(C.T @ C)[0])) for C in Cparams]
#            percAcc = np.stack([np.cumsum(eVals)/np.sum(eVals) for eVals in shEigs])
#            
#            meanPercAcc = np.mean(percAcc, axis=0)
#            stdPercAcc = np.std(percAcc, axis = 0)
#            xDimBest = np.where(meanPercAcc>shCovThresh)[0][0]+1
#            xDimBestAll.append(xDimBest)
#            xDimScoreBestAll.append(xDimScoreBest)

        print("All FA training done in %d seconds" % (time()-tSt))
        # using getattr here allows me to preset gpfaPrepAll to an empty array
        # and have the outputs be empty arrays if nothing happens (i.e. an
        # error with gpfa--or a handled mistake where there are too few neurons
        # for the desired dimension, for example
        faPrepDimOutputAll = [getattr(fa, 'dimOutput', None) for fa in faPrepAll]
        faTestIndsAll = [getattr(fa, 'testInds', None) for fa in faPrepAll]
        faTrainIndsAll = [getattr(fa, 'trainInds', None) for fa in faPrepAll]
#        faScoreAll = np.stack(faScoreAll)

        return faPrepDimOutputAll, faTestIndsAll, faTrainIndsAll, faScoreAll

    def gpfa(self, groupedBalancedSpikes, outputPathToConditions, condDescriptors, xDim, labelUse, crossvalidateNum = 4, forceNewGpfaRun = False):
        from classes.GPFA import GPFA
        from time import time

        condSavePaths = []
        rankConvInfo = []
        xDimBestAll = []
        xDimScoreBestAll = []
        gpfaPrepAll = []
        loadedSaved = []
        gpfaScoreAll = []
        tSt = time()

        cvApproach = "logLikelihood"
        # use Williamson 2016 method with svCovThresh (should be 0.95 or
        # 95%...) threshold to find dimensionality
        shCovThresh = 0.95

        for idx, (grpSpks, condDesc) in enumerate(zip(groupedBalancedSpikes,condDescriptors)):
            # note that this is a conservative check, as trl < xDim might be
            # True, but trl*tmPt > xDim (where tmPt is the third dimension for
            # grpSpks. I'll get to this problem when it arrives, as it could
            # require dealing with object arrays...
            if np.any(np.array(grpSpks.shape[:2]) < xDim): 
                grpSpkShape = np.array(grpSpks.shape)
                print("Either not enough trials (%d) or not enough channels (%d) to train a %d-dimensional GPFA. Maximum possible dimensionality: %d" % (grpSpkShape[0], grpSpkShape[1], xDim, np.min(np.array(grpSpkShape))))
                gpfaPrepAll.append([]) # to stay on track with the condition number
                continue
            else:
                print("** Training GPFA for condition %d/%d **" % (idx+1, len(groupedBalancedSpikes)))
                gpfaPrep = GPFA(grpSpks)
                gpfaPrepAll.append(gpfaPrep)

                # we first want to check if the variables of interest have been presaved...
                fullOutputPath = outputPathToConditions /  ("cond" + str(condDesc))
                fullOutputPath.mkdir(parents=True, exist_ok = True)
#                preSavedDataPath = fullOutputPath / "procRes.npz"
#                if False:#preSavedDataPath.exists():
                # loading for each dimension and concatenating
                gpfaScoreCond = np.empty((1,crossvalidateNum))

#                print("Testing/loading dimensionality %d. Left to test: " % xDim + (str(xDimTest[idxXdim+1:]) if idxXdim+1<len(xDimTest) else "none"))
                preSavedDataPath = fullOutputPath / ("gpfaResultsDim%d.npz" % xDim)
                condSavePaths.append(preSavedDataPath)
                if preSavedDataPath.exists() and not forceNewGpfaRun:
                    try:
                        gpfaDimSaved = np.load(preSavedDataPath, allow_pickle=True)

                        gpfaScoreCond[0,:] = gpfaDimSaved['score' if 'score' in gpfaDimSaved else 'normalGpfaScore']
#                            normalGpfaScoreCond[idxXdim, :] = np.repeat(np.nan, crossvalidateNum) # placeholder
#                            normalGpfaScoreAndErrAll = [nGSE.append(savNGSE) for nGSE, savNGSE in zip(normalGpfaScoreAndErrAll, gpfaSaved['normalGpfaScoreAndErr'])]
                        gpfaPrep.dimOutput[xDim] = gpfaDimSaved['dimOutput'][()]
                        # make sure right number of channels are in here...
#                            if gpfaPrep.dimOutput[xDim]['allEstParams'][0]['C'].shape[0] != grpSpks.shape[1]:
#                                raise(Exception("ChanSavedProcDiff"))
                        # These are getting replaced every time. I think... that's... fine...
                        gpfaPrep.testInds = gpfaDimSaved['testInds']
                        gpfaPrep.trainInds = gpfaDimSaved['trainInds']

                    except (KeyError, Exception) as e:
                        loadedDimCond = False
                        if type(e) is Exception:
                            if e.args[0] == "ChanSavedProcDiff":
                                print("Diff chans expected vs saved: (dimTest, expChNum, savedChNum) - (%d, %d, %d)" % (xDim, self.shape[1], gpfaPrep.dimOutput[xDim]['allEstParams'][0]['C'].shape[0]))
                            else:
                                raise(e)

                        print("Error loading GPFA: %s", e.args[0])
                        # take care to clear out any information that was added
                        gpfaScoreCond[0, :] = np.repeat(np.nan, crossvalidateNum) # placeholder
                        gpfaPrep.dimOutput.pop(xDim, None);
                        gpfaPrep.testInds = None
                        gpfaPrep.trainInds = None

                    else:
                        loadedDimCond = True
                else:
                    gpfaScoreCond[0, :] = np.repeat(np.nan, crossvalidateNum) # placeholder
                    loadedDimCond = False

                if not loadedDimCond or forceNewGpfaRun:
                    try:
                        gpfaPrep.runGpfaInMatlab(fname=fullOutputPath,  crossvalidateNum=crossvalidateNum, xDim=xDim, forceNewGpfaRun = forceNewGpfaRun);
#                            if gpfaPrep.dimOutput[xDim]['allEstParams'][0]['C'].shape[0] != grpSpks.shape[1]:
#                                print("Diff chans expected vs saved by MATLAB: (dimTest, expChNum, savedChNum) - (%d, %d, %d)" % (xDim, self.shape[1], gpfaPrep.dimOutput[xDim]['allEstParams'][0]['C'].shape[0]))
#                                print("Trying to force rerun of GPFA in case inputs have changed")
#                                gpfaPrep.runGpfaInMatlab(fname=fullOutputPath,  crossvalidateNum=crossvalidateNum, xDim=xDim, forceNewGpfaRun = True);
#                                if gpfaPrep.dimOutput[xDim]['allEstParams'][0]['C'].shape[0] != grpSpks.shape[1]:
#                                    # MAKE SURE you're not asking GPFA to do any channel filtering
#                                    breakpoint()
                    except Exception as e:
                        from matlab import engine
                        if type(e) is engine.MatlabExecutionError:
                            print(e)
#                                breakpoint()
                            continue
                        else:
                            raise(e)
#                        gpfaSaved = np.load(preSavedDataPath, allow_pickle=True)
#                        xDimBestAll.append(gpfaSaved['xDimBest'])
#                        xDimScoreBestAll.append(gpfaSaved['xDimScoreBest'])
#                gpfaScoreAll.append(gpfaScoreCond)
                loadedSaved = loadedDimCond
                converged = [estParam['converge'] for estParam in gpfaPrep.dimOutput[xDim]['allEstParams']]
                trainIsFullRank = gpfaPrep.dimOutput[xDim]['fullRank']
                rankConvInfo.append([trainIsFullRank,converged])
#                        gpfaPrep.dimOutput = gpfaSaved['dimOutput'][()]
#                        gpfaPrep.testInds = gpfaSaved['testInds']
#                        gpfaPrep.trainInds = gpfaSaved['trainInds']
#                else:
#                    normalGpfaScoreAndErrAll.append([])
#                    xDimBestAll.append([])
#                    xDimScoreBestAll.append([])
#                    loadedSaved.append(False)

            print("GPFA training for condition %d/%d done" % (idx+1, len(groupedBalancedSpikes)))
                        
#        for idx, gpfaPrep in enumerate(gpfaPrepAll):
            print("** Crossvalidating GPFA for condition %d/%d **" % (idx+1, len(groupedBalancedSpikes)))
            # find any dimensions that still need crossvalidating
            dimsCVal = []
#            indsCVal = []
            if not loadedSaved:
                dimsCVal.append(xDim)

#            # crossvalidate dimensions as needed
#            if len(dimsCVal) > 0:
                gpfaScore, gpfaScoreErr, reducedGpfaScore = gpfaPrep.crossvalidatedGpfaError(approach = cvApproach, dimsCrossvalidate = dimsCVal)
                gpfaScoreCurr = gpfaScoreCond
                gpfaScoreCurr[0, :] = gpfaScore
#                gpfaScoreCurr[indsCVal, :] = gpfaScore
                gpfaScoreCond = gpfaScoreCurr

            # find the best xdim based on the crossvalidation approach
            gpfaScore = gpfaScoreCond
#            gpfaScoreMn = gpfaScore.mean(axis=1)
#            if cvApproach is "logLikelihood":
#                xDimScoreBest = xDimTest[np.argmax(gpfaScoreMn)]
#            elif cvApproach is "squaredError":
#                xDimScoreBest = xDimTest[np.argmin(gpfaScoreMn)]
#
#            
#            Cparams = [prm['C'] for prm in gpfaPrep.dimOutput[xDimScoreBest]['allEstParams']]
#            shEigs = [np.flip(np.sort(np.linalg.eig(C.T @ C)[0])) for C in Cparams]
#            percAcc = np.stack([np.cumsum(eVals)/np.sum(eVals) for eVals in shEigs])
#            
#            meanPercAcc = np.mean(percAcc, axis=0)
#            stdPercAcc = np.std(percAcc, axis = 0)
#            xDimBest = np.where(meanPercAcc>shCovThresh)[0][0]+1
#            xDimBestAll.append(xDimBest)
#            xDimScoreBestAll.append(xDimScoreBest)

            # save the computation!
            # But save it per dimension! This allows later loading for specific
            # dimension tests, and avoids rewriting previous loads of many
            # dimensions
#            if not np.all([*loadedSaved[idx].values()]):
            if not loadedSaved:
                print("Saving output...")
                t = time()

                fullOutputPath = outputPathToConditions /  ("cond" + str(condDesc))
                preSavedDataPath = fullOutputPath / ("gpfaResultsDim%d.npz" % xDim)
                np.savez(preSavedDataPath, dimOutput=gpfaPrep.dimOutput[xDim], testInds = gpfaPrep.testInds, trainInds=gpfaPrep.trainInds, score=gpfaScore[0,:], alignmentBins = grpSpks.alignmentBins, condLabel = grpSpks.labels[labelUse], binSize = grpSpks.binSize  )

                tElapse = time()-t
                print("Output saved in %d seconds" % tElapse)

        print("All GPFA training/crossvalidating done in %d seconds" % (time()-tSt))
            
        # using getattr here allows me to preset gpfaPrepAll to an empty array
        # and have the outputs be empty arrays if nothing happens (i.e. an
        # error with gpfa--or a handled mistake where there are too few neurons
        # for the desired dimension, for example
        gpfaPrepDimOutputAll = [getattr(gpfa, 'dimOutput', None) for gpfa in gpfaPrepAll]
        gpfaTestIndsAll = [getattr(gpfa, 'testInds', None) for gpfa in gpfaPrepAll]
        gpfaTrainIndsAll = [getattr(gpfa, 'trainInds', None) for gpfa in gpfaPrepAll]

#        return xDimBestAll, xDimScoreBestAll, gpfaPrepDimOutputAll, gpfaTestIndsAll, gpfaTrainIndsAll, condSavePaths
        return gpfaPrepDimOutputAll, gpfaTestIndsAll, gpfaTrainIndsAll, rankConvInfo, condSavePaths

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
    units = np.stack([arr.units for arr in arrays])
    if not np.all(binSizes == binSizes[0]):
        raise(Exception("Can't (shouldn't?) concatenate BinnedSpikeSets with different bin sizes!"))
    if not np.all(units == units[0]):
        raise(Exception("Can't (shouldn't?) concatenate BinnedSpikeSets with different units!"))
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
        
    concatTrad = np.concatenate(arrayNd, axis = axis, out = out)
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
        return BinnedSpikeSet(concatTrad, start=concatStart, end=concatEnd, binSize=binSizes[0], labels=newLabels, alignmentBins=concatAlBins, units = self.units)
    elif axis < 3:
        # TODO for axis = 1 (each channel is its own trial) we can just expand start/end/alignment bins to reflect change
        # TODO for axis = 2 (one trial that's really long) we could just offset start/end/alignmentBins so there's many in one trial that are appropriately offshifted
        return BinnedSpikeSet(concatTrad, start=None, end=None, binSize=binSizes[0], labels={}, alignmentBins=None, units = units[0])
    else:
        raise(Exception("Not really sure what concatenating in more than 3 dims means... let's talk later"))

# implementation of stack for BinnedSpikeSet objects
@implements(np.stack)
def stack(arrays, axis=0, out=None):
    binSizes = np.stack([arr.binSize for arr in arrays])
    units = np.stack([arr.units for arr in arrays])
    if not np.all(binSizes == binSizes[0]):
        raise(Exception("Can't (shouldn't?) concatenate BinnedSpikeSets with different bin sizes!"))
    if not np.all(units == units[0]):
        raise(Exception("Can't (shouldn't?) concatenate BinnedSpikeSets with different units!"))
    from itertools import chain
    if type(arrays) in [list, tuple]:
        # Convert tuples into lists again... np.stack on ndarrays handles them the same, methinks...
        arrayNd = [arr.view(np.ndarray) for arr in arrays]
    elif  type(arrays) is BinnedSpikeSet and arrays.dtype=='object':
        # we're converting from a BinnedSpikeSet trl x chan object whose elements
        # are spike trains, into a # trials length list of single trials 1 x trl x chan
        arrayNd = [np.stack(arr.tolist())[None,:,:] for arr in arrays]
    elif type(arrays) is BinnedSpikeSet: # and it's not an object
        arrayNd = arrays.view(np.ndarray)
    else:
        raise(Exception("Stacking anything but lists of BinnedSpikeSet not implemented yet!"))
        
    stackTrad = np.stack(arrayNd, axis = axis, out=out)
    if axis == 0:
        stackStart = np.stack([arr.start for arr in arrays])
        if np.all(stackStart==None):
            stackStart = None

        stackEnd = np.stack([arr.end for arr in arrays])
        if np.all(stackEnd==None):
            stackEnd = None
        
        unLabelKeys = np.unique([arr.labels.keys() for arr in arrays])
        newLabels = {}
        for key in unLabelKeys:
            # might need to return to the below if keys start having to be lists...
            keyVals = np.stack([arr.labels[list(key)[0]] for arr in arrays])
            newLabels[list(key)[0]] = keyVals # list(chain(*keyVals))
            
        stackAlBins = np.stack([arr.alignmentBins for arr in arrays])
        if np.all(stackAlBins == None):
            stackAlBins = None

        return BinnedSpikeSet(stackTrad, start=stackStart, end=stackEnd, binSize=binSizes[0], labels=newLabels, alignmentBins=stackAlBins, units = units[0])
    elif axis < 3:
        # TODO for axis = 1 (each channel is its own trial) we can just expand start/end/alignment bins to reflect change
        # TODO for axis = 2 (one trial that's really long) we could just offset start/end/alignmentBins so there's many in one trial that are appropriately offshifted
        return BinnedSpikeSet(stackTrad, start=None, end=None, binSize=binSizes[0], labels=None, alignmentBins=None, units = units[0])
    else:
        raise(Exception("Not really sure what concatenating in more than 3 dims means... let's talk later"))

# implementation of squeeze for BinnedSpikeSet objects
@implements(np.squeeze)
def squeeze(array, axis=None):
    binSize = array.binSize
    units = array.units

    from itertools import chain
    if  type(array) is BinnedSpikeSet and array.dtype=='object':
        if axis < 3:
            raise Exception("Can't really squeeze object arrays for the moment...")
        # NOTE I think numpy can deal with object averaging... we'll have to see...
        arrayNd = array.view(np.ndarray)
    elif type(array) is BinnedSpikeSet: # and it's not an object
        arrayNd = array.view(np.ndarray)
    else:
        raise(Exception("Squeezing anything but lists of BinnedSpikeSet not implemented yet!"))
        
    squeezeTrad = np.squeeze(arrayNd, axis = axis)
    
    # note that we're just letting all the metadata be carried forward here,
    # even though axes may have been wiped out... dual reason is that I don't
    # want to take the effort to figure out which axis was wiped out and I also
    # am pretty sure the indexing to get to a singleton dimension will
    # appropriately take care of most of this metadata anyway
    return BinnedSpikeSet(squeezeTrad, start=array.start, end=array.end, binSize=binSize, labels=array.labels, alignmentBins=array.alignmentBins, units = units)

# implementation of average for BinnedSpikeSet objects
@implements(np.average)
def average(array, axis=None, weights=None, returned=False):
    binSize = array.binSize
    units = array.units

    from itertools import chain
    if  type(array) is BinnedSpikeSet and array.dtype=='object':
        if axis < 3:
            raise Exception("Can't really average object arrays for the moment...")
        # NOTE I think numpy can deal with object averaging... we'll have to see...
        arrayNd = array.view(np.ndarray)
    elif type(array) is BinnedSpikeSet: # and it's not an object
        arrayNd = array.view(np.ndarray)
    else:
        raise(Exception("Averaging anything but lists of BinnedSpikeSet not implemented yet!"))
        
    avgTrad, sumOfWeights = np.average(arrayNd, axis = axis, weights=weights, returned = True)
    if axis == 0:
        # Keep start and end vals if they happen to be the same for the entire array... (unlikely, but eh)
        startVals = array.start
        if startVals is not None is not None and len(np.unique(startVals, axis=0)) == 1:
            startVal = startVals[0]
        else:
            startVal = None

        endVals = array.end
        if endVals is not None is not None and len(np.unique(endVals, axis=0)) == 1:
            endVal = endVals[0]
        else:
            endVal = None
        
        unLabelKeys = np.unique(array.labels.keys())[0]
        newLabels = {}
        for key in unLabelKeys:
            # might need to return to the below if keys start having to be lists...
            keyVals = array.labels[key]
            if len(np.unique(keyVals, axis=0)) == 1:
                newLabels[key] = keyVals[0] # list(chain(*keyVals))
            
        # Keep alignment bin if they happen to all be the same (unlikely, but meh)
        alBins = array.alignmentBins
        if alBins is not None and len(np.unique(alBins, axis=0)) == 1:
            alBin = alBins[0]
        else:
            alBin = None

        binSpikeAvgOut = BinnedSpikeSet(avgTrad, start=startVal, end=endVal, binSize=binSize, labels=newLabels, alignmentBins=alBin, units = units)
    elif axis == 1:
        # as long as we're averaging across channels, we can keep everything
        binSpikeAvgOut = BinnedSpikeSet(avgTrad, start=array.start, end=array.end, binSize=binSize, labels=array.labels, alignmentBins=array.alignmentBins, units = units)
    elif axis == 2:
        # when averaging across time, we get rid of most time related
        # metadata... for bin size, though, we treat this return value as a bin
        # across the entire duration--up to the user to understand that it's an
        # average not a count
        binSize = binSize*array.shape[2]
        binSpikeAvgOut = BinnedSpikeSet(avgTrad, start=None, end=None, binSize=None, labels=array.labels, alignmentBins=None, units = units)
    elif axis is None:
        # when averaging the flattened array (which is what None represents),
        # the labels might still make sense, but all time related things are
        # gone... except for the bin size, which we return as the total length
        # of time averaged
        unLabelKeys = np.unique(array.labels.keys())[0]
        newLabels = {}
        for key in unLabelKeys:
            # might need to return to the below if keys start having to be lists...
            keyVals = array.labels[key]
            if len(np.unique(keyVals, axis=0)) == 1:
                newLabels[key] = keyVals[0] # list(chain(*keyVals))

        binSize = binSize*array.shape[2]

        binSpikeAvgOut = BinnedSpikeSet(avgTrad, start=None, end=None, binSize=binSize, labels=newLabels, alignmentBins=None, units = units)
    else:
        raise(Exception("Not really sure what averaging in more than 3 dims means... let's talk later"))

    if returned:
        return binSpikeAvgOut, sumOfWeights
    return binSpikeAvgOut

# implementation of std for BinnedSpikeSet objects
@implements(np.std)
def std(array, axis=None, dtype=None, out=None, ddof = 0, keepdims = np._NoValue):
    binSize = array.binSize
    units = array.units

    from itertools import chain
    if  type(array) is BinnedSpikeSet and array.dtype=='object':
        if axis < 3:
            raise Exception("Can't really std object arrays for the moment...")
        # NOTE I think numpy can deal with object averaging... we'll have to see...
        arrayNd = array.view(np.ndarray)
    elif type(array) is BinnedSpikeSet: # and it's not an object
        arrayNd = array.view(np.ndarray)
    else:
        raise(Exception("Stding anything but lists of BinnedSpikeSet not implemented yet!"))
        
    stdTrad = np.std(arrayNd, axis = axis, dtype=dtype, out=out, ddof = ddof, keepdims = keepdims)
    if axis == 0:
        # Keep start and end vals if they happen to be the same for the entire array... (unlikely, but eh)
        startVals = array.start
        if startVals is not None and len(np.unique(startVals, axis=0)) == 1:
            startVal = startVals[0]
        else:
            startVal = None

        endVals = array.end
        if endVals is not None and len(np.unique(endVals, axis=0)) == 1:
            endVal = endVals[0]
        else:
            endVal = None
        
        unLabelKeys = np.unique(array.labels.keys())[0]
        newLabels = {}
        for key in unLabelKeys:
            # might need to return to the below if keys start having to be lists...
            keyVals = array.labels[key]
            if len(np.unique(keyVals, axis=0)) == 1:
                newLabels[key] = keyVals[0] # list(chain(*keyVals))
            
        # Keep alignment bin if they happen to all be the same (unlikely, but meh)
        alBins = array.alignmentBins
        if alBins is not None and len(np.unique(alBins, axis=0)) == 1:
            alBin = alBins[0]
        else:
            alBin = None

        binSpikeStdOut = BinnedSpikeSet(stdTrad, start=startVal, end=endVal, binSize=binSize, labels=newLabels, alignmentBins=alBin, units = units)
    elif axis == 1:
        # as long as we're averaging across channels, we can keep everything
        binSpikeStdOut = BinnedSpikeSet(stdTrad, start=array.start, end=array.end, binSize=binSize, labels=array.labels, alignmentBins=array.alignmentBins, units = units)
    elif axis == 2:
        # when averaging across time, we get rid of most time related
        # metadata... for bin size, though, we treat this return value as a bin
        # across the entire duration--up to the user to understand that it's an
        # average not a count
        binSize = binSize*array.shape[2]
        binSpikeStdOut = BinnedSpikeSet(stdTrad, start=None, end=None, binSize=None, labels=array.labels, alignmentBins=None, units = units)
    elif axis is None:
        # when averaging the flattened array (which is what None represents),
        # the labels might still make sense, but all time related things are
        # gone... except for the bin size, which we return as the total length
        # of time averaged
        unLabelKeys = np.unique(array.labels.keys())[0]
        newLabels = {}
        for key in unLabelKeys:
            # might need to return to the below if keys start having to be lists...
            keyVals = array.labels[key]
            if len(np.unique(keyVals, axis=0)) == 1:
                newLabels[key] = keyVals[0] # list(chain(*keyVals))

        if array.ndim == 3:
            binSize = binSize*array.shape[2]
        else:
            binSize = binSize

        binSpikeStdOut = BinnedSpikeSet(stdTrad, start=None, end=None, binSize=binSize, labels=newLabels, alignmentBins=None, units = units)
    else:
        raise(Exception("Not really sure what stding in more than 3 dims means... let's talk later"))

    return binSpikeStdOut

@implements(np.copyto)
def copyto(arr, fill_val, **kwargs):
    return np.copyto(np.array(arr), fill_val, **kwargs)

@implements(np.where)
def where(condition, x=None, y=None):
    
    if x is None and y is None:
        return np.where(np.array(condition))
    else:
        return np.where(np.array(condition), np.array(x), np.array(y))

