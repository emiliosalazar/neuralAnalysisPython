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
from methods.plotMethods.GpfaPlotMethods import visualizeGpfaResults
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
            if isinstance(item, (list,slice, int, np.ndarray,range)) or np.issubdtype(type(item), np.integer):
                # print('test')
                self._new_label_index = item
                #setattr(self,'_new_label_index', item)
            elif isinstance(item[0], (list,slice, int, np.ndarray, range)) or np.issubdtype(type(item[0]), np.integer):
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

        # Note: this allows subclasses that don't override
        # __array_function__ to handle BinnedSpikeSet objects
        if not all(issubclass(t, BinnedSpikeSet) for t in types):
            return NotImplemented
        return HANDLED_FUNCTIONS[func](*args, **kwargs)
        
    def copy(self, *args, **kwargs):
        out = BinnedSpikeSet(self.view(np.ndarray).copy(*args, **kwargs), start=self.start, end=self.end, binSize=self.binSize, labels=self.labels.copy() if self.labels is not None else {}, alignmentBins=self.alignmentBins.copy() if self.alignmentBins is not None else None, units=self.units)
        
        
        return out
#%% general methods
        
    def timeAverage(self):
        if self.size == 0:
            # the average of nothing is nothing
            if self.shape[2] == 0:
#                print('no timepoints, huh?')
                # this retains the metadata of labels, etc.
                tmAvg = self[:,0,:]
                tmAvg.binSize = None # since there are no bins, there is no bin size... (matches how np.average is implemented below for BinnedSpikeSet)
                tmAvg.start = None
                tmAvg.end = None
                return tmAvg #np.empty((self.shape[0], 0)).view(BinnedSpikeSet)
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
                unAlignBins = np.unique(self.alignmentBins, axis=0)
                if unAlignBins.shape[0] == 1:
                    # here we're averaging trials whose alignment is in the
                    # *same* place, but whose length might be different (say the
                    # end was cut off differently)
                    trlsAsObj = [np.stack(trl) for trl in self]
                    mxLen = np.max([trl.shape[1] for trl in trlsAsObj])
                    padSelf = np.stack([np.pad(trl, ((0,0),(0,mxLen-trl.shape[1])), constant_values=np.nan) for trl in trlsAsObj])
                    meanOut = np.nanmean(padSelf.view(BinnedSpikeSet), axis=0)
                    meanOut.alignmentBins = unAlignBins[0]
                    return meanOut
                elif np.unique(np.diff(unAlignBins, axis=1)).size == 1:
                    # here we're averaging a bunch of trials whose alignment is
                    # in different places, *but* which have equal spacing (so
                    # maybe some binned spikes start earlier or something, so
                    # the start is cutoff)
                    trlsAsObj = [np.stack(trl) for trl in self]
                    trlAlBinFirst = [trl.alignmentBins[0].astype(int) for trl in self]
                    mxFirstAlign = np.max(trlAlBinFirst)
                    finalAlignmentBins = self[np.argmax(trlAlBinFirst)].alignmentBins
                    padSelfInit = [np.pad(trl, ((0,0), (mxFirstAlign-tABF, 0)), constant_values=np.nan) for trl, tABF in zip(trlsAsObj,trlAlBinFirst)]
                    # whereas in the previous if statement we knew all trials
                    # had the same number of values before the first alignment
                    # bin but not after the final one, here we don't know either
                    # way--so we pad on the end as well in case its needed!
                    mxLen = np.max([trl.shape[1] for trl in padSelfInit])
                    padSelf = np.stack([np.pad(trl, ((0,0), (0, mxLen-trl.shape[1])), constant_values=np.nan) for trl in padSelfInit])
                    meanOut = np.nanmean(padSelf.view(BinnedSpikeSet), axis=0)
                    meanOut.alignmentBins = finalAlignmentBins
                    #NOTE check about labels field...
                    return meanOut
                else:
                    raise ValueError('Can only trial average trials of different lengths if they have equally spaced alignment points!')
            else:
                return np.average(self, axis=0)
        else:
            # this retains the metadata of labels, etc.
            return self[0,:]#np.empty_like(self.shape[1:]).view(BinnedSpikeSet)
        
    def trialStd(self):
        if self.size > 0:
            if self.dtype == 'object':
                unAlignBins = np.unique(self.alignmentBins, axis=0)
                if unAlignBins.shape[0] == 1:
                    # here we're averaging trials whose alignment is in the
                    # *same* place, but whose length might be different (say the
                    # end was cut off differently)
                    trlsAsObj = [np.stack(trl) for trl in self]
                    mxLen = np.max([trl.shape[1] for trl in trlsAsObj])
                    padSelf = np.stack([np.pad(trl, ((0,0),(0,mxLen-trl.shape[1])), constant_values=np.nan) for trl in trlsAsObj])
                    #NOTE check about labels field...
                    return np.nanstd(padSelf,axis=0)
                elif np.unique(np.diff(unAlignBins, axis=1)).size == 1:
                    # here we're averaging a bunch of trials whose alignment is
                    # in different places, *but* which have equal spacing (so
                    # maybe some binned spikes start earlier or something, so
                    # the start is cutoff)
                    trlsAsObj = [np.stack(trl) for trl in self]
                    trlAlBinFirst = [trl.alignmentBins[0].astype(int) for trl in self]
                    mxFirstAlign = np.max(trlAlBinFirst)
                    finalAlignmentBins = self[np.argmax(trlAlBinFirst)].alignmentBins
                    padSelfInit = [np.pad(trl, ((0,0), (mxFirstAlign-tABF, 0)), constant_values=np.nan) for trl, tABF in zip(trlsAsObj,trlAlBinFirst)]
                    # whereas in the previous if statement we knew all trials
                    # had the same number of values before the first alignment
                    # bin but not after the final one, here we don't know either
                    # way--so we pad on the end as well in case its needed!
                    mxLen = np.max([trl.shape[1] for trl in padSelfInit])
                    padSelf = np.stack([np.pad(trl, ((0,0), (0, mxLen-trl.shape[1])), constant_values=np.nan) for trl in padSelfInit])
                    stdOut = np.nanstd(padSelf.view(BinnedSpikeSet), axis=0)
                    stdOut.alignmentBins = finalAlignmentBins
                    #NOTE check about labels field...
                    return stdOut
                else:
                    raise Exception('Can only trial average trials of different lengths if they have equallys spaced alignment points!')
            else:
                return np.std(self, axis=0, ddof=1)
        else:
            return None

    def trialSem(self):
        if self.size > 0:
            return self.trialStd()/np.sqrt(self.shape[0])
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
            elif cnt==minCnt:
                idxUse.append(inds)
            else:
                raise Exception("One of your labels doesn't have the minimum number of samples")
                
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
                from copy import deepcopy
                spikesUse = deepcopy(self) # with an object, you gotta make sure to copy the object dimension >.>
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
            # NOTE this outputs an np.array type while the else outputs a
            # BinnedSpikeSet... I don't think it matters at this level, but in
            # case it does at some point I'm letting it be remembered here...
            avgFiringStdChan = np.array([np.stack([trl.mean() for trl in chan]).std(ddof=1) for chan in chanFirst])

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
#            avgCountChanOrig = np.array([np.sum(np.hstack(chan), axis=0)/chan.shape[0] for chan in chanFirst])
            # NOTE that here I'm returning the PER-BIN sum... this is to even
            # out discrepencies in trial length but kinda lies to the user
            # given the method name >.>
            avgCountChan = np.array([np.stack([trl.mean() for trl in chan]).mean() for chan in chanFirst])
#            breakpoint() # because... I'm lying to the user with this method name >.>
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
#            avgCountStdChan = np.stack([np.std(np.hstack(chan), axis=1) for chan in chanFirst])
            # NOTE that here I'm returning the PER-BIN std... this is to even
            # out discrepencies in trial length but kinda lies to the user
            # given the method name >.>
            avgCountStdChan = np.array([np.hstack([trl for trl in chan]).std(ddof=1) for chan in chanFirst])
#            breakpoint() # because... I'm lying to the user with this method name >.>
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
            chanFlat = np.concatenate(self, axis=2).squeeze()
            avgValChan = chanFlat.mean(axis=1)
        else:
            avgValChan = self.timeAverage().trialAverage()
        
        return avgValChan

    def stdValByChannelOverBins(self):
        if self.size == 0:
            stdValChan = np.empty(self.shape[1])
        elif self.dtype == 'object':
            chanFlat = np.concatenate(self, axis=2).squeeze()
            stdValChan = chanFlat.std(axis=1,ddof=1)
        else:
            numChannels = self.shape[1]
            stdValChan = self.swapaxes(0,1).reshape(numChannels,-1).std(axis=1,ddof=1)
        
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
            breakpoint() # because I'm pretty sure this is wrong since it doesn't average over trials
        
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
    def computeCosTuningCurves(self, label='stimulusMainLabel', plot=False):
        labels = self.labels['stimulusMainLabel']
        groupedSpikes, uniqueTargAngle = self.groupByLabel(labels)
        if np.any(np.abs(uniqueTargAngle)>7): # dumb way to check if radians by seeing if anything > 2*pi...
            uniqueTargAngle = uniqueTargAngle/180*np.pi

        targTmAvgList = [grpSp.convertUnitsTo('Hz').timeAverage().trialAverage() for grpSp in groupedSpikes]
        targTmAndTrlSpkAvgArr = np.stack(targTmAvgList).view(BinnedSpikeSet)
        tmAvgList = [grpSp.convertUnitsTo('Hz').timeAverage() for grpSp in groupedSpikes]
        tmAvgArr = np.vstack(tmAvgList)
        
        targAvgNumSpks = targTmAndTrlSpkAvgArr
        cosTuningCurveParams = {}
        cosTuningCurveParams['targAvgNumSpks'] = targAvgNumSpks

        predictors = np.concatenate((np.ones_like(uniqueTargAngle), np.sin(uniqueTargAngle), np.cos(uniqueTargAngle)), axis=1)
        out = [np.linalg.lstsq(predictors, targAvgNumSpks[:, chan, None], rcond=None)[0] for chan in range(targAvgNumSpks.shape[1])]
        residuals = [np.linalg.lstsq(predictors, targAvgNumSpks[:, chan, None], rcond=None)[1] for chan in range(targAvgNumSpks.shape[1])]
        totalSumSquares = [((chanSpks-chanSpks.mean())**2).sum() for chanSpks in targAvgNumSpks.T]
        r2 = np.array([1-sse/sst for sse, sst in zip(residuals, totalSumSquares)])

        predictorsIndividual = np.vstack([np.repeat(p[None,:], grp.shape[0],axis=0) for p, grp in zip(predictors, groupedSpikes)])
        outInd = [np.linalg.lstsq(predictorsIndividual, chanSpks[:,None], rcond=None)[0] for chanSpks in tmAvgArr.T]
        residInd = [np.linalg.lstsq(predictorsIndividual, chanSpks[:,None], rcond=None)[1] for chanSpks in tmAvgArr.T]
        totalSSInd = [((chanSpks-chanSpks.mean())**2).sum() for chanSpks in tmAvgArr.T]
        r2Ind = np.array([1-sse/sst for sse, sst in zip(residInd, totalSSInd)])
        
        paramsPerChan = np.stack(out).squeeze()
        thBestPerChan = np.arctan2(paramsPerChan[:, 1], paramsPerChan[:, 2])
        modPerChan = np.sqrt(pow(paramsPerChan[:,2], 2) + pow(paramsPerChan[:,1], 2))
        bslnPerChan = paramsPerChan[:, 0]
        
        cosTuningCurveParams['thBestPerChan'] = thBestPerChan
        cosTuningCurveParams['modPerChan'] = modPerChan
        cosTuningCurveParams['bslnPerChan'] = bslnPerChan
        cosTuningCurveParams['r2OnMean'] = r2
        cosTuningCurveParams['r2OnIndTrials'] = r2Ind

        if plot:
            from methods.plotMethods.ScatterBar import scatterBar

            fig, ax = plt.subplots(2,3)
            fig.tight_layout()

            # in the following we grab the 0th third index because those are
            # meant to be bars per group, but there's only one bar per group
            # here!
            scatterXY, _ = scatterBar(thBestPerChan)

            ax[0,0].scatter(*(scatterXY[:, :, 0].T))
            ax[0,0].set_title('$\\theta$ pref')
            ax[0,0].set_ylabel('$\\theta$')

            scatterXY, _ = scatterBar(modPerChan)

            ax[0,1].scatter(*(scatterXY[:, :, 0].T))
            ax[0,1].set_title('modulations')
            ax[0,1].set_ylabel('FR')

            scatterXY, _ = scatterBar(bslnPerChan)

            ax[0,2].scatter(*(scatterXY[:, :, 0].T))
            ax[0,2].set_title('baselines')
            ax[0,2].set_ylabel('FR')

            ax[1,0].scatter(modPerChan, r2)
            ax[1,0].set_title('mod vs R^2')
            ax[1,0].set_xlabel('modulation')
            ax[1,0].set_ylabel('R^2')

            ax[1,1].scatter(bslnPerChan, r2)
            ax[1,1].set_title('baseline vs R^2')
            ax[1,1].set_xlabel('baseline FR')
            ax[1,1].set_ylabel('R^2')

            ax[1,2].scatter(thBestPerChan, r2)
            ax[1,2].set_title('pref angle vs R^2')
            ax[1,2].set_xlabel('theta')
            ax[1,2].set_ylabel('R^2')



        
        angs = np.linspace(np.min(uniqueTargAngle), np.min(uniqueTargAngle)+2*np.pi)
        cosTuningCurves = np.stack([chBs + chMod*np.cos(chTh-angs) for chBs, chMod, chTh in zip(bslnPerChan, modPerChan, thBestPerChan)])
        
        cosTuningCurveParams['tuningCurves'] = cosTuningCurves
        cosTuningCurveParams['tuningCurveAngs'] = angs
        
        return cosTuningCurveParams

    def pca(self, baselineSubtract = False, labels = None, n_components = None, plot = False):
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
            fig = plt.figure()
            supTitle = plot['supTitle'] if 'supTitle' in plot else ''
            fig.suptitle(supTitle)

            ax = np.empty((2,2), dtype='object')
            ax[0,0] = plt.subplot(221)
            ax[0,1] = plt.subplot(222)
            ax[1,0] = plt.subplot(223)
            ax[1,1] = plt.subplot(224,projection='3d')


            trialNumbersAll = np.arange(trialsPresented.shape[0])
            if self.ndim == 3:
                trialNumbersAllExp = np.repeat(trialNumbersAll, self.shape[2])
            else:
                trialNumbersAllExp = trialNumbersAll

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
                        ax[0,0].plot(xDimRed[trlsUse, 0],xDimRed[trlsUse, 1], '-', color = colorUse)
                        ax[0,0].set_xlabel('PC1')
                        ax[0,0].set_ylabel('PC2')

                        ax[0,1].plot(xDimRed[trlsUse, 0],xDimRed[trlsUse, 2], '-', color = colorUse)
                        ax[0,1].set_xlabel('PC1')
                        ax[0,1].set_ylabel('PC3')

                        ax[1,0].plot(xDimRed[trlsUse, 1],xDimRed[trlsUse, 2], '-', color = colorUse)
                        ax[1,0].set_xlabel('PC2')
                        ax[1,0].set_ylabel('PC3')

                        ax[1,1].plot(xDimRed[trlsUse, 0],xDimRed[trlsUse, 1],xDimRed[trlsUse, 2], '-', color = colorUse)
                        ax[1,1].set_xlabel('PC1')
                        ax[1,1].set_ylabel('PC2')
                        ax[1,1].set_zlabel('PC3')
                    else:
                        ax[0,0].plot(xDimRed[trlsUse, 0],xDimRed[trlsUse, 1], '.', color = colorUse)
                        ax[0,0].set_xlabel('PC1')
                        ax[0,0].set_ylabel('PC2')

                        ax[0,1].plot(xDimRed[trlsUse, 0],xDimRed[trlsUse, 2], '.', color = colorUse)
                        ax[0,1].set_xlabel('PC1')
                        ax[0,1].set_ylabel('PC3')

                        ax[1,0].plot(xDimRed[trlsUse, 1],xDimRed[trlsUse, 2], '.', color = colorUse)
                        ax[1,0].set_xlabel('PC2')
                        ax[1,0].set_ylabel('PC3')

                        ax[1,1].plot(xDimRed[trlsUse, 0],xDimRed[trlsUse, 1],xDimRed[trlsUse, 2], '.', color = colorUse)
                        ax[1,1].set_xlabel('PC1')
                        ax[1,1].set_ylabel('PC2')
                        ax[1,1].set_zlabel('PC3')
        
        plt.tight_layout()
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
    
    def decode(self, labels=None, decodeType = 'naiveBayes', trainFrac = 0.75, zScoreRespFirst = False, plot=True, numCVal = 5):
        # this is going to use a Gaussian naive Bayes classifier to try and
        # predict the labels of a held out set...
        from sklearn.naive_bayes import GaussianNB
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        from sklearn.svm import LinearSVC


        if labels is None:
            labels = self.labels['stimulusMainLabel']

        if zScoreRespFirst:
            if labels.ndim > 1:
                breakpoint() # this function should only take in a 1D list of labels...
            labelGroup = labels[:,None]
            grpSpksNpArr, lblPerGrp = self.groupByLabel(labelGroup) 
            # grpSpksNpArr = [(g - g.mean(axis=(0,2))[:,None])/g.std(axis=(0,2))[:,None] for g in grpSpksNpArr]
            # turns out that Bayes classification makes use of the mean...  so
            # subtracting it out is dumb.  Honestly should also be resistant to
            # scaled differences of variances, but who knows
            grpSpksNpArr = [g/g.std(axis=(0,2))[:,None] for g in grpSpksNpArr]
            spksUse = np.concatenate(grpSpksNpArr, axis=0) 
            labels = np.hstack([np.full(g.shape[0], lbl) for g, lbl in zip(grpSpksNpArr, lblPerGrp)])
            # here we get rid of any channels that didn't respond to any
            # particular condition basically (at that point its z-score ends up
            # as nan because its standard deviation for that condition (which
            # is divided above) equals 0
            # DOING BELOW INSTEAD
#            spksUse = spksUse[:, ~np.any(np.isnan(spksUse), axis=(0,2)), :]
        else:
            spksUse = self
#        for lbNum, lab in enumerate(labels):
#            lbMn.append(1)

        idxUse = np.arange(self.shape[0])
        tmAvgBins = spksUse.timeAverage()
        # here we get rid of any channels that didn't respond to any
        # particular condition basically (at that point its z-score ends up
        # as nan because its standard deviation for that condition (which
        # is divided above) equals 0, and if this happens for each time point
        # the time average is also nan (can consider at some point doing time
        # averages that ignore nans as well...
        tmAvgBins = tmAvgBins[:, ~np.any(np.isnan(tmAvgBins), axis=0)]
        trls, chans = tmAvgBins.shape # trials are samples, channels are features

        if decodeType == 'naiveBayes':
            cvalAccuracies = []
            unLab = np.unique(labels, axis=0)
            unLabInd = [np.all(labels == uL, axis=1).nonzero()[0] for uL in unLab]
            randIndPerLab = [indUL[np.random.permutation(np.arange(indUL.size))] for indUL in unLabInd]
            for cv in range(numCVal):
                # decided to implement stratified k-fold crossvalidation, to
                # ensure balance of conditions across folds
                testIndPerLab = [rI[int(rI.size*cv/numCVal):int(rI.size*(cv+1)/numCVal)] for rI in randIndPerLab]
                trainIndPerLab = [np.hstack((rI[:int(rI.size*cv/numCVal)], rI[int(rI.size*(cv+1)/numCVal):])) for rI in randIndPerLab]
                trainInds = np.hstack(trainIndPerLab)
                testInds = np.hstack(testIndPerLab)

                bayesClassifier = GaussianNB()
                bayesClassifier.fit(tmAvgBins[trainInds], labels[trainInds].squeeze())
                # breakpoint()

                testLabels = labels[testInds]

                accuracy = bayesClassifier.score(tmAvgBins[testInds], labels[testInds].squeeze())
                cvalAccuracies.append(accuracy)

            meanAccuracy = np.mean(cvalAccuracies)
            stdAccuracy = np.std(cvalAccuracies)

            out1 = meanAccuracy
            out2 = stdAccuracy
        elif decodeType == 'LDA':
            cvalAccuracies = []
            unLab = np.unique(labels, axis=0)
            unLabInd = [np.all(labels == uL, axis=1).nonzero()[0] for uL in unLab]
            randIndPerLab = [indUL[np.random.permutation(np.arange(indUL.size))] for indUL in unLabInd]
            for cv in range(numCVal):
                # decided to implement stratified k-fold crossvalidation, to
                # ensure balance of conditions across folds
                testIndPerLab = [rI[int(rI.size*cv/numCVal):int(rI.size*(cv+1)/numCVal)] for rI in randIndPerLab]
                trainIndPerLab = [np.hstack((rI[:int(rI.size*cv/numCVal)], rI[int(rI.size*(cv+1)/numCVal):])) for rI in randIndPerLab]
                trainInds = np.hstack(trainIndPerLab)
                testInds = np.hstack(testIndPerLab)

                ldaClassifier = LinearDiscriminantAnalysis()
                ldaClassifier.fit(tmAvgBins[trainInds], labels[trainInds].squeeze())
                # breakpoint()

                testLabels = labels[testInds]

                accuracy = ldaClassifier.score(tmAvgBins[testInds], labels[testInds].squeeze())
                cvalAccuracies.append(accuracy)

            meanAccuracy = np.mean(cvalAccuracies)
            stdAccuracy = np.std(cvalAccuracies)

            out1 = meanAccuracy
            out2 = stdAccuracy
        elif decodeType == 'linearSvm':
            clf = LinearSVC(max_iter=10000) # didn't work with default 1000
            clf.fit(tmAvgBins, labels)
            decFunc = clf.decision_function(tmAvgBins)

            unLbs = np.unique(labels)
            if unLbs.size>2:
                # haven't yet implemented/thought about SVM for >2 categories
                breakpoint()
            
            lb1 = decFunc[decFunc<0]
            lb2 = decFunc[decFunc>0]

            mnLb1 = np.abs(lb1).mean()
            mnLb2 = np.abs(lb2).mean()

            stdBoth = np.concatenate([lb1,lb2]).std()
            dprime = ((mnLb2-mnLb1)/stdBoth)**2

            out1 = dprime
            out2 = []
                


        return out1, out2

    def fisherInformation(self, labels=None):
        # this computation is based on
        # Kanitscheider et al. "Measuring Fisher Information Accurately in Correlated Neural Populations",
        # https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1004218#pcbi.1004218.e007
        # which allows me to use few trials to get a semi-valid information number

        if labels is None:
            labels = self.labels['stimulusMainLabel']

        unLbs, labsPres = np.unique(labels, return_inverse=True)
        if unLbs.size>2:
            raise(Exception("Fisher information is not really well defined for more than two categories I don't think..."))

        if self.shape[2]>1:
            breakpoint() # not sure what to do with multiple-timepoint data...
        else:
            bSpUse = self.timeAverage() # squashes time dimension...

        # get rid of nonresponsive channels...
        bSpUse = bSpUse[:, ~np.all(bSpUse==0, axis=0)]

        trls, chans = bSpUse.shape # trials are samples, channels are features

        condOne = bSpUse[labsPres == 0]
        condTwo = bSpUse[labsPres == 1]

        condOneMn = condOne.trialAverage()
        condTwoMn = condTwo.trialAverage()

        mnDiff = (condOneMn - condTwoMn)[:, None] # give it back a dimension to make it an n x 1 vector

        conds = self.labels['stimulusMainLabel']
        condOneLab = np.unique(conds[labsPres==0])
        condTwoLab = np.unique(conds[labsPres==1])
        diffCond = condOneLab - condTwoLab
        
        condOneCov = np.cov(condOne.T, ddof=1)
        condTwoCov = np.cov(condTwo.T, ddof=1)
        mnCov = (condOneCov + condTwoCov)/2

        try:
            invCov = np.linalg.inv(mnCov)
        except np.linalg.LinAlgError:
            # NOTE: this feels... mathematically... inappropriate...
            condOneMnSub = condOne - condOneMn
            condTwoMnSub = condTwo - condTwoMn
            condConcat = np.vstack([condOneMnSub, condTwoMnSub])
            mnCov = np.cov(condConcat.T)
            invCov = np.linalg.inv(mnCov)




        invCov = np.array(invCov)
        mnDiff = np.array(mnDiff)
        infoRaw = mnDiff.T @ invCov @ mnDiff / diffCond**2

        biasCorrScaling = (2*trls - chans - 3) / (2*trls - 2)
        biasCorrShift = 2*chans/(trls*diffCond**2)

        biasCorrInfo = infoRaw*biasCorrScaling - biasCorrShift

        return biasCorrInfo.squeeze()[None]

        


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
        
        if residualSpikes.dtype == 'object':
            residualSpikesFlat = np.concatenate(residualSpikes, axis=2) 
            chanFlat = residualSpikesFlat.squeeze()
            # make an array of number of time points for each trial; will be
            # use for appropriately repeating the labels
            numTp = [trl[0].shape[0] for trl in residualSpikes]

            # do the same as above for the baseline subtracted values, without baseline
            # subtracting--now we can find the true geometric mean firing rate
            flatCnt = np.concatenate(self[:,chansGood], axis=2).squeeze()
            flatCntMn = flatCnt.mean(axis=1)
            flatCntMn = np.expand_dims(flatCntMn, axis=1) # need to add back a lost dimension
        else:
            chanFirst = residualSpikesGoodChans.swapaxes(0, 1) # channels are now main axis, for sampling in reshape below
            chanFlat = chanFirst.reshape((residualSpikesGoodChans.shape[1], -1)) # chan are now rows, time is cols
            numTp = self.shape[2]
            # do the same as above for the baseline subtracted values, without baseline
            # subtracting--now we can find the true geometric mean firing rate
            flatCnt = np.array(self[:,chansGood,:].swapaxes(0,1).reshape((chansGood.size,-1)))
            flatCntMn = flatCnt.mean(axis=1)
            flatCntMn = np.expand_dims(flatCntMn, axis=1) # need to add back a lost dimension
        
        # let's remove trials that are larger than 3*std
        stdChanResp = np.std(chanFlat, axis=1)
#        chanMask = np.abs(chanFlat) > (3*stdChanResp[:,None]) # < 0
        chanMask = np.abs(chanFlat) < 0 # mask nothing... for now
        maskChanFlat = np.ma.array(np.array(chanFlat), mask = np.array(chanMask))
        
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
            labelPresented = np.repeat(labelPresented, numTp)
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
            labelPresented = np.repeat(labelPresented, numTp)
            # when combining the labels, we z-score the spike counts first
            for lblNum, _ in enumerate(uniqueLabel):
                lblSpks = maskChanFlat[:, labelPresented==lblNum]
                maskChanFlat[:, labelPresented==lblNum] = (lblSpks - lblSpks.mean(axis=1)[:,None])/lblSpks.std(axis=1)[:,None]
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
             labelUse = 'stimulusMainLabel', condNums=[0], combineConds = False, overallFiringRateThresh = 1, perConditionGroupFiringRateThresh = 1, balanceConds = True, 
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
        
        # note that overallFiringRateThresh should be a negative number if we're inputting residuals...
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
        
        infoTrlChKeep = dict(trialsKeep = trlIndsUseLabel, channelsKeep = chIndsUseFR)
        uniqueTargAngle, trialsPresented = np.unique(newLabels, return_inverse=True, axis=0)
        
        if condNums is None:
            condsUse = np.arange(uniqueTargAngle.shape[0])
        else:
            # we allow the top level to choose random condNums if it wants them
            # but we sort them in case there's a combo--this ensures the names are the same for combos
            condsUse = np.sort(np.stack(condNums))

        
        uniqueTargAngleDeg = uniqueTargAngle*180/np.pi
        
        
        if perConditionGroupFiringRateThresh > 0 and combineConds:
            breakpoint()
            # I keep the below comment for posterity, but the solution is that
            # it *doesn't make sense* to have a per condition firing rate if
            # we're combining conditions, because GPFA is being computed for
            # *everything*. To make things explicit, then, tell the user to
            # choose one or the other
            #
            # so there's a confusion if both of these are true... especially
            # when computeResiduals is in the mix--if computeResiduals is True,
            # we want to compute the residuals on a per-condition basis
            # (right?) and then combine the conditions... but in order to
            # compute the residuals, we first want to remove the low firing
            # rate channels... but if we do that the conditions might not be
            # combined correctly because some channels for some conditions
            # might be kicked off. In any case, I'm not sure when we'll run
            # into this, but when we do I will revisit.
            raise Exception("If you are combining conditions, it doesn't make sense to have a per-condition firing rate threshold--either set perConditionGroupFiringRateThresh to 0, or combineConds to False")

            grpSpksNpArr, _ = binnedSpikesUse.channelsAboveThresholdFiringRate(firingRateThresh=perConditionGroupFiringRateThresh)
        
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
            # If we combine conditions we are going to z-score each channel's
            # response *within* each condition first. This reduces the fact
            # that PSTH is not the only thing that changes between conditions
            # (which looking at residuals would take care of since the PSTH is
            # removed), but that variance *also* changes, because these are
            # assumed Poisson. (Also note that when looking at residuals,
            # you've already mean subtracted, since they're residuals, but this
            # function allows for that not to be the case, so appropriate
            # z-scoring must include mean subtraction!)
            #
            # NOTE: I'm computing the mean and standard deviation over all time
            # bins (so instead of subtracting a mean/std per time point if I
            # were to compute them over just axis=0, or per trial if I were to
            # compute with axis=2, I combine both and subtract over all of them
            grpSpksNpArr = [(g - g.mean(axis=(0,2))[:,None])/g.std(axis=(0,2))[:,None] for g in grpSpksNpArr]
            grpSpksNpArr = [np.concatenate(grpSpksNpArr, axis=0)] # grpSpksNpArr
            # I'm not sure that this is the best way to approach this, but if
            # the standard deviation is 0 for a channel in a condition, then
            # that channel/condition's entry is np.nan, and I'm going to change
            # that to 0. I do this since I've only witnessed it when the
            # channel has zero response. It could conceivably occur elsewhere,
            # but I'll cross that bridge when I get to it?
            if np.any(np.isnan(grpSpksNpArr)[0]):
                breakpoint() # NOTE I added this here because I want to know/remember what happens if the perCondFR up above removes a channel from one condition but not the other
            grpSpksNpArr[0][np.isnan(grpSpksNpArr[0])] = 0
            condDescriptors = ['s' + '-'.join(['%d' % stN for stN in condsUse]) + 'Grpd']
            # grouping all of them together
            condsUse = [condsUse] # list turns out to be important keeping things together...
        else:
            grpSpksNpArr = grpSpksNpArr
            condDescriptors = ['%d' % stN for stN in condsUse]
            # grouping into a list of individual lists
            condsUse = [[cond] for cond in condsUse]




        # simple change of variable name
        groupedBalancedSpikes = grpSpksNpArr
            
        return groupedBalancedSpikes, condDescriptors, condsUse, infoTrlChKeep

    def fa(self, groupedBalancedSpikes, outputPathToConditions, condDescriptors, xDim, labelUse, crossvalidateNum = 4, expMaxIterationMaxNum = 500, tolerance = 1e-8, tolType = 'ratio', forceNewFaRun = False):
        assert isinstance(xDim, (int,np.integer)), "Must only provide one integer xDim at a time"
        from classes.FA import FA
        from time import time
        tSt = time()

        condSavePaths = []
        xDimBestAll = []
        xDimScoreBestAll = []
        gpfaPrepAll = []
        loadedSaved = []
        faRunFinalDetails = []
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
                preSavedDataPath = fullOutputPath / ("faResultsDim%d.npz" % xDim)
                condSavePaths.append(preSavedDataPath)

                if preSavedDataPath.exists() and not forceNewFaRun:
                    print("Loading FA for condition %d/%d" % (idx+1, len(groupedBalancedSpikes)))
                    with np.load(preSavedDataPath, allow_pickle=True) as gpfaRes:
                        faResLoaded = dict(
                            zip((k for k in gpfaRes), (gpfaRes[k] for k in gpfaRes))
                        )
                        faScoreCond[0,:] = faResLoaded['score' if 'score' in faResLoaded else 'normalGpfaScore']
                        faPrep.dimOutput[xDim] = faResLoaded['dimOutput'][()]
                        
                        # These are getting replaced every time. I think... that's... fine...
                        faPrep.testInds = faResLoaded['testInds']
                        faPrep.trainInds = faResLoaded['trainInds']
                else:
                    preSavedDataPath.parent.mkdir(parents=True, exist_ok = True)
                    faScoreCond[0, :] = faPrep.runFa( numDim=xDim, gpfaResultsPath = fullOutputPath )[0]



                    print("FA training for condition %d/%d done" % (idx+1, len(groupedBalancedSpikes)))
                            
                    np.savez(preSavedDataPath, dimOutput=faPrep.dimOutput[xDim], testInds = faPrep.testInds, trainInds=faPrep.trainInds, score=faScoreCond[0,:], alignmentBins = grpSpks.alignmentBins, condLabel = grpSpks.labels[labelUse], binSize = grpSpks.binSize  )

                converged = [estParam['converge'] for estParam in faPrep.dimOutput[xDim]['allEstParams']]
                finalRatioChange = [estParam['finalRatioChange'] for estParam in faPrep.dimOutput[xDim]['allEstParams']]
                finalDiffChange = [estParam['finalDiffChange'] for estParam in faPrep.dimOutput[xDim]['allEstParams']]
                trainIsFullRank = faPrep.dimOutput[xDim]['fullRank']

                faRunFinalDetails.append([trainIsFullRank,converged,finalRatioChange,finalDiffChange])
                faScoreAll.append(faScoreCond)

        print("All FA training done in %d seconds" % (time()-tSt))
        # using getattr here allows me to preset gpfaPrepAll to an empty array
        # and have the outputs be empty arrays if nothing happens (i.e. an
        # error with gpfa--or a handled mistake where there are too few neurons
        # for the desired dimension, for example
        faPrepDimOutputAll = [getattr(fa, 'dimOutput', None) for fa in faPrepAll]
        faTestIndsAll = [getattr(fa, 'testInds', None) for fa in faPrepAll]
        faTrainIndsAll = [getattr(fa, 'trainInds', None) for fa in faPrepAll]

        return faPrepDimOutputAll, faTestIndsAll, faTrainIndsAll, faScoreAll, faRunFinalDetails, condSavePaths


    def gpfa(self, groupedBalancedSpikes, outputPathToConditions, condDescriptors, xDim, labelUse, crossvalidateNum = 4, expMaxIterationMaxNum = 500, tolerance = 1e-8, tolType = 'ratio', forceNewGpfaRun = False):
        from classes.GPFA import GPFA
        from time import time

        condSavePaths = []
        gpfaRunFinalDetails = []
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

                # loading for each dimension and concatenating
                gpfaScoreCond = np.empty((1,crossvalidateNum))

                preSavedDataPath = fullOutputPath / ("gpfaResultsDim%d.npz" % xDim)
                condSavePaths.append(preSavedDataPath)
                if preSavedDataPath.exists() and not forceNewGpfaRun:
                    try:
                        gpfaDimSaved = np.load(preSavedDataPath, allow_pickle=True)

                        gpfaScoreCond[0,:] = gpfaDimSaved['score' if 'score' in gpfaDimSaved else 'normalGpfaScore']
                        gpfaPrep.dimOutput[xDim] = gpfaDimSaved['dimOutput'][()]
                        
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
                        gpfaPrep.runGpfaInMatlab(fname=fullOutputPath,  crossvalidateNum=crossvalidateNum, xDim=xDim, expMaxIterationMaxNum = expMaxIterationMaxNum, tolerance = tolerance, tolType = tolType, forceNewGpfaRun = forceNewGpfaRun);
                    except Exception as e:
                        from matlab import engine
                        if type(e) is engine.MatlabExecutionError:
                            print(e)
#                                breakpoint()
                            continue
                        else:
                            raise(e)

                loadedSaved = loadedDimCond
                converged = [estParam['converge'] for estParam in gpfaPrep.dimOutput[xDim]['allEstParams']]
                finalRatioChange = [estParam['finalRatioChange'] for estParam in gpfaPrep.dimOutput[xDim]['allEstParams']]
                finalDiffChange = [estParam['finalDiffChange'] for estParam in gpfaPrep.dimOutput[xDim]['allEstParams']]
                trainIsFullRank = gpfaPrep.dimOutput[xDim]['fullRank']
                gpfaRunFinalDetails.append([trainIsFullRank,converged,finalRatioChange,finalDiffChange])

            print("GPFA training for condition %d/%d done" % (idx+1, len(groupedBalancedSpikes)))
                        
            print("** Crossvalidating GPFA for condition %d/%d **" % (idx+1, len(groupedBalancedSpikes)))
            # find any dimensions that still need crossvalidating
            dimsCVal = []
            if not loadedSaved:
                dimsCVal.append(xDim)

                gpfaScore, gpfaScoreErr, reducedGpfaScore = gpfaPrep.crossvalidatedGpfaError(approach = cvApproach, dimsCrossvalidate = dimsCVal)
                gpfaScoreCurr = gpfaScoreCond
                gpfaScoreCurr[0, :] = gpfaScore
#                gpfaScoreCurr[indsCVal, :] = gpfaScore
                gpfaScoreCond = gpfaScoreCurr

            # find the best xdim based on the crossvalidation approach
            gpfaScore = gpfaScoreCond

            # save the computation!
            # But save it per dimension! This allows later loading for specific
            # dimension tests, and avoids rewriting previous loads of many
            # dimensions
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

        return gpfaPrepDimOutputAll, gpfaTestIndsAll, gpfaTrainIndsAll, gpfaRunFinalDetails, condSavePaths

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
        
        newLabels = {}
        unLabelKeys = np.unique([list(arr.labels.keys()) for arr in arrays])
        for key in unLabelKeys:
            # might need to return to the below if keys start having to be lists...
            keyVals = np.concatenate([arr.labels[key] for arr in arrays])
            newLabels[key] = keyVals # list(chain(*keyVals))
            
        concatAlBinsInit = np.stack([arr.alignmentBins for arr in arrays])
        concatAlBins = list(chain(*concatAlBinsInit))
        units = arrays[0].units
        return BinnedSpikeSet(concatTrad, start=concatStart, end=concatEnd, binSize=binSizes[0], labels=newLabels, alignmentBins=concatAlBins, units = units)
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
        
        # unLabelKeys = np.unique([arr.labels.keys() for arr in arrays])
        # newLabels = {}
        # # breakpoint() # ermm... is this iterating correctly? Check how I did it in vstack?
        # # NOTE ... replaced with vstack label coe for now...
        # for key in unLabelKeys:
        #     # might need to return to the below if keys start having to be lists...
        #     keyVals = np.stack([arr.labels[list(key)[0]] for arr in arrays])
        #     newLabels[list(key)[0]] = keyVals # list(chain(*keyVals))
        unLabelKeys = np.unique([arr.labels.keys() for arr in arrays])
        newLabels = {}
        for key in unLabelKeys[0]:
            # might need to return to the below if keys start having to be lists...
            # keyVals = np.concatenate([arr.labels[list(key)[0]] for arr in arrays])
            keyVals = np.vstack([arr.labels[key] for arr in arrays])
            newLabels[key] = keyVals # list(chain(*keyVals))
            
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

# implementation of vstack for BinnedSpikeSetObjects
@implements(np.vstack)
def vstack(arrays):
    binSizes = np.stack([arr.binSize for arr in arrays])
    units = np.stack([arr.units for arr in arrays])
    if not np.all(binSizes == binSizes[0]):
        raise(Exception("Can't (shouldn't?) concatenate BinnedSpikeSets with different bin sizes!"))
    if not np.all(units == units[0]):
        raise(Exception("Can't (shouldn't?) concatenate BinnedSpikeSets with different units!"))

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
        
    vstackTrad = np.vstack(arrayNd)

    # not sure why, but a concatenate works here where a stack doesn't...
    try:
        stackStart = np.concatenate([arr.start for arr in arrays]) 
    except ValueError:
        stackStart = np.vstack([arr.start for arr in arrays])
        if np.all(stackStart==None):
            stackStart = None

    # not sure why, but a concatenate works here where a stack doesn't...
    try:
        stackEnd = np.concatenate([arr.end for arr in arrays])
    except ValueError:
        stackEnd = np.vstack([arr.end for arr in arrays])
        if np.all(stackEnd==None):
            stackEnd = None
    
    unLabelKeys = np.unique([arr.labels.keys() for arr in arrays])
    newLabels = {}
    for key in unLabelKeys[0]:
        # might need to return to the below if keys start having to be lists...
        # keyVals = np.concatenate([arr.labels[list(key)[0]] for arr in arrays])
        keyVals = np.vstack([arr.labels[key] for arr in arrays])
        newLabels[key] = keyVals # list(chain(*keyVals))
        
    stackAlBins = np.vstack([arr.alignmentBins for arr in arrays])
    if np.all(stackAlBins == None):
        stackAlBins = None

    return BinnedSpikeSet(vstackTrad, start=stackStart, end=stackEnd, binSize=binSizes[0], labels=newLabels, alignmentBins=stackAlBins, units = units[0])

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
    if type(array) is BinnedSpikeSet and array.dtype=='object':
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

