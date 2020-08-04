#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 15:26:50 2020

@author: emilio
"""
from classes.MatlabConversion import MatlabConversion as mc
# there are weird and common chance when multiprocessing with numpy fails
# because of OpenBLAS's automatic underlying parallelization, but these
# environment variables (set BEFORE numpy is imported), take care of the
# problem
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['GOTO_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np
import scipy as sp
from multiprocessing import Pool, TimeoutError
# I don't want *all* future imports of numpy to be affected like this... so
# clearing the environment variables here...
#del os.environ['OPENBLAS_NUM_THREADS']
#del os.environ['GOTO_NUM_THREADS']
#del os.environ['OMP_NUM_THREADS']
from classes.BinnedSpikeSet import BinnedSpikeSet
from methods.GeneralMethods import prepareMatlab
from matlab.engine import MatlabExecutionError
from decorators.ParallelProcessingDecorators import multiprocessNumpy

class GPFA:
    def __init__(self, binnedSpikes):
#        if type(binnedSpikes) is list or binnedSpikes.dtype=='object':
#            # I'm honestly not really sure why this works, given that with the
#            # axis=1 in the else statement, axis 1 stays the same size, but in
#            # this statement with axis=2, axis 2 *changes size*...
#            allTrlTogether = np.concatenate(binnedSpikes, axis=2)
#        else:
#            allTrlTogether = np.concatenate(binnedSpikes, axis=1)[None,:,:]
#        
#        allTrlTogether.timeAverage().trialAverage()
#        allTrlTogether.units = binnedSpikes.units
#        _, chansKeep = allTrlTogether.channelsAboveThresholdFiringRate(firingRateThresh=firingRateThresh)
        # chansKeepCoinc = allTrlTogether.removeCoincidentSpikes()
        # chansKeep = np.logical_and(chansKeepThresh, chansKeepCoinc)
        if type(binnedSpikes) is list:
            # the 0 index squeezes out that dimension--and it should be what happens,
            # as in this structuring you should have only one trial in here...
            self.binnedSpikes = [bnSp[0] for bnSp in binnedSpikes]
        elif binnedSpikes.dtype=='object':
            self.binnedSpikes = [np.stack(bnSp) for bnSp in binnedSpikes]
        else:
            self.binnedSpikes = [bnSp for bnSp in binnedSpikes]
#        breakpoint()
        self.gpfaSeqDict = self.binSpikesToGpfaInputDict()
        self.trainInds = None
        self.testInds = None
        self.dimOutput = {}
        
        if type(binnedSpikes) is list:
            self.binSize = binnedSpikes[0].binSize
        else:
            self.binSize = binnedSpikes.binSize
    
    def binSpikesToGpfaInputDict(self, binnedSpikes = None):
        if binnedSpikes is None:
            binnedSpikes = self.binnedSpikes
        
        gpfaSeqDict = []
        for trlNum, trl in enumerate(binnedSpikes):
#            if sqrtSpks:
#                trl = np.sqrt(trl)
            gpfaSeqDict.append({'trialId':trlNum,
                            'T': (trl.shape[1])*1.0,
                            'y': mc.convertNumpyArrayToMatlab(trl)})
        
        return gpfaSeqDict

    def gpfaInputDictToBinSpikeList(self, seq=None):
        if seq is None:
            seq = self.gpfaSeqDict
        return [mc.convertMatlabArrayToNumpy(sq['y']) for sq in seq]

    
    # train and test inds are computed as randomly dispersed to avoid
    # session-long signals from messing with the overall training (i.e. a
    # session-length increase in baseline would show up if the trials used to
    # train weren't randomized across teh session)
    #
    # the initSeed input is there in case one of the cross validations fails
    # and a new set of inds needs to be found--this has only happened once in
    # all my experience, so I expect it to be rarely used
    def computeTrainTestIndRandom(self, numFolds = 4, initSeed = 0):
        seqAll = self.gpfaSeqDict
        numTrls = len(seqAll)
        
        initSt = np.random.get_state()
        np.random.seed(seed=initSeed)
        testInds = []
        trainInds = []
        for fld in range(numFolds):
            randomizedIndOrder = np.random.permutation(np.arange(numTrls))
            testInds.append(randomizedIndOrder[:round((1/numFolds)*numTrls)])
            trainInds.append(randomizedIndOrder[round((1/numFolds)*numTrls):])
        
       
        
        np.random.set_state(initSt)
        
        return trainInds, testInds  # for splitting binned spikes into train and test sets
        
    # Sequential sets of test inds, non-overlapping between each fold, a-la original
    # Yu 2009 code
    def computeTrainTestIndOrdered(self, numFolds=4):
        seqAll = self.gpfaSeqDict
        numTrls = len(seqAll)
        
        testBinEdges = np.floor(np.linspace(0,numTrls,numFolds+1))
        allInds = np.arange(numTrls)
        trainInds = [np.concatenate((allInds[:int(tBS)], allInds[int(tBE):])) for tBS, tBE in zip(testBinEdges[:-1], testBinEdges[1:])]
        testInds = [allInds[int(tBS):int(tBE)] for tBS, tBE in zip(testBinEdges[:-1], testBinEdges[1:])]
        
       
         
        return trainInds, testInds
    
    def trainAndTestSeq(self):
        seqAll = self.gpfaSeqDict
        seqsTest = []
        seqsTrain = []
        for trnInd, tstInd in zip(self.trainInds, self.testInds):
            seqsTrain.append([seqAll[idx] for idx in trnInd])
            seqsTest.append([seqAll[idx] for idx in tstInd])
         
        return seqsTrain, seqsTest
            
        
        
    
    @multiprocessNumpy
    def runGpfaInMatlab(self, fname,   crossvalidateNum = 0, xDim = 8, eng=None, segLength = None, forceNewGpfaRun = False):
        
        # if eng is None:
        #     from matlab import engine 
        #     eng = engine.start_matlab()
        
        if crossvalidateNum == 1: # effectively not crossvalidating...
            self.trainInds, self.testInds =  self.computeTrainTestIndRandom(numFolds = crossvalidateNum)
            # with no crossvalidations, the train inds *are* the test inds...
            self.trainInds = self.testInds
            seqsTrain, seqsTest = self.trainAndTestSeq()
        else:
            self.trainInds, self.testInds =  self.computeTrainTestIndRandom(numFolds = crossvalidateNum)
            # self.trainInds, self.testInds = self.computeTrainTestIndOrdered(numFolds = crossvalidateNum)
            seqsTrain, seqsTest = self.trainAndTestSeq()

        seqTrainStr = "[seqTrain{:}]"
        seqTestStr = "[seqTest{:}]"
            
        if segLength is None:
            segLength = min([sq['T'] for sq in self.gpfaSeqDict])
            
        binWidth = self.binSize
        
        seqsTrainNew = []
        seqsTestNew = []
        allEstParams = []
        
        
        
        # In order to ensure numpy processes work well in a multiprocess
        # environment, we set these environment variables (they need to exist
        # both when numpy gets loaded for the first time *AND* whenever the
        # processes are running
#        os.environ['OPENBLAS_NUM_THREADS'] = '1'
#        os.environ['GOTO_NUM_THREADS'] = '1'
#        os.environ['OMP_NUM_THREADS'] = '1'
        with Pool() as poolHere:
            res = []
            fullRank = []
            for cvNum, (sqTrn, sqTst) in enumerate(zip(seqsTrain, seqsTest)):
                datTest = self.gpfaInputDictToBinSpikeList(seq = sqTrn)
                datTest = np.concatenate(datTest, axis=1)
                fullRank.append(np.linalg.matrix_rank(datTest) >= datTest.shape[0])

                res.append(poolHere.apply_async(parallelGpfa, (fname, cvNum, xDim, sqTrn, sqTst, forceNewGpfaRun, binWidth, segLength, seqTrainStr, seqTestStr)))
            
            # from time import sleep
            # print('blah')
            # sleep(20)
            # print('bleh')
            resultsByCrossVal = [[]] * len(res) # empty set of lists
            badCrossVals = []
            for cvNum, (rs, fR) in enumerate(zip(res, fullRank)):
                try:
                    resultsByCrossVal[cvNum] = rs.get() + (fR,)
                except MatlabExecutionError as e:
                    # something failed with one of the crossvalidations... so act as if it didn't happen!
                    badCrossVals.append(cvNum)
                    continue

            if len(badCrossVals) != 0:
                # set initSeed to 1 just so it's different from 0...
                trainInds, testInds = self.computeTrainTestIndRandom(numFolds = crossvalidateNum, initSeed = 1)
                for bCVNum in badCrossVals:
                    success = False
                    while trainInds:
                        self.trainInds[bCVNum] = trainInds.pop()
                        self.testInds[bCVNum] = testInds.pop()
                        seqsTrain, seqsTest = self.trainAndTestSeq()

                        datTest = self.gpfaInputDictToBinSpikeList(seq = seqsTrain)
                        datTest = np.stack(datTest, axis=1)
                        fRNew = np.linalg.matrix_rank(datTests) >= datTest.shape[0]
                        breakpoint()

                        # now we're doing it as a pool because if we do it on
                        # the main process Matlab won't run again until we
                        # restart Python >.>
                        resNew = poolHere.apply_async(parallelGpfa, (fname, bCVNum, xDim, seqsTrain[bCVNum], seqsTest[bCVNum], forceNewGpfaRun, binWidth, segLength, seqTrainStr, seqTestStr))
                        try:
                            resultsByCrossVal[bCVNum] = resNew.get() + fRNew
                        except MatlabExecutionError as e:
                            pass
                        else:
                            success = True
                            break
                    if not success:
                        print("Something went wrong with crossval #%d... this GPFA will now have one less crossvalidation!" % bCVNum)


            
#            poolHere.close()
#            poolHere.join()
#            breakpoint()
            
            resultsByVar = list(zip(*resultsByCrossVal))
            allEstParams = resultsByVar[0]
            seqsTrainNew = resultsByVar[1]
            seqsTestNew = resultsByVar[2]
            fullRank = resultsByVar[3]
            
        # To prevent these variables from causing other methods/classes to load
        # numpy in a non-'optimized' (parallel) state, we delete (effectively
        # unset) the variables after our multiprocessing has finished
#        del os.environ['OPENBLAS_NUM_THREADS']
#        del os.environ['GOTO_NUM_THREADS']
#        del os.environ['OMP_NUM_THREADS']
        
        # print('whowhat?')
        self.dimOutput[xDim] = {}
        self.dimOutput[xDim]['crossvalidateNum'] = crossvalidateNum
        self.dimOutput[xDim]['allEstParams'] = allEstParams
        self.dimOutput[xDim]['seqsTrainNew'] = seqsTrainNew
        self.dimOutput[xDim]['seqsTestNew'] = seqsTestNew
        self.dimOutput[xDim]['fullRank'] = fullRank
            
        return allEstParams, seqsTrainNew, seqsTestNew, fullRank
    
    @multiprocessNumpy
    def crossvalidatedGpfaError(self, approach = "logLikelihood", eng=None, dimsCrossvalidate = None):
        
        # if eng is None:
        #     from methods.GeneralMethods import prepareMatlab
        #     eng = prepareMatlab()
        eng = None
        
        if approach is "logLikelihood":
            _, allSeqsTest = self.trainAndTestSeq()
            
                
            ll = {}
            llErr = {}
            numFolds = len(allSeqsTest)
            
            
            
            # In order to ensure numpy processes work well in a multiprocess
            # environment, we set these environment variables (they need to exist
            # both when numpy gets loaded for the first time *AND* whenever the
            # processes are running
#            os.environ['OPENBLAS_NUM_THREADS'] = '1'
#            os.environ['GOTO_NUM_THREADS'] = '1'
#            os.environ['OMP_NUM_THREADS'] = '1'
            with Pool() as plWrap:
                res = []
                dimUsed = []
                if dimsCrossvalidate is None: # actually kinda the opposite... this means it's all..
                    for dimMax, paramsGpfa in self.dimOutput.items():
                        res.append(plWrap.apply_async(cvalSuperWrap, (paramsGpfa, allSeqsTest)))
                        dimUsed.append(dimMax)
                else:
                    for dim in dimsCrossvalidate:
                        paramsGpfa = self.dimOutput[dim]
                        res.append(plWrap.apply_async(cvalSuperWrap, (paramsGpfa, allSeqsTest)))
                        dimUsed.append(dim)
                    
                resultsByDim = []
                timeoutSecs = 60 # this time just keeps getting longer...
                for ind, rs in enumerate(res):
                    try:
                        #NOTE CHANGE
                        thisResult = rs.get()#timeout=timeoutSecs)
                    except TimeoutError as e:
                        print("Careful we're establishing a Matlab at the top level...")
                        dimRedo = dimUsed[ind]
                        paramsGpfa = self.dimOutput[dimRedo]
                        thisResult = cvalSuperWrap(paramsGpfa, allSeqsTest)
                        # lower the timeout if it already failed once...
                        timeoutSecs = 3

                    resultsByDim.append(thisResult)

                        

#                resultsByDim = [rs.get() for rs in res]
                # resultsByVar = list(zip(*resultsByDim))
                
                if dimsCrossvalidate is None:
                    ll = {dim : np.stack(llDim) for dim, llDim in zip(self.dimOutput.keys(), resultsByDim)}        
                else:
                    ll = {dim : np.stack(llDim) for dim, llDim in zip(dimsCrossvalidate, resultsByDim)}        
                    # llTemp = np.stack(llTemp)
                    # llTemp = llTemp - np.min(llTemp)
                    # llTemp = llTemp/np.max(llTemp)
                    # ll[dimMax] = np.stack(llTemp)#np.mean(np.stack(llTemp))
                # llErr[dimMax] = np.std(np.stack(llTemp))
                    
                
            # To prevent these variables from causing other methods/classes to load
            # numpy in a non-'optimized' (parallel) state, we delete (effectively
            # unset) the variables after our multiprocessing has finished
#            del os.environ['OPENBLAS_NUM_THREADS']
#            del os.environ['GOTO_NUM_THREADS']
#            del os.environ['OMP_NUM_THREADS']

            reducedGpfaScore = np.stack([np.nan])
            gpfaScore = np.stack([llH for _, llH in ll.items()])
            gpfaScoreErr = np.stack([np.nan])
#            normalizedGpfaScore = (normalGpfaScore - np.min(normalGpfaScore, axis=0))/(np.max(normalGpfaScore,axis=0)-np.min(normalGpfaScore,axis=0))
#            normalGpfaScore = np.mean(normalizedGpfaScore, axis=1)
#            normalGpfaScoreErr = np.std(normalizedGpfaScore,axis=1)
            
            
            
                    
        elif approach is "squaredError":
            errs = {}
            for dimMax, paramsGpfa in self.dimOutput.items():
                errs[dimMax] = {}
                for testSeqs in paramsGpfa['seqsTestNew']:
                    squashTrueOut = [tstSq['y'] for tstSq in testSeqs]
                    squashTrueOut = np.concatenate(squashTrueOut, axis=1)
                    for dim in range(dimMax):
                        try:
                            errs[dimMax][dim]
                        except KeyError as e:
                            errs[dimMax][dim] = []
                        squashDimTestOut = [tstSq['ycsOrth%02d' % (dim+1)] for tstSq in testSeqs]
                        squashDimTestOut = np.concatenate(squashDimTestOut, axis=1)
                        errs[dimMax][dim] = np.append(errs[dimMax][dim], np.sum(np.power(squashDimTestOut - squashTrueOut, 2).flatten()))
            
            dimsTested = [dim for dim in errs.keys()]
            maxDimGpfa = max(dimsTested)
            
            # Looking at code from Yu et al 2009, reducedGpfa loss is computed by looking
            # at the error of dimensions up to the max dimension when the maximum amount of
            # dimensions tested is used to extract the GPFA traces
            reducedGpfaScore = np.stack([np.mean(dimErr) for _, dimErr in errs[maxDimGpfa].items()])
            
            # Normal GPFA error is computed by looking at the error of all of the dimensions
            # for GPFAs run using different numbers of dimensions
            gpfaScore = np.stack([np.mean(gpfaDim[maxDim-1]) for maxDim, gpfaDim in errs.items()])
            gpfaScore = np.stack([np.std(gpfaDim[maxDim-1]) for maxDim, gpfaDim in errs.items()])
            
        return gpfaScore, gpfaScoreErr, reducedGpfaScore
    
    def shuffleGpfaControl(self, estParams, cvalTest=0, numShuffle = 500,  eng=None): 

                    


        from multiprocessing import Pool
        with Pool() as poolHere:
            res = []
            eng = None
                
            res.append(poolHere.apply_async(shuffParallel, (self.binnedSpikes, self.testInds, estParams, cvalTest, numShuffle, eng)))

            resultsShuff = [rs.get() for rs in res]
            # resultsByVar = list(zip(*resultsByDim))

        return resultsShuff[0]


        

    
# NOTE: these bits of code are grabbed from the Elephant team, which I hadn't
# found originally when trying to see if GPFA had been implemented in Python yet >.>
def invPersymm(M, blk_size):
    """
    Inverts a matrix that is block persymmetric.  This function is
    faster than calling inv(M) directly because it only computes the
    top half of inv(M).  The bottom half of inv(M) is made up of
    elements from the top half of inv(M).
    WARNING: If the input matrix M is not block persymmetric, no
    error message will be produced and the output of this function will
    not be meaningful.
    Parameters
    ----------
    M : (blkSize*T, blkSize*T) np.ndarray
        The block persymmetric matrix to be inverted.
        Each block is blkSize x blkSize, arranged in a T x T grid.
    blk_size : int
        Edge length of one block
    Returns
    -------
    invM : (blkSize*T, blkSize*T) np.ndarray
        Inverse of M
    logdet_M : float
        Log determinant of M
    """
    T = int(M.shape[0] / blk_size)
    Thalf = np.int(np.ceil(T / 2.0))
    mkr = blk_size * Thalf

    invA11 = np.linalg.inv(M[:mkr, :mkr])
    invA11 = (invA11 + invA11.T) / 2

    # Multiplication of a sparse matrix by a dense matrix is not supported by
    # SciPy. Making A12 a sparse matrix here  an error later.
    off_diag_sparse = False
    if off_diag_sparse:
        A12 = sp.sparse.csr_matrix(M[:mkr, mkr:])
    else:
        A12 = M[:mkr, mkr:]

    term = invA11.dot(A12)
    F22 = M[mkr:, mkr:] - A12.T.dot(term)

    res12 = rdiv(-term, F22)
    res11 = invA11 - res12.dot(term.T)
    res11 = (res11 + res11.T) / 2

    # Fill in bottom half of invM by picking elements from res11 and res12
    invM = fillPerSymm(np.hstack([res11, res12]), blk_size, T)

    logdet_M = -logdet(invA11) + logdet(F22)

    return invM, logdet_M


def fillPerSymm(p_in, blk_size, n_blocks, blk_size_vert=None):
    """
     Fills in the bottom half of a block persymmetric matrix, given the
     top half.
     Parameters
     ----------
     p_in :  (xDim*Thalf, xDim*T) np.ndarray
        Top half of block persymmetric matrix, where Thalf = ceil(T/2)
     blk_size : int
        Edge length of one block
     n_blocks : int
        Number of blocks making up a row of Pin
     blk_size_vert : int, optional
        Vertical block edge length if blocks are not square.
        `blk_size` is assumed to be the horizontal block edge length.
     Returns
     -------
     Pout : (xDim*T, xDim*T) np.ndarray
        Full block persymmetric matrix
    """
    if blk_size_vert is None:
        blk_size_vert = blk_size

    Nh = blk_size * n_blocks
    Nv = blk_size_vert * n_blocks
    Thalf = np.int(np.floor(n_blocks / 2.0))
    THalf = np.int(np.ceil(n_blocks / 2.0))

    Pout = np.empty((blk_size_vert * n_blocks, blk_size * n_blocks))
    Pout[:blk_size_vert * THalf, :] = p_in
    for i in range(Thalf):
        for j in range(n_blocks):
            Pout[Nv - (i + 1) * blk_size_vert:Nv - i * blk_size_vert,
                 Nh - (j + 1) * blk_size:Nh - j * blk_size] \
                = p_in[i * blk_size_vert:(i + 1) *
                       blk_size_vert,
                       j * blk_size:(j + 1) * blk_size]

    return Pout

def rdiv(a, b):
    """
    Returns the solution to x b = a. Equivalent to MATLAB right matrix
    division: a / b
    """
    return np.linalg.solve(b.T, a.T).T

def logdet(A):
    """
    log(det(A)) where A is positive-definite.
    This is faster and more stable than using log(det(A)).
    Written by Tom Minka
    (c) Microsoft Corporation. All rights reserved.
    """
    U = np.linalg.cholesky(A)
    return 2 * (np.log(np.diag(U))).sum()
    
# note that this is code from the elephant program
def orthogonalize(highDimSpikes, unorthCMat):
    xDim = unorthCMat.shape[1]
    if xDim == 1:
        TT = np.sqrt(np.dot(unorthCMat.T, unorthCMat))
        Lorth = rdiv(unorthCMat, TT)
        Xorth = np.dot(TT, highDimSpikes)
    else:
        UU, DD, VV = sp.linalg.svd(unorthCMat, full_matrices=False)
        # TT is transform matrix
        TT = np.dot(np.diag(DD), VV)

        Lorth = UU
        Xorth = np.dot(TT, highDimSpikes)
    return Xorth, Lorth, TT

def makeKBig(params, T, covType = 'rbf', eng=None):
    if eng is None:
        eng = prepareMatlab()
        
    xDim = params['C'].shape[1]
    
    idx = np.arange(start=0, stop=xDim*T, step=xDim)
    KAll = np.zeros((xDim*T, xDim*T))
    KAllInv = np.zeros_like(KAll)
    logdetKAll = 0
    time = np.arange(T)[:,None]
    timeDiff = time - time.T
    
    for dim in range(xDim):
        if covType is 'rbf':
            K = (1 - params['eps'][0,dim]) * np.exp(-params["gamma"][0,dim] / 2 * np.power(timeDiff, 2)) + params["eps"][0,dim] * np.eye(T);
            
        else:
            raise Exception("GPFA:CovType", "unknown or unprogrammed covariance type")
            
        KAll[(idx+dim)[:,None], idx+dim] = K
        eng.workspace['K'] = mc.convertNumpyArrayToMatlab(K)
        eng.evalc("[K_big_inv, logdet_K] = invToeplitz(K);")
        KAllInv[(idx+dim)[:,None], idx+dim] = mc.convertMatlabArrayToNumpy(eng.eval("K_big_inv"))
        logdetKAll += mc.convertMatlabArrayToNumpy(eng.eval("logdet_K"))
    
    
    return KAll, KAllInv, logdetKAll
    
def parallelGpfa(fname, cvNum, xDim, sqTrn, sqTst, forceNewGpfaRun, binWidth, segLength, seqTrainStr, seqTestStr):

    print('Running GPFA crossvalidation #%d' % (cvNum+1))
    
    # eng was input and should be on... but let's check
    eng = prepareMatlab(None)
    # eng = pMat(engA)
        
    fnameOutput = fname / ("%s_xDim%02d_cv%02d" % ("gpfa", xDim, cvNum))
    if not fnameOutput.with_suffix('.mat').exists() or forceNewGpfaRun:
        eng.workspace['seqTrain'] = sqTrn
        eng.workspace['seqTest'] = sqTst
        eng.workspace['binWidth'] = binWidth*1.0
        eng.workspace['xDim'] = xDim*1.0
        eng.workspace['fname'] = str(fnameOutput)
        eng.workspace['segLength'] = np.inf #segLength*1.0
        
        
        
        eng.eval("gpfaEngine(" + seqTrainStr + ", " + seqTestStr + ", fname, 'xDim', xDim, 'binWidth', binWidth, 'segLength', segLength)", nargout=0)

    eng.evalc("results = load('"+str(fnameOutput)+"')")
    eng.evalc("results.method='gpfa'") # this is needed for the next step...
    eng.evalc("[estParams, seqTrain, seqTest] = postprocess(results)")
    
    estParams = mc.convertMatlabDictToNumpy(eng.eval('estParams'))
    seqsTrainNew = [mc.convertMatlabDictToNumpy(eng.eval("seqTrain(" + str(seqNum+1) + ")")) for seqNum in range(len(sqTrn))]
    seqsTestNew = [mc.convertMatlabDictToNumpy(eng.eval("seqTest(" + str(seqNum+1) + ")")) for seqNum in range(len(sqTst))]
    
    eng.exit()
    
    
    print('Finished GPFA crossvalidation #%d' % (cvNum+1))
    
    return estParams, seqsTrainNew, seqsTestNew

def cvalSuperWrap(paramsGpfa, allSeqsTest):
    llTemp = []
    eng = prepareMatlab()
                    
    for cvNum, (paramsEst, seqsTest) in enumerate(zip(paramsGpfa['allEstParams'], allSeqsTest)):
        print('Computing LL of crossvalidation #%d' % (cvNum + 1))
        llTemp.append(cvalWrap(seqsTest, paramsEst, eng))
        print('LL of crossvalidation #%d computed' % (cvNum+1))
        
    return llTemp

def cvalWrap(seqsTest, paramsEst, eng):
    
    
    unT, seqWithT = np.unique(np.array([seq['T'] for seq in seqsTest]), return_inverse=True)
                
    yDim, xDim = paramsEst['C'].shape
    
    if paramsEst['notes']['RforceDiagonal']:
        diagR = np.diag(paramsEst['R'])
        Rinv = np.diag(1/diagR)
        logdetR = np.sum(np.log(diagR))
    else:
        raise Exception("GPFA:RforceDiagonal", "Dunno what to do if the diagonal's not forced...")
        
    CRinv  = paramsEst["C"].T @ Rinv;
    CRinvC = CRinv @ paramsEst["C"];
    
    llT = []
    for uniqueT in unT:
        llT.append(cvalRun(paramsEst, seqsTest, uniqueT,  Rinv, CRinv, CRinvC, logdetR, xDim, yDim, eng=eng))
       
    llT = np.sum(llT)/2 # following Matlab GPFA code, but unsure why there's a divide-by-2 here...
    
    return llT

def cvalRun(paramsEst, seqsTest, uniqueT, Rinv, CRinv, CRinvC, logdetR, xDim, yDim, eng=None):
    
    if eng is None:
        print('what')
        
    uniqueT = int(uniqueT)
    seqsUse = [seq for seq in seqsTest if seq['T'] == uniqueT]
    
    KAll, KAllInv, logdetKAll = makeKBig(paramsEst, uniqueT, eng=eng)
    
    listCRinvC = [CRinvC for numTms in range(uniqueT)]
    blockDiagCRinvC = sp.linalg.block_diag(*listCRinvC)

    
    invM, logdetM = invPersymm(KAllInv + blockDiagCRinvC, xDim)
    
    
    # idx = np.arange(start=0, stop=xDim*T, step=xDim)
    
    # Vsm = np.dstack([invM[idxH + np.arange(xDim),idxH + np.arange(xDim)] for idxH in idx])
    # VsmGP = np.dstack([invM[idx+stp,idx+stp] for stp in range(xDim)])
    
    dif = np.hstack([seq['y'] for seq in seqsUse]) - paramsEst['d']
    
    term1Mat = CRinv @ dif
    term1Mat = term1Mat.reshape((xDim*uniqueT, -1), order = 'F')
    
    # compute log likelihood
    val = -uniqueT * logdetR - logdetKAll - logdetM -yDim * uniqueT * np.log(2*np.pi) 
    ll = len(seqsUse) * val - np.sum(np.sum((Rinv @ dif) * dif)) + np.sum(np.sum((term1Mat.T @ invM) * term1Mat.T))
    return ll

def shuffParallel(binnedSpikes, testInds, estParams, cvalTest, numShuffle, eng):

    unShuffTestInds = testInds[cvalTest]
    shuffPerChan = []

    # numberTrials
    numTestTrials = unShuffTestInds.shape[0]
    # number of channels
    numChans = binnedSpikes[0].shape[0]

    # new chan x shuffle matrix that tells us, for each channel, which trial to choose
    # (it gives an integer index trial to use)
    shuffTestInds = np.random.randint(low=0, high=numTestTrials, size=(numChans, numShuffle))

    binnedSpikesOrig = np.asarray([[spCh for spCh in spTrl] for spTrl in binnedSpikes.copy()])
    binnedSpikesTestOrig = binnedSpikesOrig[unShuffTestInds]

    binnedSpikesChanFirst = binnedSpikesTestOrig.swapaxes(0, 1) # put channels first
    

    binnedSpikesShuff = np.stack([bnSpCh[shf] for bnSpCh, shf in zip(binnedSpikesChanFirst, shuffTestInds)])
    binnedSpikesShuffByTrial = binnedSpikesShuff.swapaxes(0,1) # trials first again


    seqShuffDict = GPFA.binSpikesToGpfaInputDict([], binnedSpikes = binnedSpikesShuffByTrial)

    seqShuffNew = projectTrajectory(seqShuffDict, estParams, eng)
    return seqShuffNew

def projectTrajectory(seqDict, estParams, eng=None):
    # eng is sometimes not input...
    if eng is None:
        eng = prepareMatlab(None)

    eng.workspace['seqProj'] = seqDict
    eng.workspace['estParams'] = mc.convertNumpyDictToMatlab(estParams)
    # because lists become cells, we must combine them again to structs
    eng.evalc('seqProj = [seqProj{:}];')
    eng.evalc("seqProj = exactInferenceWithLL(seqProj, estParams);")
    eng.evalc("X = [seqProj.xsm];")
    eng.evalc("C = estParams.C;")
    eng.evalc("[Xorth, Corth] = orthogonalize(X, C);")
    eng.evalc("seqProj = segmentByTrial(seqProj, Xorth, 'xorth');")

    # matlab engine can only output one struct at a time
    numProj = int(eng.eval("length(seqProj)"))
    seqShuffNew = [mc.convertMatlabDictToNumpy(eng.eval('seqProj(%d)' % (seqNum+1))) for seqNum in range(numProj)]

    return seqShuffNew


