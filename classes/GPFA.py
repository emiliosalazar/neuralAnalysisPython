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
    def __init__(self, binnedSpikes, firingRateThresh=5):
        from classes.BinnedSpikeSet import BinnedSpikeSet
        allTrlTogether = np.concatenate(binnedSpikes, axis=1)[None, :, :].view(BinnedSpikeSet)
        allTrlTogether.binSize = binnedSpikes[0].binSize
        _, chansKeep = allTrlTogether.channelsAboveThresholdFiringRate(firingRateThresh=firingRateThresh)
        # chansKeepCoinc = allTrlTogether.removeCoincidentSpikes()
        # chansKeep = np.logical_and(chansKeepThresh, chansKeepCoinc)
        self.binnedSpikes = [bnSp[chansKeep, :] for bnSp in binnedSpikes]
        self.gpfaSeqDict = self.binSpikesToGpfaInputDict()
        self.trainInds = None
        self.testInds = None
        self.dimOutput = {}
        
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
    
    # This probably won't be used much, but it's here for generating random 
    # crossvalidation train/test sequences. Since it's unclear whether sequential
    # ones would overlap, however, it makes more sense to use the function that
    # generates them in order
    # NOTE should be updated to return numFolds sets of sequences!
    def computeTestTrainIndRandom(self, numFolds = 4):
        seqAll = self.gpfaSeqDict
        numTrls = len(seqAll)
        
        initSt = np.random.get_state()
        np.random.seed(seed=0)
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
    def computeTestTrainIndOrdered(self, numFolds=4):
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
            
        
        
    
    def runGpfaInMatlab(self, fname="~/Documents/BatistaLabData/analysesOutput/gpfaTest", runDescriptor = "", condDescriptor = "", crossvalidateNum = 0, xDim = 8, eng=None, segLength = None):
        forceNewGpfaRun = False # I'm gonna keep this flag in here for now... you shouldn't *want* to run another GPFA, right?
        
        fname = fname / "gpfa" / ("run" + str(runDescriptor)) / ("cond" + str(condDescriptor))
        fname.mkdir(parents=True, exist_ok = True)
        # if eng is None:
        #     from matlab import engine 
        #     eng = engine.start_matlab()
        
        if not crossvalidateNum:
            seqTrain = self.gpfaSeqDict
            seqTest = []
            seqTrainStr = "[seqTrain{:}]"
            seqTestStr = "subsref([seqTrain{:}], struct('type', '()', 'subs', {{[]}}))"
        else:
            # NOTE: INCOMPLETE!
            # self.computeTestTrainIndRandom(numFolds = crossvalidateNum)
            self.trainInds, self.testInds = self.computeTestTrainIndOrdered(numFolds = crossvalidateNum)
            seqsTrain, seqsTest = self.trainAndTestSeq()
            seqTrainStr = "[seqTrain{:}]"
            seqTestStr = "[seqTest{:}]"
            
        if segLength is None:
            segLength = min([sq['T'] for sq in self.gpfaSeqDict])
            
        binWidth = self.binSize
        
        seqsTrainNew = []
        seqsTestNew = []
        allEstParams = []
        from multiprocessing import Pool
        
        
        
        with Pool() as poolHere:
            res = []
            # from matlab.engine import pythonengine
            # engA = pythonengine.getMATLAB
            engA = None
            for cvNum, (sqTrn, sqTst) in enumerate(zip(seqsTrain, seqsTest)):
                
                res.append(poolHere.apply_async(parallelGpfa, (fname, cvNum, xDim, sqTrn, sqTst, forceNewGpfaRun, binWidth, segLength, seqTrainStr, seqTestStr)))
            
            # from time import sleep
            # print('blah')
            # sleep(20)
            # print('bleh')
            resultsByCrossVal = [rs.get() for rs in res]    
            
            poolHere.close()
            poolHere.join()
            
            resultsByVar = list(zip(*resultsByCrossVal))
            allEstParams = resultsByVar[0]
            seqsTrainNew = resultsByVar[1]
            seqsTestNew = resultsByVar[2]
            
        
        # print('whowhat?')
        self.dimOutput[xDim] = {}
        self.dimOutput[xDim]['crossvalidateNum'] = crossvalidateNum
        self.dimOutput[xDim]['allEstParams'] = allEstParams
        self.dimOutput[xDim]['seqsTrainNew'] = seqsTrainNew
        self.dimOutput[xDim]['seqsTestNew'] = seqsTestNew
            
        # self.crossvalidatedGpfaError(eng = eng)
        return allEstParams, seqsTrainNew, seqsTestNew
    
    def crossvalidatedGpfaError(self, approach = "logLikelihood", eng=None):
        from multiprocessing import Pool
        
        # if eng is None:
        #     from methods.GeneralMethods import prepareMatlab
        #     eng = prepareMatlab()
        eng = None
        
        if approach is "logLikelihood":
            _, allSeqsTest = self.trainAndTestSeq()
            
                
            ll = {}
            llErr = {}
            numFolds = len(allSeqsTest)
            
            
            
            for dimMax, paramsGpfa in self.dimOutput.items():
                llTemp = []
                
                with Pool() as plWrap:
                    res = []
                    for cvNum, (paramsEst, seqsTest) in enumerate(zip(paramsGpfa['allEstParams'], allSeqsTest)):
                        res.append(plWrap.apply_async(cvalWrap, (seqsTest, paramsEst, cvNum+1)))
                        
                    resultsByCrossVal = [rs.get() for rs in res]
                    
                    resultsByVar = list(zip(*resultsByCrossVal))
                    
                    llTemp = resultsByVar[0]
                
                    
                    
                # llTemp = np.stack(llTemp)
                # llTemp = llTemp - np.min(llTemp)
                # llTemp = llTemp/np.max(llTemp)
                ll[dimMax] = np.stack(llTemp)#np.mean(np.stack(llTemp))
                # llErr[dimMax] = np.std(np.stack(llTemp))
                    
            reducedGpfaScore = np.stack([np.nan])
            normalGpfaScore = np.stack([llH for _, llH in ll.items()])
            normalizedGpfaScore = (normalGpfaScore - np.min(normalGpfaScore, axis=0))/(np.max(normalGpfaScore,axis=0)-np.min(normalGpfaScore,axis=0))
            normalGpfaScore = np.mean(normalizedGpfaScore, axis=1)
            normalGpfaScoreErr = np.std(normalizedGpfaScore,axis=1)
            
            
            
                    
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
            normalGpfaScore = np.stack([np.mean(gpfaDim[maxDim-1]) for maxDim, gpfaDim in errs.items()])
            normalGpfaScore = np.stack([np.std(gpfaDim[maxDim-1]) for maxDim, gpfaDim in errs.items()])
            
        return normalGpfaScore, normalGpfaScoreErr, reducedGpfaScore
    
    
    
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
    
    
def parallelGpfa(fname, cvNum, xDim, sqTrn, sqTst, forceNewGpfaRun, binWidth, segLength, seqTrainStr, seqTestStr):
    from methods.GeneralMethods import pMat, prepareMatlab

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
        eng.workspace['segLength'] = segLength*1.0
        
        
        
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

def cvalWrap(seqsTest, paramsEst, cvNum):
    print('Computing LL of crossvalidation #%d' % cvNum)
    from methods.GeneralMethods import prepareMatlab
    eng = prepareMatlab()
    
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
        
    print('LL of crossvalidation #%d computed' % cvNum)
    
    return llT

def cvalRun(paramsEst, seqsTest, uniqueT, Rinv, CRinv, CRinvC, logdetR, xDim, yDim, eng=None):
    from scipy.linalg import block_diag
    
    if eng is None:
        from methods.GeneralMethods import prepareMatlab
        print('what')
        
    uniqueT = int(uniqueT)
    seqsUse = [seq for seq in seqsTest if seq['T'] == uniqueT]
    
    KAll, KAllInv, logdetKAll = makeKBig(paramsEst, uniqueT, eng=eng)
    
    listCRinvC = [CRinvC for numTms in range(uniqueT)]
    blockDiagCRinvC = block_diag(*listCRinvC)
    
    invM, logdetM = invPersymm(KAllInv + blockDiagCRinvC, xDim)
    
    
    # idx = np.arange(start=0, stop=xDim*T, step=xDim)
    
    # Vsm = np.dstack([invM[idxH + np.arange(xDim),idxH + np.arange(xDim)] for idxH in idx])
    # VsmGP = np.dstack([invM[idx+stp,idx+stp] for stp in range(xDim)])
    
    dif = np.hstack([seq['y'] for seq in seqsUse]) - paramsEst['d']
    
    term1Mat = CRinv @ dif
    term1Mat = term1Mat.reshape((xDim*uniqueT, -1), order = 'F')
    
    # Thalf = np.ceil(uniqueT/2)
    # idx = np.arange(start=0, stop=xDim*Thalf, step=xDim)
    
    # blkProd = np.vstack([CRinvC @ invM[idxH + np.arange(xDim), :] for idxH in idx])
    # blkProd = KAll[:xDim*Thalf, :] @ self.fillPerSymm(np.eye(xDim*Thalf, xDim*T), xDim, T)
    
    # xsmMat  = self.fillPerSymm(blkProd, xDim, uniqueT) @ term1Mat
    
    # compute log likelihood
    val = -uniqueT * logdetR - logdetKAll - logdetM -yDim * uniqueT * np.log(2*np.pi) 
    ll = len(seqsUse) * val - np.sum(np.sum((Rinv @ dif) * dif)) + np.sum(np.sum((term1Mat.T @ invM) * term1Mat.T))
    return ll

def makeKBig(params, T, covType = 'rbf', eng=None):
    if eng is None:
        from methods.GeneralMethods import prepareMatlab
        print('what2')
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