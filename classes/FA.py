"""
This has been done before... but to make things kinda my way this file will
parallel GPFA.py in output and calling but for factor analysis
"""
import numpy as np
from classes.FactorAnalysis import FactorAnalysis
from methods.MatFileMethods import LoadMatFile

class FA:
    def __init__(self, binnedSpikes, crossvalidateNum):
        self.dimOutput = {}
        self.binnedSpikes = binnedSpikes
        self.crossvalidateNum = crossvalidateNum
        self.trainInds, self.testInds =  self.computeTrainTestIndRandom(numFolds = crossvalidateNum)

    def computeTrainTestIndRandom(self, numFolds = 4, initSeed = 0):
        numTrl = self.binnedSpikes.shape[0]
        
        initSt = np.random.get_state()
        np.random.seed(seed=initSeed)
        testInds = []
        trainInds = []
        for fld in range(numFolds):
            randomizedIndOrder = np.random.permutation(np.arange(numTrl))
            testInds.append(randomizedIndOrder[:round((1/numFolds)*numTrl)])
            trainInds.append(randomizedIndOrder[round((1/numFolds)*numTrl):])
        
        np.random.set_state(initSt)
        
        return trainInds, testInds  # for splitting binned spikes into train and test sets

    def runFa(self,  numDim = 8, gpfaResultsPath = None):
        ll = np.ndarray(0)
        faScore = np.ndarray(0)
        fADims = []
        bnSp = self.binnedSpikes

        trainInds = self.trainInds
        testInds = self.testInds
        if self.crossvalidateNum == 1: # effectively no crossvalidations
            trainInds = testInds


        self.dimOutput[numDim] = {}
        self.dimOutput[numDim]['crossvalidateNum'] = self.crossvalidateNum
        allEstParams = []
        seqsTrainNew = []
        seqsTestNew = []
        fullRank = []
        for cVal, (trnInd, tstInd) in enumerate(zip(trainInds, testInds)):
            fA = FactorAnalysis(n_components = numDim)

            if bnSp.dtype=='object':
                # I'm honestly not really sure why this works, given that with the
                # axis=1 in the else statement, axis 1 stays the same size, but in
                # this statement with axis=2, axis 2 *changes size*...
                concatTrlTrain = np.concatenate(bnSp[trnInd, :], axis=2)[0]
                concatTrlTest = np.concatenate(bnSp[tstInd, :], axis=2)[0]
            else:
                concatTrlTrain = np.concatenate(bnSp[trnInd, :], axis=1)
                concatTrlTest = np.concatenate(bnSp[tstInd, :], axis=1)

            if gpfaResultsPath is not None:
            fullRank.append(np.linalg.matrix_rank(concatTrlTest) >= concatTrlTest.shape[0])

                cValGpfaResultsPath = gpfaResultsPath / ("%s_xDim%02d_cv%02d.mat" % ("gpfa", numDim, cVal))

            if gpfaResultsPath is None or not cValGpfaResultsPath.exists():

                try:
                    # Note that we transpose here because FactorAnalysis class
                    # expects a samples x features matrix, and here the trials
                    # will be samples and the channels will be features. (Before
                    # transposing it comes in as a channels x trials matrix...)
                    fA.fit(concatTrlTrain.T)
                except Exception as e:
                    if e.args[0] == "FA:NumObs":
                        print(e.args[1])
                        faScore = np.append(faScore, np.nan)
                        continue
                    else:
                        raise(e)
                
                ll = np.append(ll, fA.loglike_[-1])
                fADims.append(fA.components_)
                faScore = np.append(faScore, fA.score(concatTrlTest.T))

                L = fA.components_
                psi = fA.noise_variance_
                d = fA.mean_
                converged = fA.n_iter_ < fA.max_iter
                finalRatioChange = fA.finalRatioChange_
                finalDiffChange = fA.finalDiffChange_
                Lorth, singVal, rightSingVec = np.linalg.svd(L)
                Lorth = Lorth[:, :L.shape[1]]
                allEstParams.append({
                    'C': L, # here I'm matching GPFA naming of C
                    'Corth': Lorth, # here I'm matching GPFA naming of C
                    'd': d,
                    'R': np.diag(psi), # here I'm matching GPFA naming of R...
                    'converge' : converged, # keeping track of whether we converged...
                    'finalRatioChange' : finalRatioChange,
                    'finalDiffChange' : finalDiffChange,
                })
                
            else:
                breakpoint()
                cValGpfaResults = LoadMatFile(cValGpfaResultsPath)
                faParams =  cValGpfaResults['faParams']
                ll =  np.append(ll, cValGpfaResults['faLL'][0,-1])

                # used for the projection below
                L = faParams['L']
                d = faParams['d'].squeeze() # mean is only single dimensional
                Ph = faParams['Ph'].squeeze() # noise variance is saved as single-dimensional diagonal elements
                Lorth, singVal, rightSingVec = np.linalg.svd(L)
                Lorth = Lorth[:, :L.shape[1]]

                fADims.append(L)

                allEstParams.append({
                    'C': L, # here I'm matching GPFA naming of C
                    'Corth': Lorth, # here I'm matching GPFA naming of Corth
                    'd': d,
                    'R': np.diag(Ph), # here I'm matching GPFA naming of R...
                })

                fA.components_ = L
                fA.noise_variance_ = Ph
                fA.mean_ = d

                faScore = np.append(faScore, fA.score(concatTrlTest.T))

            # here I'm again matching to the GPFA output... (I'm trying to use
            # the same downstream code, see...
            projTrainBnSp = np.stack([(fA.transform(bS.T)).T for bS in bnSp[trnInd, :].view(np.ndarray)])
            projTestBnSp = np.stack([(fA.transform(bS.T)).T for bS in bnSp[tstInd, :].view(np.ndarray)])
            seqsTrainNew.append([{'trialId' : trlNum, 'xsm' : trlProj, 'y': trlOrig} for trlNum, (trlProj, trlOrig) in enumerate(zip(projTrainBnSp, bnSp[trnInd, :].view(np.ndarray)))])
            seqsTestNew.append([{'trialId' : trlNum, 'xsm' : trlProj, 'y': trlOrig} for trlNum, (trlProj, trlOrig) in enumerate(zip(projTestBnSp, bnSp[tstInd, :].view(np.ndarray)))])


        self.dimOutput[numDim]['allEstParams'] = allEstParams
        self.dimOutput[numDim]['seqsTrainNew'] = seqsTrainNew
        self.dimOutput[numDim]['seqsTestNew'] = seqsTestNew
        self.dimOutput[numDim]['fullRank'] = fullRank

        return faScore, allEstParams, seqsTrainNew, seqsTestNew, fullRank

