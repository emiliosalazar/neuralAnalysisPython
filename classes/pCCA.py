"""
This has been done before... but to make things kinda my way this file will
parallel GPFA.py in output and calling but for factor analysis
"""
import numpy as np
from classes.ProbabilisticCanonicalCorrelationAnalysis import ProbabilisticCanonicalCorrelationAnalysis
from methods.MatFileMethods import LoadMatFile

class pCCA:
    def __init__(self, binnedSpikesAreaA, binnedSpikesAreaB, crossvalidateNum):
        self.dimOutput = {}
        self.binnedSpikesA = binnedSpikesAreaA
        self.binnedSpikesB = binnedSpikesAreaB
        self.crossvalidateNum = crossvalidateNum
        self.trainInds, self.testInds =  self.computeTrainTestIndRandom(numFolds = crossvalidateNum)

    def computeTrainTestIndRandom(self, numFolds = 4, initSeed = 0):
        numTrl = self.binnedSpikesA.shape[0]
        
        initSt = np.random.get_state()
        np.random.seed(seed=initSeed)
        randomizedIndOrder = np.random.permutation(np.arange(numTrl))
        testInds = []
        trainInds = []
        for fld in range(numFolds):
            testInds.append(randomizedIndOrder[int(numTrl*fld/numFolds):int(numTrl*(fld+1)/numFolds)])
            trainInds.append(np.hstack([randomizedIndOrder[:int(numTrl*fld/numFolds)], randomizedIndOrder[int(numTrl*(fld+1)/numFolds):]]))
        
        np.random.set_state(initSt)
        
        return trainInds, testInds  # for splitting binned spikes into train and test sets

    def runPcca(self,  numDir = 8, gpfaResultsPath = None):
        ll = np.ndarray(0)
        pccaScore = np.ndarray(0)
        pccaDirs = []
        bnSpA = self.binnedSpikesA
        bnSpB = self.binnedSpikesB

        trainInds = self.trainInds
        testInds = self.testInds
        if self.crossvalidateNum == 1: # effectively no crossvalidations
            trainInds = testInds


        self.dimOutput[numDir] = {}
        self.dimOutput[numDir]['crossvalidateNum'] = self.crossvalidateNum
        allEstParams = []
        seqsTrainNew = []
        seqsTestNew = []
        fullRank = []
        for cVal, (trnInd, tstInd) in enumerate(zip(trainInds, testInds)):
            pccA = ProbabilisticCanonicalCorrelationAnalysis(n_components = numDir)

            if bnSpA.dtype=='object':
                # I'm honestly not really sure why this works, given that with the
                # axis=1 in the else statement, axis 1 stays the same size, but in
                # this statement with axis=2, axis 2 *changes size*...
                concatTrlTrainA = np.concatenate(bnSpA[trnInd, :], axis=2)[0]
                concatTrlTestA = np.concatenate(bnSpA[tstInd, :], axis=2)[0]
                concatTrlTrainB = np.concatenate(bnSpB[trnInd, :], axis=2)[0]
                concatTrlTestB = np.concatenate(bnSpB[tstInd, :], axis=2)[0]
            else:
                concatTrlTrainA = np.concatenate(bnSpA[trnInd, :], axis=1)
                concatTrlTestA = np.concatenate(bnSpA[tstInd, :], axis=1)
                concatTrlTrainB = np.concatenate(bnSpB[trnInd, :], axis=1)
                concatTrlTestB = np.concatenate(bnSpB[tstInd, :], axis=1)

            fullRank.append((np.linalg.matrix_rank(concatTrlTestA) >= concatTrlTestA.shape[0]) & (np.linalg.matrix_rank(concatTrlTestB) >= concatTrlTestB.shape[0]))

            if gpfaResultsPath is not None:
                cValGpfaResultsPath = gpfaResultsPath / ("%s_xDim%02d_cv%02d.mat" % ("gpfa", numDir, cVal))

            if gpfaResultsPath is None or not cValGpfaResultsPath.exists():

                try:
                    # Note that we transpose here because ProbabilisticCanonicalCorrelationAnalysis class
                    # expects a samples x features matrix, and here the trials
                    # will be samples and the channels will be features. (Before
                    # transposing it comes in as a channels x trials matrix...)
                    pccA.fit(X_1 = concatTrlTrainA.T, X_2 = concatTrlTrainB.T)
                except Exception as e:
                    if e.args[0] == "pCCA:NumObs":
                        print(e.args[1])
                        pccaScore = np.append(pccaScore, np.nan)
                        continue
                    else:
                        raise(e)
                
                ll = np.append(ll, pccA.loglike_[-1])
                pccaDirs.append(pccA.components_1_)
                pccaDirs.append(pccA.components_2_)
                pccaScore = np.append(pccaScore, pccA.score(concatTrlTestA.T, concatTrlTestB.T))

                W = pccA.components_
                W1 = pccA.components_1_
                W2 = pccA.components_2_
                psi = pccA.noise_variance_
                psi1 = pccA.noise_variance_1_
                psi2 = pccA.noise_variance_2_
                d = pccA.mean_
                d1 = pccA.mean_1_
                d2 = pccA.mean_2_
                converged = pccA.n_iter_ < pccA.max_iter
                finalRatioChange = pccA.finalRatioChange_
                finalDiffChange = pccA.finalDiffChange_
                allEstParams.append({
                    'C': W, # here I'm matching GPFA naming of C
                    'C1': W1, # here I'm matching GPFA naming of C
                    'C2': W2, # here I'm matching GPFA naming of C
                    'd': d, 
                    'd1': d1,
                    'd2': d2,
                    'R': psi, # here I'm matching GPFA naming of R...
                    'R1': psi1, # here I'm matching GPFA naming of R...
                    'R2': psi2, # here I'm matching GPFA naming of R...
                    'converge' : converged, # keeping track of whether we converged...
                    'finalRatioChange' : finalRatioChange,
                    'finalDiffChange' : finalDiffChange,
                })
                
            else:
                breakpoint() 
                # NOTE: forget why the breakpoint is here, but I think we want
                # to do something with ratio changes/etc that is expected in
                # allEstParams?
                cValGpfaResults = LoadMatFile(cValGpfaResultsPath)
                faParams =  cValGpfaResults['faParams']
                ll =  np.append(ll, cValGpfaResults['faLL'][0,-1])

                # used for the projection below
                L = faParams['L']
                d = faParams['d'].squeeze() # mean is only single dimensional
                Ph = faParams['Ph'].squeeze() # noise variance is saved as single-dimensional diagonal elements
                Lorth, singVal, rightSingVec = np.linalg.svd(L)
                Lorth = Lorth[:, :L.shape[1]]

                pccaDirs.append(L)

                allEstParams.append({
                    'C': L, # here I'm matching GPFA naming of C
                    'Corth': Lorth, # here I'm matching GPFA naming of Corth
                    'd': d,
                    'R': np.diag(Ph), # here I'm matching GPFA naming of R...
                })

                pccA.components_ = L
                pccA.noise_variance_ = Ph
                pccA.mean_ = d

                pccaScore = np.append(pccaScore, pccA.score(concatTrlTest.T))

            # here I'm again matching to the GPFA output... (I'm trying to use
            # the same downstream code, see...
            projTrainBnSp = np.stack([(pccA.transform(bSA.T, bSB.T)).T for bSA, bSB in zip(bnSpA[trnInd, :].view(np.ndarray),bnSpB[trnInd, :].view(np.ndarray))])
            projTestBnSp = np.stack([(pccA.transform(bSA.T, bSB.T)).T for bSA, bSB in zip(bnSpA[tstInd, :].view(np.ndarray),bnSpB[tstInd, :].view(np.ndarray))])
            seqsTrainNew.append([{'trialId' : trlNum, 'xsm' : trlProj, 'yA': trlAOrig, 'yB': trlBOrig} for trlNum, (trlProj, trlAOrig, trlBOrig) in enumerate(zip(projTrainBnSp, bnSpA[trnInd, :].view(np.ndarray),bnSpB[trnInd, :].view(np.ndarray)))])
            seqsTestNew.append([{'trialId' : trlNum, 'xsm' : trlProj, 'yA': trlAOrig, 'yB': trlBOrig} for trlNum, (trlProj, trlAOrig, trlBOrig) in enumerate(zip(projTestBnSp, bnSpA[tstInd, :].view(np.ndarray),bnSpB[tstInd, :].view(np.ndarray)))])


        self.dimOutput[numDir]['allEstParams'] = allEstParams
        self.dimOutput[numDir]['seqsTrainNew'] = seqsTrainNew
        self.dimOutput[numDir]['seqsTestNew'] = seqsTestNew
        self.dimOutput[numDir]['fullRank'] = fullRank

        return pccaScore, allEstParams, seqsTrainNew, seqsTestNew, fullRank

