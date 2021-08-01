"""
We're going to be putting some GPFA related methods here that are separate from
what the class can do
"""

from pathlib import Path
import numpy as np

def computePopulationMetrics(gpfaResultsByExtractionParams, logLikelihoodDimensionality, dimensionalityExtractionParams, binSizeMs):

    # prep the outputs we're getting
    coefDetLowDByExParams = []
    coefDetByLatentByExParams = []
    shCovPropPopByExParams = []
    shCovPropNeurAvgByExParams = []
    shCovPropNeurStdByExParams = []
    shCovPropNeurNormDimByExParams = []
    shCovPropByLatentByExParams = []
    shCovPropNeurGeoNormDimExParams = []
    privVarSpreadExParams = []
    ffLoadingSimByExParams = []
    overallLoadingSimByExParams = []
    efLoadingSimByExParams = []
    tmscAllByExParams = []
    tmscMnsByExParams = []
    tmscStdsByExParams = []
    varExpByExParams = []
    varExpByDimByExParams = []
    shVarGeoMnVsMnByExParams = []
    participationRatioByExParams = []
    participationRatioRawCovByExParams = []
    for gpfaResult, dimsGpfaUse in zip(gpfaResultsByExtractionParams, logLikelihoodDimensionality):
        # lists of subsamples; often of area subsamples, but sometimes could be
        # subsamples with certain parameters (say, neuron number or trial
        # number per condition)
        if type(gpfaResult) is not list:
            gpfaResult = [gpfaResult]

        coefDetLowDBySubset = []
        coefDetByLatentBySubset = []
        shCovPropPopBySubset = []
        shCovPropNeurAvgBySubset = []
        shCovPropNeurStdBySubset = []
        shCovPropNeurNormDimBySubset = []
        shCovPropByLatentBySubset = []
        shCovPropNeurGeoNormDimBySubset = []
        privVarSpreadBySubset = []
        ffLoadingSimBySubset = []
        overallLoadingSimBySubset = []
        efLoadingSimBySubset = []
        tmscAllBySubset = []
        tmscMnsBySubset = []
        tmscStdsBySubset = []
        varExpBySubset = []
        varExpByDimBySubset = []
        shVarGeoMnVsMnBySubset = []
        participationRatioBySubset = []
        participationRatioRawCovBySubset = []
        for (gpSubset, dGU) in zip(gpfaResult, dimsGpfaUse):
            # extract C and R parameters
            CR = []
            CERorth = []
            Corth = []
            timescales = []
            coefDetLowD = []
            coefDetByLatent = []
            varExpTog = []
            varExpByDim = []
            for gpfaCond, llDim in zip(gpSubset, dGU):
                C = gpfaCond[int(llDim)]['allEstParams'][0]['C']
                testSeqsOrigAndInferred = gpfaCond[int(llDim)]['seqsTestNew'][0]
                if C.shape[1] != llDim:
                    # I at some point was indexing this, but I no longer
                    # think that's necessary because I'm now using the log
                    # likelihood dimensionality, not the Williamson
                    # dimensionality
                    breakpoint()
                    raise Exception("AAAH")
                R = gpfaCond[int(llDim)]['allEstParams'][0]['R']
                if 'Corth' in gpfaCond[int(llDim)]['allEstParams'][0]:
                    Co = gpfaCond[int(llDim)]['allEstParams'][0]['Corth']
                else:
                    Co, sval, _ = np.linalg.svd(C)
                    if not np.all(sval[:-1] >= sval[1:]):
                        breakpoint() # careful, singular values aren't sorted
                    Co = C[:, :C.shape[1]]

                # doing some timescale things here, including checking how much
                # variance is explained by each dimension (which has an
                # associated timescale)
                if 'gamma' in gpfaCond[int(llDim)]['allEstParams'][0]:
                    timescale = gpfaCond[int(llDim)]['allEstParams'][0]['gamma']
                    if llDim == 1:
                        timescale = np.array([[timescale]]) # make it a 2d scalar array
                else:
                    timescale = np.array([[np.nan]])
                # checking for the 'shape' lets me see if xsm is matrix
                # multiplicable or a scalar--I could also check if it's an
                # ndarray... but I'm actually not sure what typically comes
                # out of how xsm is made... so I'll just leave this here
                # for now
                lowDSeqsNonorth = [seq['xsm'] if hasattr(seq['xsm'], 'shape') else np.array(seq['xsm'])[None,None] for seq in testSeqsOrigAndInferred]
                lowDSeqsOrth = [seq['xorth'] if hasattr(seq['xorth'], 'shape') else np.array(seq['xorth'])[None,None] for seq in testSeqsOrigAndInferred]

                highDReprojAll = [C@xsm for xsm in lowDSeqsNonorth]
                stackIndTrls = np.dstack(lowDSeqsOrth)
                meanProj = stackIndTrls.mean(axis=2)
                meanOfMeanInEachDim = meanProj.mean(axis=1)
                varOfMean = ((meanProj.T - meanOfMeanInEachDim)**2).sum(axis=1).mean()
                varOfTrialsToMeanTraj = ((stackIndTrls.T-meanProj.T)**2).sum()/np.prod(stackIndTrls.shape[1:])
                varOfTrialsToOverallMean = ((stackIndTrls.T-meanOfMeanInEachDim)**2).sum()/np.prod(stackIndTrls.shape[1:])
                coefDetLowD.append(1 - varOfTrialsToMeanTraj/varOfTrialsToOverallMean)

                varOfTrialsToOverallByLatent = ((stackIndTrls.T-meanOfMeanInEachDim)**2).sum(axis=(0,1))/np.prod(stackIndTrls.shape[1:])
                varOfTrialsToMeanTrajByLatent = ((stackIndTrls.T-meanProj.T)**2).sum(axis=(0,1))/np.prod(stackIndTrls.shape[1:])
                coefDetByLatent.append(1 - varOfTrialsToMeanTrajByLatent/varOfTrialsToOverallByLatent)

                
                # the transpose on C lets me iterate through dimensions, the
                # dimensional expansion of [:,None] on cDim then effectively
                # undoes the tranpose
                highDReprojByDimAll = [[cDim[:,None] @ xsmDim[None, :] for cDim, xsmDim in zip(C.T, xsm)] for xsm in lowDSeqsNonorth ]
#                highDReprojByDimAll = [[cDim[:,None] @ xsmDim[None, :] for cDim, xsmDim in zip(C.T, xsm if hasattr(xsm, 'shape') else np.array(xsm)[None,None])] for xsm in lowDSeqsNonorth ]
                grpByDim = list(zip(*highDReprojByDimAll)) # was grouped by trial
                highDReprojByDimTog = [np.concatenate(dimProj, axis=1) for dimProj in grpByDim]
                origSeqs = [seq['y'] for seq in testSeqsOrigAndInferred]

                # no bueno... remove when ready
#                varExp = [1-np.sum((prjDm-yOrig) ** 2) / np.trace(yOrig@yOrig.T) for prjDm, yOrig in zip(highDReprojAll, origSeqs)]
                highDReprojTog = np.concatenate(highDReprojAll, axis=1)
                origSeqsTog = np.concatenate(origSeqs, axis=1)

                CR.append((C,R))
                CoRecalc, egsOrth, _ = np.linalg.svd(C)
                CoRecalc = CoRecalc[:, :egsOrth.size]
                if np.abs((CoRecalc - Co).mean()) > 1e-10:
                    breakpoint() # these should be identical to some really small error...
                CERorth.append((CoRecalc, egsOrth, R))
                Corth.append(Co)
                timescales.append(binSizeMs/np.sqrt(timescale))

            shCovPropPop = [np.trace(C @ C.T) / (np.trace(C @ C.T) + np.trace(R)) for C, R in CR] 
            shCovPropNeurAvg = [np.mean(np.diag(C @ C.T) / (np.diag(C @ C.T) + np.diag(R))) for C, R in CR] 
            shCovPropByLatent = [np.mean(np.diag(eVal**2 * C[:,None] @ C[:,None].T) / (np.diag(Call @ np.diag(eigAll) @ np.diag(eigAll) @ Call.T) + np.diag(R))) for Call, eigAll, Rall in CERorth for C, eVal in zip(Call.T, eigAll)] 
            shCovPropNeurStd = [np.std(np.diag(C @ C.T) / (np.diag(C @ C.T) + np.diag(R))) for C, R in CR] 
            shCovPropNeurNormDim = [np.mean(np.diag(C @ C.T) / (np.diag(C @ C.T) + np.diag(R)))/C.shape[1] for C, R in CR] 
            shCovPropNeurGeoNormDim = [np.mean(np.diag(C @ C.T) / (np.diag(C @ C.T) + np.diag(R)))**(1/C.shape[1]) if C.shape[1]>0 else np.array([0]) for C, R in CR] 
            privVarSpread = [np.diag(R).max()-np.diag(R).min() for C, R in CR]

            
            # NOTE: I'm doing all of this on the orthonormalized C! I believe
            # this is correct as it reflects the dimensions matched to the
            # eigenvalues--e.g. the first eigenvalue describes how much
            # variance is explained by the first orthonormalized dimension
            #
            # Also remember that Python is zero-indexed for grabbing that first
            # one >.>
            Cnorm1 = [C[:,0]/np.sqrt((C[:,0]**2).sum()) if C.shape[1]>0 else np.array([np.nan]) for C in Corth]
            firstFactorLoadingSimilarity = [1-Cn1.size*Cn1.var() for Cn1 in Cnorm1]
            Cnorm = [C/np.sqrt((C**2).sum()) for C in Corth]
            overallLoadingSimilarity = [1-Cn.size*Cn.var() for Cn in Cnorm]
            # note that Corth is already normalized (i.e. unit norm)
            # it's unclear why past me did the renormalization for Cnorm1 above, but whatevs. 
            # Otoh, the normalization for Cnorm is *different* in that it's attempting 
            # to normalize the entire latent matrix as if it were a vector
            eachFactorLoadingSimilarity = [1 - Cdim.size*Cdim.var() for Cn in Corth for Cdim in Cn.T ]

            # Some o' dat timescale junk
            timescalesAll = [tm for tmGrp in timescales for tm in tmGrp]

            tmsclMn = [tmsc.mean() for tmsc in timescales]
            tmsclStd = [tmsc.std() for tmsc in timescales]

            # and its variance explained bretheren...
            # Don't forget the 1- ! (because I had...)
            varExpByDimHere = [1-np.sum((indivDimPrj-origSeqsTog)**2) / np.trace(origSeqsTog @ origSeqsTog.T) for indivDimPrj in highDReprojByDimTog]
            varExpTog = [1-np.sum((highDReprojTog-origSeqsTog)**2) / np.trace(origSeqsTog @ origSeqsTog.T)]

            # note all we know is the determinant is the product of the e-vals
            # and the trace is their sum--this is a way to get at the geomean
            # and the mean without actually computing the eigenvalues
            # themselves XD
            if np.linalg.slogdet(C.T @ C)[0] != 1:
                breakpoint() # determinant suggests shrinking!
            shVarGeoMnVsMn = [(np.trace(C @ C.T)/C.shape[1]) / np.exp(1/C.shape[1]*np.linalg.slogdet(C.T @ C)[1]) if C.shape[1]>0 else np.array(np.nan)  for C, R in CR]
            participationRatio = [np.trace(C @ C.T)**2 / (np.trace(C @ C.T @ C @ C.T)) for C, R in CR] 
            participationRatioRawCov = [np.trace(C @ C.T + R)**2 / (np.trace((C @ C.T + R) @ (C @ C.T + R) )) for C, R in CR] 

            
            coefDetLowDBySubset.append(np.array(coefDetLowD))
            coefDetByLatentBySubset.append(np.array(coefDetByLatent).squeeze())
            shCovPropPopBySubset.append(np.array(shCovPropPop))
            shCovPropNeurAvgBySubset.append(np.array(shCovPropNeurAvg))
            shCovPropNeurStdBySubset.append(np.array(shCovPropNeurStd))
            shCovPropNeurNormDimBySubset.append(np.array(shCovPropNeurNormDim))
            shCovPropByLatentBySubset.append(np.array(shCovPropByLatent))
            shCovPropNeurGeoNormDimBySubset.append(np.array(shCovPropNeurGeoNormDim))
            privVarSpreadBySubset.append(np.array(privVarSpread))
            ffLoadingSimBySubset.append(np.array(firstFactorLoadingSimilarity))
            overallLoadingSimBySubset.append(np.array(overallLoadingSimilarity))
            efLoadingSimBySubset.append(np.array(eachFactorLoadingSimilarity))
            tmscAllBySubset.append(np.array(timescalesAll))
            tmscMnsBySubset.append(np.array(tmsclMn))
            tmscStdsBySubset.append(np.array(tmsclStd))
            varExpBySubset.append(np.array(varExpTog))
            varExpByDimBySubset.append(np.array(varExpByDimHere))
            shVarGeoMnVsMnBySubset.append(np.array(shVarGeoMnVsMn))
            participationRatioBySubset.append(np.array(participationRatio))
            participationRatioRawCovBySubset.append(np.array(participationRatioRawCov))
        
        coefDetLowDByExParams.append(coefDetLowDBySubset)
        coefDetByLatentByExParams.append(np.hstack(coefDetByLatentBySubset))
        shCovPropPopByExParams.append(shCovPropPopBySubset)
        shCovPropNeurAvgByExParams.append(shCovPropNeurAvgBySubset)
        shCovPropNeurStdByExParams.append(shCovPropNeurStdBySubset)
        shCovPropNeurNormDimByExParams.append(shCovPropNeurNormDimBySubset)
        shCovPropByLatentByExParams.append(np.concatenate(shCovPropByLatentBySubset, axis=0)) # can only be plotted vs by dim var exp and by dim load factor
        shCovPropNeurGeoNormDimExParams.append(shCovPropNeurGeoNormDimBySubset)
        privVarSpreadExParams.append(privVarSpreadBySubset)
        ffLoadingSimByExParams.append(np.array(ffLoadingSimBySubset))
        overallLoadingSimByExParams.append(np.array(overallLoadingSimBySubset))
        efLoadingSimByExParams.append(np.concatenate(efLoadingSimBySubset, axis=0)) # can only be plotted vs by dim var exp and tmscl
        tmscAllByExParams.append(np.concatenate(tmscAllBySubset, axis=1)) # can only be plotted vs by dim var exp and by dim load factor
        tmscMnsByExParams.append(np.array(tmscMnsBySubset))
        tmscStdsByExParams.append(np.array(tmscStdsBySubset))
        varExpByExParams.append(np.array(varExpBySubset))
        varExpByDimByExParams.append(np.concatenate(varExpByDimBySubset,axis=0)) # can only be plotted vs tmscl and by dim load factor
        shVarGeoMnVsMnByExParams.append(shVarGeoMnVsMnBySubset)
        participationRatioByExParams.append(participationRatioBySubset)
        participationRatioRawCovByExParams.append(participationRatioRawCovBySubset)


    resultsDict = {
        'R^2 for overall mean in low-d' : [np.hstack(coefDet) for coefDet in coefDetLowDByExParams],
        'R^2 for mean by latent in low-d' : [np.hstack(coefDetByL) if coefDetByL.size>0 else coefDetByL for coefDetByL in coefDetByLatentByExParams],
        '%sv mean' : [np.hstack(shPropN) for shPropN in shCovPropNeurAvgByExParams],
        '%sv std' : [np.hstack(shPropN) for shPropN in shCovPropNeurStdByExParams],
        '%sv norm dim' : [np.hstack(shPropNormDim) for shPropNormDim in shCovPropNeurNormDimByExParams],
        '%sv by latent' : [np.hstack(shCovByLatent) if shCovByLatent.size>0 else shCovByLatent for shCovByLatent in shCovPropByLatentByExParams],
#        '%sv geonorm dim' : [np.hstack(shPropGNormDim) for shPropGNormDim in shCovPropNeurGeoNormDimExParams],
        # 'priv var spread' : [np.hstack(privVar) for privVar in privVarSpreadExParams],
        'dimensionality' : [np.hstack(dm) for dm in dimensionalityExtractionParams],
        # 'sh pop cov' : [np.hstack(shProp) for shProp in shCovPropPopByExParams],
        '1st factor load sim' : [np.hstack(ffLdSim) for ffLdSim in ffLoadingSimByExParams],
        # 'all factor load sim' : [np.hstack(allLdSim) for allLdSim in overallLoadingSimByExParams],
        'each factor load sim' : [np.stack(efLdSim) if efLdSim.size>0 else efLdSim for efLdSim in efLoadingSimByExParams],
        # 'mean timescales' : [np.hstack(tmsc) for tmsc in tmscMnsByExParams],
        # 'std timescales' : [np.hstack(tmsc) for tmsc in tmscStdsByExParams],
        'mean/geomean sh var' : [np.hstack(gMvMShV) for gMvMShV in shVarGeoMnVsMnByExParams],
        # 'participation ratio' : [np.hstack(pR) for pR in participationRatioByExParams],
        # 'participation ratio raw cov' : [np.hstack(pR) for pR in participationRatioRawCovByExParams],
        # 'all timescales' : [np.hstack(tmsc) for tmsc in tmscAllByExParams],
        'var exp all' : [np.hstack(vrExp) for vrExp in varExpByExParams],
        # 'var exp by dim' : [np.hstack(vrExpBD) for vrExpBD in varExpByDimByExParams],
    }

    return resultsDict

def computeProjectionMetrics(binnedSpikeList, binnedSpikeListBaseline, gpfaResultsByExtractionParams, logLikelihoodDimensionality, gpfaTestInds, gpfaParams):
    # Note that gpfaParams are passed here so that various firing rates can be
    # accurately used to remove channels that are actually use to find the
    # latents

    projFirstLatentByExParams = []
    snrFirstLatentBySubsetByExParams = []
    varExpLatentOnlyByExParams = []
    projSignalFirstLatentByExParams = []
    projAllSignalAllLatentByExParams = []
    projFirstLatentOnConditionDiffsByExParams = []
    sigToNoiseDiffByExParams = []
    projFirstLatent95pctPCByExParams = []
    projFirstLatentIncSignalByExParams = []
    varAccountedForLastSignalPCByExParams = []
    ratioVarMnAccForVarShLatAccForByExParams = []
    onesDirProjFirstLatentByExParams = []
    onesDirProjEachLatentByExParams = []
    projOnSignalZscByExParams = []
    meanOnSignalByExParams = []
    latentsOnSignalZscByExParams = []
    # this first for loop goes through a list extracted by a specific parameter
    # set (so if there are 5 sessions and 2 parameters sets, it would be a 10
    # item list)
    for i in range(0):# bSpParams, bSpParamsBaseline, gpfaResultForParams, gpfaDimUseParams, gpfaTestIndsParams in zip(binnedSpikeList, binnedSpikeListBaseline, gpfaResultsByExtractionParams, logLikelihoodDimensionality, gpfaTestInds):
        # each of these items might have subsets that were extracted with a
        # given parameter set, and we loop through those subsets here
        # for bnSpSubset, bnSpBaselineSubset, gpfaResultSubset, dimsUseSubset, gpfaTestIndsSubset in zip(bSpParams, bSpParamsBaseline, gpfaResultForParams, gpfaDimUseParams, gpfaTestIndsParams):
        for bnSpSubset, bnSpBaselineSubset, gpfaResultSubset, dimsUseSubset, gpfaTestIndsSubset in zip(binnedSpikeList, binnedSpikeListBaseline, gpfaResultsByExtractionParams, logLikelihoodDimensionality, gpfaTestInds):
            # because GPFA gets rid of some channels based on thresholds before
            # running, we have to do that here as well to make sure the number
            # of channels match the number of channels in the latents
            bnSpSubset, highFiringChannels = bnSpSubset.channelsAboveThresholdFiringRate(firingRateThresh=gpfaParams['overallFiringRateThresh'])
            bnSpBaselineSubset = bnSpBaselineSubset[:,highFiringChannels]
            unLabsBaseline, unLabIndBaseline = np.unique(bnSpBaselineSubset.labels['stimulusMainLabel'], axis=0, return_inverse = True)

            condAvgs = []
            dirsEigs = []
            latents = []
            eigsSharedOrth = []
            meanDiffsNorm = []
            meanDiffsNormFactor = []
            stdActMnDiffs = []
            meanDiffsProjNormSigSp = []
            meanDiffsProjNormFactor = []
            trlProjsOnSignal = []
            mnProjsOnSignal = []
            latentsProjSignal = []
            # each of the subsets might have multiple GPFA runs for each of
            # their conditions... the bnSpSubset is not cut by condition, but
            # the gpfaTestInds will tell us how to do that in this loop
            breakpoint()
            for gpfaResultCond, dimsUseCond, gpfaTestInd in zip(gpfaResultSubset, dimsUseSubset, gpfaTestIndsSubset):
                trlCondBnSp = bnSpSubset[gpfaTestInd.squeeze()]
                if gpfaParams['perConditionGroupFiringRateThresh']>0:
                    trlCondBnSp.channelsAboveThresholdFiringRate(firingRateThresh=gpfaParams['perConditionGroupFiringRateThresh'])

                unLabs, unLabInd = np.unique(trlCondBnSp.labels['stimulusMainLabel'], axis=0, return_inverse = True)
                if not np.all(unLabs == unLabsBaseline):
                    breakpoint() # below assumes the inverse indexes are for identical labels

                byCondAvg = [(trlCondBnSp[unLabInd == currInd, :] - bnSpBaselineSubset[unLabIndBaseline==currInd, :].trialAverage()).trialAverage() for currInd in np.unique(unLabInd)]
                byCondAvg = np.hstack(byCondAvg).T #squeeze()
                byCondStd = [(trlCondBnSp[unLabInd == currInd, :] - bnSpBaselineSubset[unLabIndBaseline==currInd, :].trialAverage()).trialStd() for currInd in np.unique(unLabInd)]
                byCondStd = np.hstack(byCondStd).T #squeeze()
                # byCondZsc = byCondAvg/byCondStd # we're putting the mean into the units of the z-score that found the FA latents
                byCondZsc = [(trlCondBnSp[unLabInd == currInd, :] - bnSpBaselineSubset[unLabIndBaseline==currInd, :].trialAverage()).trialAverage()/byCondStd[[currInd],:].T for currInd in np.unique(unLabInd)]
                byCondZsc = np.hstack(byCondZsc).T
                # At some point I decided that if there was no response, then the z-score would be zero... helps with covariance calcs later...
                byCondZsc[(np.isnan(byCondZsc)) & (byCondAvg == 0) & (byCondStd == 0)] = 0
                neurAvgCond = byCondZsc.mean(axis=0)
                indTrlZsc = [(trlCondBnSp[unLabInd == currInd, :]/byCondStd[[currInd],:].T - neurAvgCond[:,None]).squeeze()  for currInd in np.unique(unLabInd)]
                if byCondZsc.shape[0] != unLabs.shape[0]:
                    breakpoint()

                # mnSubAvgs = byCondAvg - byCondAvg.mean(axis=0)
                mnSubAvgs = byCondZsc - byCondZsc.mean(axis=0)
                diffsBetweenMeans = mnSubAvgs - mnSubAvgs[:, None, :]
                normDiffs = np.sqrt((diffsBetweenMeans**2).sum(axis=2))
                normDiffsBetweenMeans = diffsBetweenMeans/normDiffs[:,:,None]
                meanDiffsNorm.append(normDiffsBetweenMeans)
                meanDiffsNormFactor.append(normDiffs)

                stdActivityOnMeanDiff = np.stack([np.stack(np.asarray([np.std(np.vstack([cond1Trls, cond2Trls]) @ diffsBtMn[[cond2Num]].T) for cond2Num, cond2Trls in enumerate(indTrlZsc)])) for cond1Trls, diffsBtMn in zip(indTrlZsc, normDiffsBetweenMeans)])
                stdActMnDiffs.append(stdActivityOnMeanDiff)

                covSubAvgs = 1/mnSubAvgs.shape[0]*(mnSubAvgs.T @ mnSubAvgs)
                dirs, eigs, _ = np.linalg.svd(np.array((covSubAvgs))) # this is PCA
                dirs, eigs = (dirs[:, :mnSubAvgs.shape[0]-1], eigs[:mnSubAvgs.shape[0]-1])
                dirsEigs.append((dirs, eigs))

                diffsProj = np.transpose(dirsEigs[0][0].T @ np.transpose(diffsBetweenMeans, axes=(0,2,1)), axes=(0,2,1))
                normDiffsProj = np.sqrt((diffsProj**2).sum(axis=2))
                meanDiffsProjNormSigSp.append(diffsProj/normDiffsProj[:,:,None])
                meanDiffsProjNormFactor.append(normDiffsProj)

                onesDir = np.ones((mnSubAvgs.shape[1],1))/np.sqrt(mnSubAvgs.shape[1])
                
                
                condAvg = (trlCondBnSp-bnSpBaselineSubset.trialAverage()).trialAverage()
                condAvgs.append(condAvg)

                gpfaOptDimParams = gpfaResultCond[dimsUseCond] 
                numCVals = gpfaOptDimParams['crossvalidateNum']

                if numCVals > 1:
                    # technically we should only use one crossval... so I'm
                    # putting this here to make a decision later as to whether
                    # multiple cvals being passed means we should average
                    # results or do something else
                    breakpoint()

                C = gpfaOptDimParams['allEstParams'][0]['C']
                Co, sqrtVarExp, _ = np.linalg.svd(C) # just for future reference, up top I do SVD on the *covariance* matrix, here, I'm doing it on the latents, hence the square root
                Co = Co[:, :sqrtVarExp.size]
                egsOrth = sqrtVarExp**2
                latents.append(Co)
                eigsSharedOrth.append(egsOrth)
               
                trlProjSignal = [trlResp @ dirs[:, :2] for trlResp in indTrlZsc]
                trlProjsOnSignal.append(trlProjSignal)
                mnProjSignal = mnSubAvgs @ dirs[:, :2]
                mnProjsOnSignal.append(mnProjSignal)
                latentsProjSignal.append(np.diag(egsOrth[[0]]) @ Co[:,[0]].T @ dirs[:, :2])


            projFirstLatentBySubset = [np.abs(xMn.T @ L[:,[0]])/np.sqrt(xMn.T @ xMn) for xMn, L in zip(condAvgs, latents)] 
            varExpLatentOnlyBySubset = [np.sum(np.sqrt(xMn.T @ L @ L.T @ xMn)/np.sqrt(xMn.T @ xMn)) for xMn, L in zip(condAvgs, latents)] 
            
            if np.any(np.array(varExpLatentOnlyBySubset) > 100):
                breakpoint()

            projSignalFirstLatentBySubset = [np.abs(dsEgs[0][:,[0]].T @ L[:,[0]]) for dsEgs, L in zip(dirsEigs, latents)]
            projAllSignalAllLatentBySubset = [np.diag(np.sqrt(dsEgs[0].T @ L @ L.T @ dsEgs[0])) for dsEgs, L in zip(dirsEigs, latents)]
            # note that in the below, the indexing by projKp works to grab only
            # the lower triangular portion of the projections, and the vstack
            # serves to order them so you get the projections onto mean
            # diffences using the first condition, then the remaining using the
            # second, etc (i.e. first col of lower triangle, then second col,
            # etc)
            # projFirstLatentOnConditionDiffsBySubset = [np.vstack([np.abs(mD[projKp+1:] @ dsEgs[0].T @ L[:,[0]]*Leigs[0]/nmD[projKp+1:,None]) for mD, nmD, projKp in zip(mnDff, normMnDff, range(mnDff.shape[0]))]) for  mnDff, normMnDff, L, Leigs, dsEgs in zip(meanDiffsProjNormSigSp, meanDiffsProjNormFactor, latents, eigsSharedOrth, dirsEigs)]
            sigToNoiseDiff = [np.hstack([dDfs[stdNum+1:]/stdDfs[stdNum+1:] for stdNum, (stdDfs, dDfs) in enumerate(zip(stdDffs, distDffs))]) for stdDffs, distDffs in zip(stdActMnDiffs, meanDiffsNormFactor)]
            # sigToNoiseDiff = [np.vstack([np.abs(mDfs[stdNum+1:] @ L[:,[0]] * Leigs[0]/stdDfs[stdNum+1:, None]) for stdNum, (stdDfs, mDfs) in enumerate(zip(stdDffs, mnDffs))]) for stdDffs, mnDffs, L, Leigs in zip(stdActMnDiffs, meanDiffsNorm, latents, eigsSharedOrth)]
            # sigToNoiseDiff = [np.vstack([stdDfs[stdNum+1:, None] for stdNum, (stdDfs, mDfs) in enumerate(zip(stdDffs, mnDffs))]) for stdDffs, mnDffs, L, Leigs in zip(stdActMnDiffs, meanDiffsNorm, latents, eigsSharedOrth)]
            # np.vstack([np.abs(mD[projKp+1:] @ dirsEigs[0][0].T @ latents[0][:,[0]]*eigsSharedOrth[0][0]/nmD[projKp+1:,None]) for mD, nmD, projKp in zip(meanDiffsProjNormSigSp[0], meanDiffsProjNormFactor[0], range(meanDiffsProjNormSigSp[0].shape[0]))])
            # breakpoint()
            breakpoint()
            projFirstLatentOnConditionDiffsBySubset = [np.vstack([np.abs((mD[projKp+1:] @ L[:,[0]] * Leigs[0])/nmD[projKp+1:,None]) for mD, nmD, projKp in zip(mnDff, normMnDff, range(mnDff.shape[0]))]) for  mnDff, normMnDff, L, Leigs in zip(meanDiffsNorm, meanDiffsNormFactor, latents, eigsSharedOrth)]
            # projFirstLatentOnConditionDiffsBySubset = [np.vstack([np.log(np.abs(nmD[projKp+1:,None]/(mD[projKp+1:] @ L[:,[0]] * Leigs[0]))) for mD, nmD, projKp in zip(mnDff, normMnDff, range(mnDff.shape[0]))]) for  mnDff, normMnDff, L, Leigs in zip(meanDiffsNorm, meanDiffsNormFactor, latents, eigsSharedOrth)]
            # projFirstLatentOnConditionDiffsBySubset = [np.vstack([dDfs[stdNum+1:, None] for stdNum, (stdDfs, dDfs) in enumerate(zip(stdDffs, distDffs))]) for stdDffs, distDffs, L, Leigs in zip(stdActMnDiffs, meanDiffsNormFactor, latents, eigsSharedOrth)]
            # projFirstLatentOnConditionDiffsBySubset = [np.vstack([np.abs(mD[projKp+1:] @ L[:,[0]] ) for mD, nmD, projKp in zip(mnDff, normMnDff, range(mnDff.shape[0]))]) for  mnDff, normMnDff, L, Leigs in zip(meanDiffsNorm, meanDiffsNormFactor, latents, eigsSharedOrth)]
            # breakpoint()
            # projFirstLatentOnConditionDiffsBySubset = [np.vstack([np.abs(mD[projKp+1:] @ L[:,[0]]) for mD, nmD, projKp in zip(mnDff, normMnDff, range(mnDff.shape[0]))]) for  mnDff, normMnDff, L, Leigs in zip(meanDiffsNorm, meanDiffsNormFactor, latents, eigsSharedOrth)][0].squeeze()
            projFirstLatent95pctPCBySubset = [np.diag(np.sqrt(L[:,[0]].T @ dsEgs[0][:,:(np.where(dsEgs[1].cumsum()/dsEgs[1].sum()>0.95)[0][0]+1)] @ dsEgs[0][:,:(np.where(dsEgs[1].cumsum()/dsEgs[1].sum()>0.95)[0][0]+1)].T @ L[:,[0]] )) for dsEgs, L in zip(dirsEigs, latents)]
            projFirstLatentIncSignalBySubset = [np.diag(np.sqrt(L[:,[0]].T @ dsEgs[0][:,:(sigUse+1)] @ dsEgs[0][:,:(sigUse+1)].T @ L[:,[0]] )) for dsEgs, L in zip(dirsEigs, latents) for sigUse in range(dsEgs[0].shape[1])]

            varAccountedForLastSignalPCBySubset = [dsEgs[1][-1]/dsEgs[1].sum() for dsEgs in dirsEigs]
            ratioVarMnAccForVarShLatAccForBySubset = [dsEgs[1].sum() / egsSh[0] for dsEgs, egsSh in zip(dirsEigs, eigsSharedOrth)]

            onesDirProjFirstLatentBySubset = [np.abs(onesDir.T @ L[:,0]) for L in latents]
            onesDirProjEachLatentBySubset = [np.abs(onesDir.T @ L) for L in latents]

        projFirstLatentByExParams.append(np.array(projFirstLatentBySubset))
        snrFirstLatentBySubsetByExParams.append(np.array(snrFirstLatentBySubset))
        varExpLatentOnlyByExParams.append(np.array(varExpLatentOnlyBySubset))
        projSignalFirstLatentByExParams.append(np.array(projSignalFirstLatentBySubset))
        projAllSignalAllLatentByExParams.append(np.array(projAllSignalAllLatentBySubset))
        projFirstLatentOnConditionDiffsByExParams.append(np.array(projFirstLatentOnConditionDiffsBySubset))
        sigToNoiseDiffByExParams.append(np.array(sigToNoiseDiff))
        projFirstLatent95pctPCByExParams.append(np.array(projFirstLatent95pctPCBySubset))
        projFirstLatentIncSignalByExParams.append(np.array(projFirstLatentIncSignalBySubset))
        varAccountedForLastSignalPCByExParams.append(np.array(varAccountedForLastSignalPCBySubset))
        ratioVarMnAccForVarShLatAccForByExParams.append(np.array(ratioVarMnAccForVarShLatAccForBySubset))
        onesDirProjFirstLatentByExParams.append(np.array(onesDirProjFirstLatentBySubset))
        onesDirProjEachLatentByExParams.append(np.array(onesDirProjEachLatentBySubset))

        projOnSignalZscByExParams.append(trlProjsOnSignal)
        meanOnSignalByExParams.append(mnProjsOnSignal)
        latentsOnSignalZscByExParams.append(latentsProjSignal)


    projDict = {
        'mean proj on 1st latent' : [np.hstack(proj) for proj in projFirstLatentByExParams],
        'snr 1st latent' : [np.hstack(proj) for proj in snrFirstLatentBySubsetByExParams],
        'mean proj on all latents/mean size' : [np.hstack(var) for var in varExpLatentOnlyByExParams],
        # 'max signal pca on 1st latent' : [np.hstack(proj) for proj in projSignalFirstLatentByExParams],
        # 'all signal pca on all latent' : [np.hstack(proj) for proj in projAllSignalAllLatentByExParams],
        # '(proj noise from 1st latent)/(cond diff mag)' : [np.hstack(proj) for proj in projFirstLatentOnConditionDiffsByExParams],
        # 'cond mean diff to std noise' : [np.hstack(proj) for proj in sigToNoiseDiffByExParams],
        'first latent on 95% PCA' : [np.hstack(proj) for proj in projFirstLatent95pctPCByExParams],
        # 'first latent on inc signal pca' : [np.hstack(proj) for proj in projFirstLatentIncSignalByExParams],
        'last signal pc var acc for' : [np.hstack(varAcc) for varAcc in varAccountedForLastSignalPCByExParams],
        'ratio mean var to first lat var' : [np.hstack(ratMnSh) for ratMnSh in ratioVarMnAccForVarShLatAccForByExParams], 
        'ones dir on 1st latent' : [np.hstack(proj) for proj in onesDirProjFirstLatentByExParams],
        'ones dir on each latent' : [np.hstack(proj) for proj in onesDirProjEachLatentByExParams],
    }

    projectedPoints = {
        'data into top two signal PCs' : projOnSignalZscByExParams,
        'mean into top two signal PCs' : meanOnSignalByExParams,
        'noise latents into signal PCs' : latentsOnSignalZscByExParams,
    }

    return projDict, projectedPoints

# note that gpfaResultsDictOfDicts is lists of spike sets of a specific set of
# GPFA results grouped by some key which has results from various
# dimensionality tests within (those results are in a dict keyed by
# dimensionality)
def crunchGpfaResults(gpfaResultsDictOfDicts, cvApproach = "logLikelihood", shCovThresh = 0.95):

    
    groupedResults = {}
    for relPathAndCond, gpfaResultsDict in gpfaResultsDictOfDicts.items():
        condition = gpfaResultsDict.pop('condition')

        # we sort things by extraction dimensionality
        dimAll = np.array(list(gpfaResultsDict.keys()))
        dimSort = list(np.sort(dimAll))
        dimResultsHere = {}
        gpfaScore = np.empty((0,0))
        for idxDim, dim in enumerate(dimSort):
            gpfaResult = gpfaResultsDict[dim]
            dimResultsHere[dim] = gpfaResult['dimOutput'][()]


            # NOTE 'normalGpfaScore' field was changed to 'score'
            if gpfaScore.size == 0:
                gpfaScoreInit = gpfaResult['score' if 'score' in gpfaResult else 'normalGpfaScore']
                gpfaScore = np.empty((len(dimSort),gpfaScoreInit.shape[0]))
                gpfaScore[idxDim,:] = gpfaScoreInit
            else:
                gpfaScore[idxDim,:] = gpfaResult['score' if 'score' in gpfaResult else 'normalGpfaScore']

        gpfaScoreSum = gpfaScore.sum(axis=1)
        # NOTE: come back to this: rerun GPFA on this dataset and see if something weird happens again; unfortunately GPFA is stochastic, so it might not... which is what's worrisome about this particular situation...
        # Btw, for future me: what I mean by weird is that for some reason it initially computed that dimensionality 12 was the maximum log likelihood dimensionality, and then it computed that it was actually 8. Not really sure why, as the same numbers should have been loaded up both times...
#        if relPathAndCond == ('memoryGuidedSaccade/Pepe/2018/07/14/ArrayNoSort2_PFC/dataset_449d9/binnedSpikeSet_096e2/filteredSpikes_01062_4d9a9/filteredSpikeSet.dill', '4', '[3]'):
#            breakpoint()
        if cvApproach is "logLikelihood":
            xDimScoreBest = dimSort[np.argmax(gpfaScoreSum)]
        elif cvApproach is "squaredError":
            xDimScoreBest = dimSort[np.argmin(gpfaScoreSum)]


        
        Cparams = [prm['C'] for prm in dimResultsHere[xDimScoreBest]['allEstParams']]
        if len(Cparams) == 0:
            continue

        shEigs = [np.flip(np.sort(np.linalg.eig(C.T @ C)[0])) for C in Cparams]
        percAcc = np.stack([np.cumsum(eVals)/np.sum(eVals) for eVals in shEigs])
        
        if xDimScoreBest > 0:
            meanPercAcc = np.mean(percAcc, axis=0)
            stdPercAcc = np.std(percAcc, axis = 0)
            xDimBest = np.where(meanPercAcc>shCovThresh)[0][0]+1
        else:
            xDimBest = 0

        # these should be the same for each dimensionality, so only append once
        # per condition
        testInds = gpfaResult['testInds']
        trainInds = gpfaResult['trainInds']
        alignmentBins = gpfaResult['alignmentBins']
        condLabel = gpfaResult['condLabel']
        binSize = gpfaResult['binSize']

        # this key is enough to put all dimensions with the same condition
        # together
        gpfaParamsKey = relPathAndCond
        if gpfaParamsKey not in groupedResults:
            groupedResults[gpfaParamsKey] = dict(
                    dimResults = [dimResultsHere],
                    xDimBestAll = [xDimBest],
                    xDimScoreBestAll = [xDimScoreBest],
                    normalGpfaScoreAll = [gpfaScore],
                    testInds = [testInds],
                    trainInds = [trainInds],
                    alignmentBins = [alignmentBins],
                    condLabel = [condLabel],
                    binSize = [binSize],
                    dimsTest = [dimSort],
                )
        else:
#            groupedResults[gpfaParamsKey]['dimResults'] = np.append(groupedResults[gpfaParamsKey]['dimResults'],dimResultsHere)
            groupedResults[gpfaParamsKey]['dimResults'].append(dimResultsHere)
            groupedResults[gpfaParamsKey]['xDimBestAll'].append(xDimBest)
            groupedResults[gpfaParamsKey]['xDimScoreBestAll'].append(xDimScoreBest)
            groupedResults[gpfaParamsKey]['normalGpfaScoreAll'].append(gpfaScore)
            groupedResults[gpfaParamsKey]['testInds'].append(testInds)
            groupedResults[gpfaParamsKey]['trainInds'].append(trainInds)
            groupedResults[gpfaParamsKey]['alignmentBins'].append(alignmentBins)
            groupedResults[gpfaParamsKey]['condLabel'].append(condLabel)
            groupedResults[gpfaParamsKey]['binSize'].append(binSize)
            groupedResults[gpfaParamsKey]['dimsTest'].append(dimSort)
        
    return groupedResults

def computeBestDimensionality(gpfaResultsDictOfDicts, cvApproach = "logLikelihood", shCovThresh = 0.95):

    outBestDims = []

    for relPathAndCond, gpfaResultsDict in gpfaResultsDictOfDicts.items():

        # we sort things by extraction dimensionality
        allKeys = list(gpfaResultsDict.keys())
        dimAll = np.array([aK for aK in allKeys if isinstance(aK, np.integer)])
        dimSort = list(np.sort(dimAll))
        dimResultsHere = {}
        gpfaScore = np.empty((0,0))
        for idxDim, dim in enumerate(dimSort):
            gpfaResult = gpfaResultsDict[dim]

            # NOTE 'normalGpfaScore' field was changed to 'score'--but I've
            # gotten rid of the reference to normalGpfaScore because I don't
            # think (I hope) it won't matter anymore because these GPFA runs
            # are new enough...
            if gpfaScore.size == 0:
                gpfaScoreInit = gpfaResult['score']# if 'score' in gpfaResult]
                gpfaScore = np.empty((len(dimSort),gpfaScoreInit.shape[0]))
                gpfaScore[idxDim,:] = gpfaScoreInit
            else:
                gpfaScore[idxDim,:] = gpfaResult['score']# if 'score' in gpfaResult]

        gpfaScoreSum = gpfaScore.sum(axis=1)

        if cvApproach is "logLikelihood":
            xDimScoreBest = dimSort[np.argmax(gpfaScoreSum)]
        elif cvApproach is "squaredError":
            xDimScoreBest = dimSort[np.argmin(gpfaScoreSum)]

        outBestDims.append(xDimScoreBest)

    return outBestDims
# note that faResultsDictOfDicts is lists of spike sets of a specific set of
# FA results grouped by some key which has results from various
# dimensionality tests within (those results are in a dict keyed by
# dimensionality)
def crunchFaResults(faResultsDictOfDicts, cvApproach = "logLikelihood", shCovThresh = 0.95):

    
    groupedResults = {}
    for relPathAndCond, gpfaResultsDict in gpfaResultsDictOfDicts.items():
        condition = gpfaResultsDict.pop('condition')

        # we sort things by extraction dimensionality
        dimAll = np.array(list(gpfaResultsDict.keys()))
        dimSort = list(np.sort(dimAll))
        dimResultsHere = {}
        gpfaScore = np.empty((0,0))
        for idxDim, dim in enumerate(dimSort):
            gpfaResult = gpfaResultsDict[dim]
            dimResultsHere[dim] = gpfaResult['dimOutput'][()]


            if gpfaScore.size == 0:
                gpfaScoreInit = gpfaResult['normalGpfaScore']
                gpfaScore = np.empty((len(dimSort),gpfaScoreInit.shape[0]))
                gpfaScore[idxDim,:] = gpfaScoreInit
            else:
                gpfaScore[idxDim,:] = gpfaResult['normalGpfaScore']

        gpfaScoreSum = gpfaScore.sum(axis=1)
        if cvApproach is "logLikelihood":
            xDimScoreBest = dimSort[np.argmax(gpfaScoreSum)]
        elif cvApproach is "squaredError":
            xDimScoreBest = dimSort[np.argmin(gpfaScoreSum)]


        
        Cparams = [prm['C'] for prm in dimResultsHere[xDimScoreBest]['allEstParams']]
        shEigs = [np.flip(np.sort(np.linalg.eig(C.T @ C)[0])) for C in Cparams]
        percAcc = np.stack([np.cumsum(eVals)/np.sum(eVals) for eVals in shEigs])
        
        if xDimScoreBest>0:
            meanPercAcc = np.mean(percAcc, axis=0)
            stdPercAcc = np.std(percAcc, axis = 0)
            xDimBest = np.where(meanPercAcc>shCovThresh)[0][0]+1
        else:
            xDimBest = 0

        # these should be the same for each dimensionality, so only append once
        # per condition
        testInds = gpfaResult['testInds']
        trainInds = gpfaResult['trainInds']
        alignmentBins = gpfaResult['alignmentBins']
        condLabel = gpfaResult['condLabel']
        binSize = gpfaResult['binSize']

        # this key is enough to put all dimensions with the same condition
        # together
        gpfaParamsKey = relPathAndCond
        if gpfaParamsKey not in groupedResults:
            groupedResults[gpfaParamsKey] = dict(
                    dimResults = [dimResultsHere],
                    xDimBestAll = [xDimBest],
                    xDimScoreBestAll = [xDimScoreBest],
                    normalGpfaScoreAll = [gpfaScore],
                    testInds = [testInds],
                    trainInds = [trainInds],
                    alignmentBins = [alignmentBins],
                    condLabel = [condLabel],
                    binSize = [binSize],
                    dimsTest = [dimSort],
                )
        else:
#            groupedResults[gpfaParamsKey]['dimResults'] = np.append(groupedResults[gpfaParamsKey]['dimResults'],dimResultsHere)
            groupedResults[gpfaParamsKey]['dimResults'].append(dimResultsHere)
            groupedResults[gpfaParamsKey]['xDimBestAll'].append(xDimBest)
            groupedResults[gpfaParamsKey]['xDimScoreBestAll'].append(xDimScoreBest)
            groupedResults[gpfaParamsKey]['normalGpfaScoreAll'].append(gpfaScore)
            groupedResults[gpfaParamsKey]['testInds'].append(testInds)
            groupedResults[gpfaParamsKey]['trainInds'].append(trainInds)
            groupedResults[gpfaParamsKey]['alignmentBins'].append(alignmentBins)
            groupedResults[gpfaParamsKey]['condLabel'].append(condLabel)
            groupedResults[gpfaParamsKey]['binSize'].append(binSize)
            groupedResults[gpfaParamsKey]['dimsTest'].append(dimSort)
        
    return groupedResults
