"""
We're going to be putting some GPFA related methods here that are separate from
what the class can do
"""

from pathlib import Path
import numpy as np

def computePopulationMetrics(gpfaResultsByExtractionParams, logLikelihoodDimensionality, dimensionalityExtractionParams, binSizeMs):

    # prep the outputs we're getting
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

                highDReprojAll = [C@xsm for xsm in lowDSeqsNonorth]
                
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
            shCovPropNeurGeoNormDim = [np.mean(np.diag(C @ C.T) / (np.diag(C @ C.T) + np.diag(R)))**(1/C.shape[1]) for C, R in CR] 
            privVarSpread = [np.diag(R).max()-np.diag(R).min() for C, R in CR]

            
            # NOTE: I'm doing all of this on the orthonormalized C! I believe
            # this is correct as it reflects the dimensions matched to the
            # eigenvalues--e.g. the first eigenvalue describes how much
            # variance is explained by the first orthonormalized dimension
            #
            # Also remember that Python is zero-indexed for grabbing that first
            # one >.>
            Cnorm1 = [C[:,0]/np.sqrt((C[:,0]**2).sum()) for C in Corth]
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
            shVarGeoMnVsMn = [(np.trace(C @ C.T)/C.shape[1]) / np.exp(1/C.shape[1]*np.linalg.slogdet(C.T @ C)[1])   for C, R in CR]
            participationRatio = [np.trace(C @ C.T)**2 / (np.trace(C @ C.T @ C @ C.T)) for C, R in CR] 
            participationRatioRawCov = [np.trace(C @ C.T + R)**2 / (np.trace((C @ C.T + R) @ (C @ C.T + R) )) for C, R in CR] 

            
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
        '%sv mean' : [np.hstack(shPropN) for shPropN in shCovPropNeurAvgByExParams],
        '%sv std' : [np.hstack(shPropN) for shPropN in shCovPropNeurStdByExParams],
        '%sv norm dim' : [np.hstack(shPropNormDim) for shPropNormDim in shCovPropNeurNormDimByExParams],
        '%sv by latent' : [np.hstack(shCovByLatent) for shCovByLatent in shCovPropByLatentByExParams],
#        '%sv geonorm dim' : [np.hstack(shPropGNormDim) for shPropGNormDim in shCovPropNeurGeoNormDimExParams],
        # 'priv var spread' : [np.hstack(privVar) for privVar in privVarSpreadExParams],
        'dimensionality' : [np.hstack(dm) for dm in dimensionalityExtractionParams],
        # 'sh pop cov' : [np.hstack(shProp) for shProp in shCovPropPopByExParams],
        '1st factor load sim' : [np.hstack(ffLdSim) for ffLdSim in ffLoadingSimByExParams],
        # 'all factor load sim' : [np.hstack(allLdSim) for allLdSim in overallLoadingSimByExParams],
        'each factor load sim' : [np.stack(efLdSim) for efLdSim in efLoadingSimByExParams],
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
        
        meanPercAcc = np.mean(percAcc, axis=0)
        stdPercAcc = np.std(percAcc, axis = 0)
        xDimBest = np.where(meanPercAcc>shCovThresh)[0][0]+1

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
        
        meanPercAcc = np.mean(percAcc, axis=0)
        stdPercAcc = np.std(percAcc, axis = 0)
        xDimBest = np.where(meanPercAcc>shCovThresh)[0][0]+1

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
