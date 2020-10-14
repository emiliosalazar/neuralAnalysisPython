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
    shCovPropNeurGeoNormDimExParams = []
    ffLoadingSimByExParams = []
    overallLoadingSimByExParams = []
    tmscMnsByExParams = []
    tmscStdsByExParams = []
    shVarGeoMnVsMnByExParams = []
    participationRatioByExParams = []
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
        shCovPropNeurGeoNormDimBySubset = []
        ffLoadingSimBySubset = []
        overallLoadingSimBySubset = []
        tmscMnsBySubset = []
        tmscStdsBySubset = []
        shVarGeoMnVsMnBySubset = []
        participationRatioBySubset = []
        for (gpSubset, dGU) in zip(gpfaResult, dimsGpfaUse):
            # extract C and R parameters
            CR = []
            Corth = []
            timescales = []
            for gpfaCond, llDim in zip(gpSubset, dGU):
                C = gpfaCond[int(llDim)]['allEstParams'][0]['C']
                if C.shape[1] != llDim:
                    # I at some point was indexing this, but I no longer
                    # think that's necessary because I'm now using the log
                    # likelihood dimensionality, not the Williamson
                    # dimensionality
                    breakpoint()
                    raise Exception("AAAH")
                R = gpfaCond[int(llDim)]['allEstParams'][0]['R']
                Co = gpfaCond[int(llDim)]['allEstParams'][0]['Corth']
                timescale = gpfaCond[int(llDim)]['allEstParams'][0]['gamma']
                CR.append((C,R))
                Corth.append(Co)
                timescales.append(binSizeMs/np.sqrt(timescale))

            shCovPropPop = [np.trace(C @ C.T) / (np.trace(C @ C.T) + np.trace(R)) for C, R in CR] 
            shCovPropNeurAvg = [np.mean(np.diag(C @ C.T) / (np.diag(C @ C.T) + np.diag(R))) for C, R in CR] 
            shCovPropNeurStd = [np.std(np.diag(C @ C.T) / (np.diag(C @ C.T) + np.diag(R))) for C, R in CR] 
            shCovPropNeurNormDim = [np.mean(np.diag(C @ C.T) / (np.diag(C @ C.T) + np.diag(R)))/C.shape[1] for C, R in CR] 
            shCovPropNeurGeoNormDim = [np.mean(np.diag(C @ C.T) / (np.diag(C @ C.T) + np.diag(R)))**(1/C.shape[1]) for C, R in CR] 

            
            # NOTE: I'm doing all of this on the orthonormalized C! I believe
            # this is correct as it reflects the dimensions matched to the
            # eigenvalues--e.g. the first eigenvalue describes how much
            # variance is explained by the first orthonormalized dimension
            Cnorm1 = [C[:,1]/np.sqrt((C[:,1]**2).sum()) for C in Corth]
            firstFactorLoadingSimilarity = [1-Cn1.size*Cn1.var() for Cn1 in Cnorm1]
            Cnorm = [C/np.sqrt((C**2).sum()) for C in Corth]
            overallLoadingSimilarity = [1-Cn.size*Cn.var() for Cn in Cnorm]
            tmsclMn = [tmsc.mean() for tmsc in timescales]
            tmsclStd = [tmsc.std() for tmsc in timescales]
#            breakpoint()

            # note all we know is the determinant is the product of the e-vals
            # and the trace is their sum--this is a way to get at the geomean
            # and the mean without actually computing the eigenvalues
            # themselves XD
            if np.linalg.slogdet(C.T @ C)[0] != 1:
                breakpoint() # determinant suggests shrinking!
            shVarGeoMnVsMn = [(np.trace(C @ C.T)/C.shape[1]) / np.exp(1/C.shape[1]*np.linalg.slogdet(C.T @ C)[1])   for C, R in CR]
            participationRatio = [np.trace(C @ C.T)**2 / (np.trace(C @ C.T @ C @ C.T)) for C, R in CR] 

            
            shCovPropPopBySubset.append(np.array(shCovPropPop))
            shCovPropNeurAvgBySubset.append(np.array(shCovPropNeurAvg))
            shCovPropNeurStdBySubset.append(np.array(shCovPropNeurStd))
            shCovPropNeurNormDimBySubset.append(np.array(shCovPropNeurNormDim))
            shCovPropNeurGeoNormDimBySubset.append(np.array(shCovPropNeurGeoNormDim))
            ffLoadingSimBySubset.append(np.array(firstFactorLoadingSimilarity))
            overallLoadingSimBySubset.append(np.array(overallLoadingSimilarity))
            tmscMnsBySubset.append(np.array(tmsclMn))
            tmscStdsBySubset.append(np.array(tmsclStd))
            shVarGeoMnVsMnBySubset.append(np.array(shVarGeoMnVsMn))
            participationRatioBySubset.append(np.array(participationRatio))
        
        shCovPropPopByExParams.append(shCovPropPopBySubset)
        shCovPropNeurAvgByExParams.append(shCovPropNeurAvgBySubset)
        shCovPropNeurStdByExParams.append(shCovPropNeurStdBySubset)
        shCovPropNeurNormDimByExParams.append(shCovPropNeurNormDimBySubset)
        shCovPropNeurGeoNormDimExParams.append(shCovPropNeurGeoNormDimBySubset)
        ffLoadingSimByExParams.append(np.array(ffLoadingSimBySubset))
        overallLoadingSimByExParams.append(np.array(overallLoadingSimBySubset))
        tmscMnsByExParams.append(np.array(tmscMnsBySubset))
        tmscStdsByExParams.append(np.array(tmscStdsBySubset))
        shVarGeoMnVsMnByExParams.append(shVarGeoMnVsMnBySubset)
        participationRatioByExParams.append(participationRatioBySubset)
#        else:
#            CR = [(gpfaCond[int(gpfaDimParamsUse)]['allEstParams'][0]['C'][:,:gpfaDimParamsUse],gpfaCond[int(gpfaDimParamsUse)]['allEstParams'][0]['R']) for gpfaCond, gpfaDimParamsUse in zip(gpfaResult, dimsGpfaUse)]
#            shCovPropPop = [np.trace(C @ C.T) / (np.trace(C @ C.T) + np.trace(R)) for C, R in CR] 
#            shCovPropNeurAvg = [np.mean(np.diag(C @ C.T) / (np.diag(C @ C.T) + np.diag(R))) for C, R in CR] 
#            shLogDetGenCovPropPop = [np.linalg.slogdet(C.T @ C)[1] / (np.linalg.slogdet(C.T @ C)[1] + np.linalg.slogdet(R)[1]) for C, R in CR]
#            
#            shCovPropPopByExParams.append(np.array(shCovPropPop))
#            shCovPropNeurAvgByExParams.append(np.array(shCovPropNeurAvg))
#            shLogDetGenCovPropPopByExParams.append(np.array(shLogDetGenCovPropPop))

#    breakpoint()
    resultsDict = {
        '%sv mean' : [np.hstack(shPropN) for shPropN in shCovPropNeurAvgByExParams],
        '%sv std' : [np.hstack(shPropN) for shPropN in shCovPropNeurStdByExParams],
        '%sv norm dim' : [np.hstack(shPropNormDim) for shPropNormDim in shCovPropNeurNormDimByExParams],
#        '%sv geonorm dim' : [np.hstack(shPropGNormDim) for shPropGNormDim in shCovPropNeurGeoNormDimExParams],
        'dimensionality' : [np.hstack(dm) for dm in dimensionalityExtractionParams],
        'sh pop cov' : [np.hstack(shProp) for shProp in shCovPropPopByExParams],
        '1st factor load sim' : [np.hstack(ffLdSim) for ffLdSim in ffLoadingSimByExParams],
        'all factor load sim' : [np.hstack(allLdSim) for allLdSim in overallLoadingSimByExParams],
#        'mean timescales' : [np.hstack(tmsc) for tmsc in tmscMnsByExParams],
#        'std timescales' : [np.hstack(tmsc) for tmsc in tmscStdsByExParams],
        'mean/geomean sh var' : [np.hstack(gMvMShV) for gMvMShV in shVarGeoMnVsMnByExParams],
        'participation ratio' : [np.hstack(pR) for pR in participationRatioByExParams],
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
        normalGpfaScore = np.empty((0,0))
        for idxDim, dim in enumerate(dimSort):
            gpfaResult = gpfaResultsDict[dim]
            dimResultsHere[dim] = gpfaResult['dimOutput'][()]


            # NOTE 'normalGpfaScore' field was changed to 'score'
            if normalGpfaScore.size == 0:
                normalGpfaScoreInit = gpfaResult['score' if 'score' in gpfaResult else 'normalGpfaScore']
                normalGpfaScore = np.empty((len(dimSort),normalGpfaScoreInit.shape[0]))
                normalGpfaScore[idxDim,:] = normalGpfaScoreInit
            else:
                normalGpfaScore[idxDim,:] = gpfaResult['score' if 'score' in gpfaResult else 'normalGpfaScore']

        normalGpfaScoreMn = normalGpfaScore.mean(axis=1)
        # NOTE: come back to this: rerun GPFA on this dataset and see if something weird happens again; unfortunately GPFA is stochastic, so it might not... which is what's worrisome about this particular situation...
        # Btw, for future me: what I mean by weird is that for some reason it initially computed that dimensionality 12 was the maximum log likelihood dimensionality, and then it computed that it was actually 8. Not really sure why, as the same numbers should have been loaded up both times...
#        if relPathAndCond == ('memoryGuidedSaccade/Pepe/2018/07/14/ArrayNoSort2_PFC/dataset_449d9/binnedSpikeSet_096e2/filteredSpikes_01062_4d9a9/filteredSpikeSet.dill', '4', '[3]'):
#            breakpoint()
        if cvApproach is "logLikelihood":
            xDimScoreBest = dimSort[np.argmax(normalGpfaScoreMn)]
        elif cvApproach is "squaredError":
            xDimScoreBest = dimSort[np.argmin(normalGpfaScoreMn)]


        
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
                    normalGpfaScoreAll = [normalGpfaScore],
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
            groupedResults[gpfaParamsKey]['normalGpfaScoreAll'].append(normalGpfaScore)
            groupedResults[gpfaParamsKey]['testInds'].append(testInds)
            groupedResults[gpfaParamsKey]['trainInds'].append(trainInds)
            groupedResults[gpfaParamsKey]['alignmentBins'].append(alignmentBins)
            groupedResults[gpfaParamsKey]['condLabel'].append(condLabel)
            groupedResults[gpfaParamsKey]['binSize'].append(binSize)
            groupedResults[gpfaParamsKey]['dimsTest'].append(dimSort)
        
    return groupedResults

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
        normalGpfaScore = np.empty((0,0))
        for idxDim, dim in enumerate(dimSort):
            gpfaResult = gpfaResultsDict[dim]
            dimResultsHere[dim] = gpfaResult['dimOutput'][()]


            if normalGpfaScore.size == 0:
                normalGpfaScoreInit = gpfaResult['normalGpfaScore']
                normalGpfaScore = np.empty((len(dimSort),normalGpfaScoreInit.shape[0]))
                normalGpfaScore[idxDim,:] = normalGpfaScoreInit
            else:
                normalGpfaScore[idxDim,:] = gpfaResult['normalGpfaScore']

        normalGpfaScoreMn = normalGpfaScore.mean(axis=1)
        if cvApproach is "logLikelihood":
            xDimScoreBest = dimSort[np.argmax(normalGpfaScoreMn)]
        elif cvApproach is "squaredError":
            xDimScoreBest = dimSort[np.argmin(normalGpfaScoreMn)]


        
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
                    normalGpfaScoreAll = [normalGpfaScore],
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
            groupedResults[gpfaParamsKey]['normalGpfaScoreAll'].append(normalGpfaScore)
            groupedResults[gpfaParamsKey]['testInds'].append(testInds)
            groupedResults[gpfaParamsKey]['trainInds'].append(trainInds)
            groupedResults[gpfaParamsKey]['alignmentBins'].append(alignmentBins)
            groupedResults[gpfaParamsKey]['condLabel'].append(condLabel)
            groupedResults[gpfaParamsKey]['binSize'].append(binSize)
            groupedResults[gpfaParamsKey]['dimsTest'].append(dimSort)
        
    return groupedResults
