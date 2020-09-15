"""
We're going to be putting some GPFA related methods here that are separate from
what the class can do
"""

from pathlib import Path
import numpy as np

def computePopulationMetrics(gpfaResultsByExtractionParams, logLikelihoodDimensionality, dimensionalityExtractionParams):

    # prep the outputs we're getting
    shCovPropPopByExParams = []
    shCovPropNeurAvgByExParams = []
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
        shVarGeoMnVsMnBySubset = []
        participationRatioBySubset = []
        for (gpSubset, dGU) in zip(gpfaResult, dimsGpfaUse):
            # extract C and R parameters
            CR = []
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
                CR.append((C,R))

            shCovPropPop = [np.trace(C @ C.T) / (np.trace(C @ C.T) + np.trace(R)) for C, R in CR] 
            shCovPropNeurAvg = [np.mean(np.diag(C @ C.T) / (np.diag(C @ C.T) + np.diag(R))) for C, R in CR] 
            # note all we know is the determinant is the product of the e-vals
            # and the trace is their sum--this is a way to get at the geomean
            # and the mean without actually computing the eigenvalues
            # themselves XD
            shVarGeoMnVsMn = [np.exp(1/C.shape[1]*np.linalg.slogdet(C.T @ C)[1]) / (np.trace(C @ C.T)/C.shape[1])  for C, R in CR]
            participationRatio = [np.trace(C @ C.T)**2 / (np.trace(C @ C.T @ C @ C.T)) for C, R in CR] 

            
            shCovPropPopBySubset.append(np.array(shCovPropPop))
            shCovPropNeurAvgBySubset.append(np.array(shCovPropNeurAvg))
            shVarGeoMnVsMnBySubset.append(np.array(shVarGeoMnVsMn))
            participationRatioBySubset.append(np.array(participationRatio))
        
        shCovPropPopByExParams.append(shCovPropPopBySubset)
        shCovPropNeurAvgByExParams.append(shCovPropNeurAvgBySubset)
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

    resultsDict = {
        '% sh var' : [np.hstack(shPropN) for shPropN in shCovPropNeurAvgByExParams],
        'dimensionality' : [np.hstack(dm) for dm in dimensionalityExtractionParams],
        'sh pop cov' : [np.hstack(shProp) for shProp in shCovPropPopByExParams],
        'geomean/mean sh var' : [np.hstack(gMvMShV) for gMvMShV in shVarGeoMnVsMnByExParams],
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
