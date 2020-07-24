"""
We're going to be putting some GPFA related methods here that are separate from
what the class can do
"""

from pathlib import Path
import numpy as np

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
        gpfaParamsKey = str(relPathAndCond)
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
        gpfaParamsKey = str(relPathAndCond)
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
