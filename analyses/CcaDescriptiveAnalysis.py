"""
This function is responsible for the cross-area analyses/scatter matrices etc
Like HeadlessAnalysisRun2-5.py for now...
"""
from methods.plotMethods.PlotUtils import MoveFigureToSubplot
from matplotlib import pyplot as plt
import plotly as ply
import plotly.express as px 
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['GOTO_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np
import scipy as sp
import re
import dill as pickle # because this is what people seem to do?
import hashlib
import json
from pathlib import Path
from scipy.stats import binned_statistic

from methods.GeneralMethods import saveFiguresToPdf
# database stuff
from setup.DataJointSetup import DatasetInfo, DatasetGeneralLoadParams, BinnedSpikeSetInfo, FilterSpikeSetParams
import datajoint as dj
# the decorator to save these to a database!
from decorators.AnalysisCallDecorators import saveCallsToDatabase

from classes.Dataset import Dataset
from classes.BinnedSpikeSet import BinnedSpikeSet


# for generating the binned spike sets
from methods.BinnedSpikeSetListMethods import generateBinnedSpikeListsAroundState as genBSLAroundState

# subsampling
from methods.BinnedSpikeSetListMethods import subsampleBinnedSpikeSetsToMatchNeuronsAndTrialsPerCondition as subsmpMatchCond

# for the pcca...
from classes.pCCA import pCCA

# for descriptions of the data
from methods.BinnedSpikeSetListMethods import generateLabelGroupStatistics as genBSLLabGrp
from methods.BinnedSpikeSetListMethods import plotExampleChannelResponses
from methods.BinnedSpikeSetListMethods import plotStimDistributionHistograms
from methods.BinnedSpikeSetListMethods import plotFiringRates

# for computing metrics
from methods.BinnedSpikeSetListMethods import rscComputations, decodeComputations, informationComputations
from methods.GpfaMethods import computePopulationMetrics, computeProjectionMetrics

# for plotting the metrics
from methods.plotMethods.PopMetricsPlotMethods import plotAllVsAll, plotMetricVsExtractionParams, plotMetricsBySeparation
from methods.plotMethods.UnsortedPlotMethods import plotPointProjections


# @saveCallsToDatabase
def ccaDescriptiveAnalysis(datasetSqlPairings, binnedSpikeSetGenerationParamsDict, plotParams):

    dsgl = DatasetGeneralLoadParams()
    dsi = DatasetInfo()
    bsi = BinnedSpikeSetInfo()
    fsp = FilterSpikeSetParams()

    outputFiguresRelativePath = []

    print("Computing/loading binned spike sets")
    subtractConditionMeans = binnedSpikeSetGenerationParamsDict.pop('subtractConditionMeans')
    bssiKeysOne = []
    bssiKeysTwo = []
    for sqlPairing in datasetSqlPairings:
        dsiOnePath = sqlPairing['dataset_relative_path']
        dsiTwoPath = sqlPairing['dataset_relative_path_pair']

        _, bssiKeyOne = genBSLAroundState(dsi['dataset_relative_path="{}"'.format(dsiOnePath)],
                                            **binnedSpikeSetGenerationParamsDict
                                            )
        _, bssiKeyTwo = genBSLAroundState(dsi['dataset_relative_path="{}"'.format(dsiTwoPath)],
                                            **binnedSpikeSetGenerationParamsDict
                                            )
        keyStateName = binnedSpikeSetGenerationParamsDict['keyStateName']

        brainAreaO = dsi[bssiKeyOne]['brain_area'][0]
        brainAreaT = dsi[bssiKeyTwo]['brain_area'][0]
        binnedSpikesOne = bsi[bssiKeyOne].grabBinnedSpikes()[0].convertUnitsTo('count')
        binnedSpikesTwo = bsi[bssiKeyTwo].grabBinnedSpikes()[0].convertUnitsTo('count')

        binnedSpikesOne = np.vstack([binnedSpikesOne[:717], binnedSpikesOne[718:]])
        binnedSpikesTwo = np.vstack([binnedSpikesTwo[:717], binnedSpikesTwo[718:]])

        if subtractConditionMeans:
            binnedSpikesOne = binnedSpikesOne.baselineSubtract(labels = binnedSpikesOne.labels['stimulusMainLabel'])[0]
            binnedSpikesTwo = binnedSpikesTwo.baselineSubtract(labels = binnedSpikesTwo.labels['stimulusMainLabel'])[0]

        bSOAvg = binnedSpikesOne#.timeSum()
        bSTAvg = binnedSpikesTwo#.timeSum()

        bSOAvgMnSub, _ = bSOAvg.baselineSubtract(labels=binnedSpikesOne.labels['stimulusMainLabel']) #np.array(bSOAvg - bSOAvg.trialAverage())
        bSTAvgMnSub, _ = bSTAvg.baselineSubtract(labels=binnedSpikesTwo.labels['stimulusMainLabel']) #np.array(bSTAvg - bSTAvg.trialAverage())

        bSOAvgMnSub = np.hstack(bSOAvgMnSub).T
        bSTAvgMnSub = np.hstack(bSTAvgMnSub).T

        numCvals = 4
        probCCAsetup = pCCA(bSOAvgMnSub[:, :, None], bSTAvgMnSub[:, :, None], numCvals)
        numCanonDirTest = [0,2,5,8,12]
        pccaScoresDirs = []
        pccaScoresDirsErr = []
        print('running coarse pCCA dims')
        for numDir in numCanonDirTest:
            pccaScore, allEstParams, seqsTrainNew, seqsTestNew, fullRank = probCCAsetup.runPcca(numDir = numDir)
            pccaScoresDirs.append(np.mean(pccaScore))
            pccaScoresDirsErr.append(np.std(pccaScore))

        bestDirInd = np.argmax(pccaScoresDirs)
        if bestDirInd != 0:
            bestDirFirstAround = numCanonDirTest[bestDirInd-1]+1
        else:
            bestDirFirstAround = numCanonDirTest[bestDirInd]+1

        if bestDirInd != len(numCanonDirTest)-1:
            bestDirLastAround = numCanonDirTest[bestDirInd+1]
        else:
            bestDirLastAround = numCanonDirTest[bestDirInd]+3

        newCanonDirTest = list(range(bestDirFirstAround, bestDirLastAround))
        print('running finer pCCA dims')
        for numDir in newCanonDirTest:
            pccaScore, allEstParams, seqsTrainNew, seqsTestNew, fullRank = probCCAsetup.runPcca(numDir = numDir)
            pccaScoresDirs.append(np.mean(pccaScore))
            pccaScoresDirsErr.append(np.std(pccaScore))
        allCanonDirTest = numCanonDirTest + newCanonDirTest
        canonDirSort = np.sort(allCanonDirTest)
        pccaScoresDirsSort = np.array(pccaScoresDirs)[np.argsort(allCanonDirTest)]
        pccaScoresDirsErrSort = np.array(pccaScoresDirsErr)[np.argsort(allCanonDirTest)]
        fig = px.line().add_scatter(
                x=canonDirSort, y=pccaScoresDirsSort,
                line=dict(color='rgb(0,100,80)'),
                mode='lines',
            ).add_scatter(
                x = np.hstack((canonDirSort, canonDirSort[::-1])),
                y = np.hstack((pccaScoresDirsSort+pccaScoresDirsErrSort,pccaScoresDirsSort[::-1]-pccaScoresDirsErrSort[::-1])) ,
                line=dict(color='rgba(255,255,255,0)'), # get rid of border lines
                fill = 'toself',
                fillcolor='rgba(0,100,80,0.2)',
                hoverinfo='skip',
                showlegend=False,
            )

        fig.update_layout(
            title=f'pCCA dimensionality LL between {brainAreaO} and {brainAreaT} during {keyStateName}',
            xaxis_title_text='number latents in pCCA model',
            yaxis_title_text='LL (average ± std of four crossvalidations)',
        )
        fig.show()
        # breakpoint()


        from sklearn.cross_decomposition import CCA

        compCCA = 25
        cca = CCA(n_components = compCCA)
        bSOTransf, bSTTransf = cca.fit_transform(bSOAvgMnSub, bSTAvgMnSub)

        bsOCanonDirs = cca.x_rotations_
        bsTCanonDirs = cca.y_rotations_

        bSOTransf = bSOTransf.T
        bSTTransf = bSTTransf.T


        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        nrows = 2
        ncols = 2
        nSubplotPerFig = nrows*ncols
        figAnglesToSave = []
        figAttToSave = []
        numCcaPlot = 10
        axsAttFirst = []
        for ccaPltNum in range(int(numCcaPlot/nSubplotPerFig)):
            # ** POP BACK HERE **
            figAngle, axsAngle = plt.subplots(nrows, ncols,constrained_layout=True)

            figAtt, axsAtt = plt.subplots(nrows, ncols,constrained_layout=True)
            if not len(axsAttFirst):
                axsAttFirst = axsAtt

            for sbPNum, (axAngle, axAtt) in enumerate(zip(axsAngle.flat, axsAtt.flat)):
                ccNum = ccaPltNum*nSubplotPerFig + sbPNum
                if ccNum >= bSOTransf.shape[0]:
                    break
                ccO = bSOTransf[ccNum, :]
                ccT = bSTTransf[ccNum, :]
                rho = np.corrcoef(ccO, ccT)
                unLabAng, labNumAng = np.unique(bSOAvg.labels['rfOrientation'],axis=0, return_inverse=True)
                colorsUse = plt.cm.Set3(np.arange(len(unLabAng)))
                scatterAngle = axAngle.scatter(ccO, ccT, c = labNumAng, alpha=0.2, edgecolor='none')
                if sbPNum == 0:
                    legObj = [*scatterAngle.legend_elements()][0]
                    legLab = ['angle 1', 'angle 2']
                    axAngle.legend(legObj, legLab, title='grating angle')
                axAngle.set_title("CCA {} $\\rho$={:.4f}".format(ccNum+1, rho[0,1]))
                axAngle.set_xlabel("{}".format(brainAreaO))
                axAngle.set_ylabel("{}".format(brainAreaT))
                
                unLabAtt, labNumAtt = np.unique(bSOAvg.labels['cueLocation'],axis=0, return_inverse=True)
                colorsUse = plt.cm.Set3(np.arange(len(unLabAtt)))
                scatterAttention = axAtt.scatter(ccO, ccT, c = labNumAtt, alpha=0.2, edgecolor='none')
                if sbPNum == 0:
                    legObj = [*scatterAttention.legend_elements()][0]
                    legLab = ['loc 1', 'loc 2']
                    axAtt.legend(legObj, legLab, title='cue location')
                axAtt.set_title("CCA {} $\\rho$={:.4f}".format(ccNum, rho[0,1]))
                axAtt.set_xlabel("{}".format(brainAreaO))
                axAtt.set_ylabel("{}".format(brainAreaT))
            # ** POP TO HERE **


            figAnglesToSave.append(figAngle.number)
            figAttToSave.append(figAtt.number)

        outputFiguresRelativePath.append(saveFiguresToPdf(analysisDescription=f'ccaAnalysis{brainAreaO}_{brainAreaT}',pdfname='{}{}_angle'.format(plotParams['pdfnameSt'],plotParams['analysisIdentifier']), figNumsToSave = figAnglesToSave))
        outputFiguresRelativePath.append(saveFiguresToPdf(analysisDescription=f'ccaAnalysis{brainAreaO}_{brainAreaT}',pdfname='{}{}_attention'.format(plotParams['pdfnameSt'],plotParams['analysisIdentifier']), figNumsToSave = figAttToSave))

        figPrincCorr, axCorr = plt.subplots(1,1,constrained_layout=True)
        corrs = np.corrcoef(bSOTransf, bSTTransf).diagonal(compCCA).copy()
        corrs = np.sort(corrs)[::-1]
        axCorr.plot(corrs, label='original correlations')

        numShuffs = 50
        cvTrain = 0.75
        corrShuffs = []
        corrCV = []
        for shuff in range(numShuffs):
            print(shuff)
            numTrls = bSOAvgMnSub.shape[0]
            trlIndsUse = np.random.permutation(range(numTrls))
            cca = CCA(n_components = compCCA)
            bSOAvgMnSubShuff = bSOAvgMnSub[trlIndsUse, :]
            bSOTransfShuff, bSTTransfShuff = cca.fit_transform(bSOAvgMnSubShuff, bSTAvgMnSub)
            bSOTransfShuff = bSOTransfShuff.T
            bSTTransfShuff = bSTTransfShuff.T
            corrShuffs.append(np.corrcoef(bSOTransfShuff, bSTTransfShuff).diagonal(compCCA))

            trlIndsTrain = trlIndsUse[:int(numTrls*cvTrain)]
            trlIndsTest = trlIndsUse[int(numTrls*cvTrain):]
            ccaCV = CCA(n_components = compCCA)
            ccaCV.fit(bSOAvgMnSub[trlIndsTrain, :], bSTAvgMnSub[trlIndsTrain, :])
            bSOTransfCV, bSTTransfCV = ccaCV.transform(bSOAvgMnSub[trlIndsTest, :], bSTAvgMnSub[trlIndsTest, :])
            bSOTransfCV = bSOTransfCV.T
            bSTTransfCV = bSTTransfCV.T
            corrCV.append(np.corrcoef(bSOTransfCV, bSTTransfCV).diagonal(compCCA))



        allShuffsStack = np.stack(corrShuffs)
        sortedCorrsShuffStack = np.stack([np.sort(shuff.copy())[::-1] for shuff in allShuffsStack])
        mnShuffs = sortedCorrsShuffStack.mean(axis=0)

        allCvStack = np.stack(corrCV)
        sortedCorrsCvStack = np.stack([np.sort(shuff.copy())[::-1] for shuff in allCvStack])
        mnCv = sortedCorrsCvStack.mean(axis=0)



        ciInt = 0.95
        ciIntEdgeDist = 1-ciInt

        ciEdgeDistShuff = [np.diff(np.sort(sortCorrs.copy())[[0,-1]])*ciIntEdgeDist/2 for sortCorrs in sortedCorrsShuffStack.T]
        smBgShuff = [np.sort(sortCorrs.copy())[[0,-1]] for sortCorrs in sortedCorrsShuffStack.T]

        axCorr.plot(mnShuffs, label=f'trial shuffle correlations ({numShuffs})', color=[0,0,0])
        axCorr.fill_between(np.arange(mnShuffs.shape[0]), np.stack([sB[0]+ciED for sB, ciED in zip(smBgShuff, ciEdgeDistShuff)]).squeeze(), np.stack([sB[1]-ciED for sB, ciED in zip(smBgShuff, ciEdgeDistShuff)]).squeeze(), alpha = 0.2, label='shuff 95\% confidence interval', color=[0,0,0], linewidth=0.0)

        ciEdgeDistCv = [np.diff(np.sort(sortCorrs.copy())[[0,-1]])*ciIntEdgeDist/2 for sortCorrs in sortedCorrsCvStack.T]
        smBgCv = [np.sort(sortCorrs.copy())[[0,-1]] for sortCorrs in sortedCorrsCvStack.T]
        ciBound = np.stack([sB+np.squeeze([ciED,-ciED]) for sB, ciED in zip(smBgCv, ciEdgeDistCv)]).squeeze()

        # ** POP BACK HERE **
        figPrincCorr, axCorr = plt.subplots(1,1,constrained_layout=True)
        plotCC = 25
        axCorr.plot(mnCv[:plotCC], label=f'{numShuffs} 75/25\% train/test CVs', color=[1,0,0])
        axCorr.fill_between(np.arange(plotCC), [cB[0] for cB in ciBound][:plotCC], [cB[1] for cB in ciBound][:plotCC], alpha = 0.2, label='95\% CI', color=[1,0,0], linewidth=0.0)
        axCorr.set_title(f'{brainAreaO}/{brainAreaT} correlations')

        # axCorr.legend()
        axCorr.set_xticks(np.arange(0, compCCA, 5))
        axCorr.set_xlabel('cca dim number')
        axCorr.set_ylabel('canonical correlation')

        xmin, xmax = axCorr.get_xlim()
        axCorr.hlines(0, xmin, xmax, linestyle='--')
        # ** POP TO HERE **

        outputFiguresRelativePath.append(saveFiguresToPdf(analysisDescription=f'ccaAnalysis{brainAreaO}_{brainAreaT}',pdfname='{}{}_princCorr'.format(plotParams['pdfnameSt'],plotParams['analysisIdentifier']), figNumsToSave = [figPrincCorr.number]))
        breakpoint()


        ciNumSig = np.nonzero((ciBound[:, 0] < 0) & (ciBound[:, 1] > 0))[0][0]
        ciNumSigHardcode = 5

        bsOCanonDirUse = bsOCanonDirs[:, :ciNumSig]
        bsTCanonDirUse = bsTCanonDirs[:, :ciNumSig]

        bsoCanonDirsOrthAll, _, _ = sp.linalg.svd(bsOCanonDirUse)
        bstCanonDirsOrthAll, _, _ = sp.linalg.svd(bsTCanonDirUse)
        bsoCanonDirsOrth = bsoCanonDirsOrthAll[:, :bsOCanonDirUse.shape[1]]
        bstCanonDirsOrth = bstCanonDirsOrthAll[:, :bsTCanonDirUse.shape[1]]

        projBsoIntoPotent =  bSOAvgMnSub @ bsoCanonDirsOrth
        projBstIntoPotent =  bSTAvgMnSub @ bstCanonDirsOrth

        bsONullCC = sp.linalg.null_space(bsOCanonDirUse.T)
        bsTNullCC = sp.linalg.null_space(bsTCanonDirUse.T)

        projBsoIntoNull = bSOAvgMnSub @ bsONullCC
        projBstIntoNull = bSTAvgMnSub @ bsTNullCC

        from classes.FA import FA
        crossvalidateNum = 4
        faBsoNull = FA(projBsoIntoNull[:,:,None], crossvalidateNum=crossvalidateNum)
        faBstNull = FA(projBstIntoNull[:,:,None], crossvalidateNum=crossvalidateNum)
        faBsoAll = FA(bSOAvgMnSub[:,:,None], crossvalidateNum=crossvalidateNum)

        maxFactTest = 55
        factStep = 5
        faScoresBso = []
        faScoresBst = []
        faScoresBsoAll = []
        initFactTest = np.arange(0, maxFactTest+1, factStep)
        print('coarse fitting both areas null FA')
        for numFact in initFactTest:
            print('testing dim {}'.format(numFact))
            # I'm gonna go ahead and assume a smooth increase 'til a max here...
            if len(faScoresBso) < 2 or (len(faScoresBso) >= 2 and faScoresBso[-1].sum() > faScoresBso[-2].sum()):
                faScoresBso.append(faBsoNull.runFa( numDim=numFact)[0])
            if len(faScoresBst) < 2 or (len(faScoresBst) >= 2 and faScoresBst[-1].sum() > faScoresBst[-2].sum()):
                faScoresBst.append(faBstNull.runFa( numDim=numFact)[0])
            if len(faScoresBsoAll) < 2 or (len(faScoresBsoAll) >= 2 and faScoresBsoAll[-1].sum() > faScoresBsoAll[-2].sum()):
                faScoresBsoAll.append(faBsoAll.runFa( numDim=numFact)[0])

        bsoNulBestFactInit = np.argmax([f.sum() for f in faScoresBso])
        bstNulBestFactInit = np.argmax([f.sum() for f in faScoresBst])
        bsoAllBestFactInit = np.argmax([f.sum() for f in faScoresBsoAll])

        bsoHonedFactTest = np.arange(initFactTest[bsoNulBestFactInit-1], initFactTest[bsoNulBestFactInit+1], 1)
        bstHonedFactTest = np.arange(initFactTest[bstNulBestFactInit-1], initFactTest[bstNulBestFactInit+1], 1)
        bsoAllHonedFactTest = np.arange(initFactTest[bsoAllBestFactInit-1], initFactTest[bsoAllBestFactInit+1], 1)

        faScoresHonedBso = []
        faScoresHonedBst = []
        faScoresHonedBsoAll = []
        print('fine fitting first area null FA')
        for numFact in bsoHonedFactTest:
            print('testing dim {}'.format(numFact))
            if len(faScoresHonedBso) < 2 or (len(faScoresHonedBso) >= 2 and faScoresHonedBso[-1].sum() > faScoresHonedBso[-2].sum()):
                faScoresHonedBso.append(faBsoNull.runFa(numDim=numFact)[0])
        bsoNullBestFactHoned = np.argmax([f.sum() for f in faScoresHonedBso])
        bsoBestFaDim = bsoHonedFactTest[bsoNullBestFactHoned]
        print('fine fitting second area null FA')
        for numFact in bstHonedFactTest:
            print('testing dim {}'.format(numFact))
            if len(faScoresHonedBst) < 2 or (len(faScoresHonedBst) >= 2 and faScoresHonedBst[-1].sum() > faScoresHonedBst[-2].sum()):
                faScoresHonedBst.append(faBstNull.runFa(numDim=numFact)[0])
        bstNullBestFactHoned = np.argmax([f.sum() for f in faScoresHonedBst])
        bstBestFaDim = bstHonedFactTest[bstNullBestFactHoned]
        print('fine fitting first area all FA')
        for numFact in bsoAllHonedFactTest:
            print('testing dim {}'.format(numFact))
            if len(faScoresHonedBsoAll) < 2 or (len(faScoresHonedBsoAll) >= 2 and faScoresHonedBsoAll[-1].sum() > faScoresHonedBsoAll[-2].sum()):
                faScoresHonedBsoAll.append(faBsoAll.runFa(numDim=numFact)[0])
        bsoAllBestFactHoned = np.argmax([f.sum() for f in faScoresHonedBsoAll])
        bsoAllBestFaDim = bsoAllHonedFactTest[bsoAllBestFactHoned]


        # rerun FA at the best dimensionality with all the data
        faBsoNullFull = FA(projBsoIntoNull[:,:,None], crossvalidateNum=1)
        faBstNullFull = FA(projBstIntoNull[:,:,None], crossvalidateNum=1)
        faBsoAllFull = FA(bSOAvgMnSub[:,:,None], crossvalidateNum=1)

        bsoEstParams = faBsoNullFull.runFa( numDim=bsoBestFaDim)[1][0]
        bstEstParams = faBstNullFull.runFa( numDim=bstBestFaDim)[1][0]
        bsoAllEstParams = faBsoAllFull.runFa( numDim = bsoAllBestFaDim)[1][0]

        bsoAC = bsoAllEstParams['C']
        bsoAR = bsoAllEstParams['R']
        shVar = np.diag((bsoAC @ bsoAC.T)/(bsoAC @ bsoAC.T + bsoAR)).mean()
        _, sv, _ = np.linalg.svd(bsoAC @ bsoAC.T)


        projBsoIntoPotentHiD = projBsoIntoPotent @ bsoCanonDirsOrth.T
        projBstIntoPotentHiD = projBstIntoPotent @ bstCanonDirsOrth.T
        bsoBtAreaCov = np.cov(projBsoIntoPotentHiD.T)
        bstBtAreaCov = np.cov(projBstIntoPotentHiD.T)

        bsoC = bsoEstParams['C']
        bsoR = bsoEstParams['R']
        bstC = bstEstParams['C']
        bstR = bstEstParams['R']

        bsoCHiD = bsONullCC @ bsoC
        bstCHiD = bsTNullCC @ bstC

        bsoBtAreaVarPerNeur = np.diag(bsoBtAreaCov)
        bstBtAreaVarPerNeur = np.diag(bstBtAreaCov)

        bsoWiAreaVarPerNeur = np.diag(bsoCHiD @ bsoCHiD.T)
        bstWiAreaVarPerNeur = np.diag(bstCHiD @ bstCHiD.T)

        bsoPrivVarPerNeur = np.diag(bsONullCC @ bsoR @ bsONullCC.T)
        bstPrivVarPerNeur = np.diag(bsTNullCC @ bstR @ bsTNullCC.T)

        bsoBtAreaVar = (bsoBtAreaVarPerNeur/(bsoBtAreaVarPerNeur + bsoWiAreaVarPerNeur + bsoPrivVarPerNeur)).mean()
        bsoWiAreaVar = (bsoWiAreaVarPerNeur/(bsoBtAreaVarPerNeur + bsoWiAreaVarPerNeur + bsoPrivVarPerNeur)).mean()
        bsoPrivVar = (bsoPrivVarPerNeur/(bsoBtAreaVarPerNeur + bsoWiAreaVarPerNeur + bsoPrivVarPerNeur)).mean()

        bstBtAreaVar = (bstBtAreaVarPerNeur/(bstBtAreaVarPerNeur + bstWiAreaVarPerNeur + bstPrivVarPerNeur)).mean()
        bstWiAreaVar = (bstWiAreaVarPerNeur/(bstBtAreaVarPerNeur + bstWiAreaVarPerNeur + bstPrivVarPerNeur)).mean()
        bstPrivVar = (bstPrivVarPerNeur/(bstBtAreaVarPerNeur + bstWiAreaVarPerNeur + bstPrivVarPerNeur)).mean()

        bsoVars = np.stack([bsoBtAreaVar, bsoWiAreaVar, bsoPrivVar])
        bstVars = np.stack([bstBtAreaVar, bstWiAreaVar, bstPrivVar])
        allVars = np.vstack([bsoVars, bstVars])

        breakpoint()
        # ** POP BACK HERE **
        figPie, axsPie = plt.subplots(1,2)
        varLabels = ('b/t area var', 'w/i area var', 'private var')
        axsPie[0].pie(bsoVars, labels = varLabels, autopct='%1.1f%%', startangle=90)
        axsPie[0].axis('equal')
        axsPie[0].set_title(f'{brainAreaO}')
        axsPie[1].pie(bstVars, labels = varLabels, autopct='%1.1f%%', startangle=90)
        axsPie[1].axis('equal')
        axsPie[1].set_title(f'{brainAreaT}')
        # ** POP BACK TO HERE **

        outputFiguresRelativePath.append(saveFiguresToPdf(analysisDescription=f'ccaAnalysis{brainAreaO}_{brainAreaT}',pdfname='{}{}_shBtWiPrivVar'.format(plotParams['pdfnameSt'],plotParams['analysisIdentifier']), figNumsToSave = [figPrincCorr.number]))

        from importlib import reload
        from methods.plotUtils import PlotUtils 
        from methods import GeneralMethods

        # ** POP BACK HERE **
        # reload(PlotUtils)
        MoveAxisToSubplot = PlotUtils.MoveAxisToSubplot
        MoveFigureToSubplot = PlotUtils.MoveFigureToSubplot
        finalFig, finalAx = plt.subplots(3,4,constrained_layout=True)
        MoveFigureToSubplot(figPrincCorr, finalFig, finalAx[1,0])
        MoveAxisToSubplot(axsAttFirst[0,0], finalFig, finalAx[1,1])
        [fAx.remove() for fAx in finalAx[0]]
        [fAx.remove() for fAx in finalAx[2]]
        # MoveFigureToSubplot(plt.figure(figAnglesToSave[0]), finalFig, finalAx[1,1])
        MoveAxisToSubplot(axsPie[0], finalFig, finalAx[1,2])
        MoveAxisToSubplot(axsPie[1], finalFig, finalAx[1,3])

        # finalFig.tight_layout()
        # finalFig.set_constrained_layout(True)
        # finalFig.set_constrained_layout_pads(w_pad=0, h_pad=0, hspace=0, wspace=0)
        # reload(GeneralMethods)
        # saveFiguresToPdf = GeneralMethods.saveFiguresToPdf
        saveFiguresToPdf(analysisDescription=f'ccaAnalysis{brainAreaO}_{brainAreaT}', pdfname='finalFig', figNumsToSave=[finalFig.number])
        # ** POP BACK TO HERE **
        breakpoint()



    outputInfo = {}
    outputInfo.update(dict(outputFiguresRelativePath = outputFiguresRelativePath)) if len(outputFiguresRelativePath)>0 else None

    outputInfo.update(dict(
        allMetrics = metricDict,
        groupingInfo = dict(
            descriptions = descriptions,
            brainAreas = brainAreas,
            tasks = tasks,
        ),
        subsamplesUsed = subsampleExpressions,
        )
    )

    return outputInfo


if __name__=='__main__':
    ccaDescriptiveAnalysis()
