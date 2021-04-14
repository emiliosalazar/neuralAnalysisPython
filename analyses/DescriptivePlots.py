"""
This function is responsible for outputting descriptive plots of datasets--useful for checking new datasets to see that they conform to what's expected;
Some of what's on here is like the olde HeadlessAnalysisRun4.py for now...
"""
from matplotlib import pyplot as plt
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['GOTO_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np
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

# for the gpfa...
from methods.BinnedSpikeSetListMethods import gpfaComputation

# for descriptions of the data
from methods.BinnedSpikeSetListMethods import generateLabelGroupStatistics as genBSLLabGrp
from methods.BinnedSpikeSetListMethods import plotExampleChannelResponses
from methods.BinnedSpikeSetListMethods import plotStimDistributionHistograms
from methods.BinnedSpikeSetListMethods import plotFiringRates

# for computing metrics
from methods.BinnedSpikeSetListMethods import rscComputations
from methods.GpfaMethods import computePopulationMetrics

# for plotting the metrics
from methods.plotUtils.PlotUtils import MoveFigureToSubplot, MoveAxisToSubplot


@saveCallsToDatabase
def datasetDescriptiveOverview(datasetSqlFilter, binnedSpikeSetGenerationParamsDict, plotParams):

    dsgl = DatasetGeneralLoadParams()
    dsi = DatasetInfo()
    bsi = BinnedSpikeSetInfo()
    fsp = FilterSpikeSetParams()

    outputFiguresRelativePath = []

    if len(datasetSqlFilter)>0:
        dsiUse = dsi[datasetSqlFilter]
    else:
        dsiUse = dsi

    print("Computing/loading binned spike sets")
    _, bssKeys = genBSLAroundState(dsiUse,
                                        **binnedSpikeSetGenerationParamsDict
                                        )


    if 'plotChannelRespParams' in plotParams and plotParams['plotChannelRespParams']:
        plotChannelRespParams = plotParams['plotChannelRespParams']
        for bssKeyOne in bssKeys:
            bssExp = bsi[bssKeyOne]
            dsID = dsi[bssExp]['dataset_name'][0].replace(' ', '_')
            pdfName = dsID + '_' + bssExp['bss_hash'][0][:5]

            bssExp.generateDescriptivePlots(plotTypes = plotChannelRespParams)
            saveFiguresToPdf(pdfname=pdfName, analysisDescription="descriptivePlots")
            plt.close('all')

    if 'plotOverviewInfoForSession' in plotParams and plotParams['plotOverviewInfoForSession']:
        plotOverviewParams = plotParams['plotOverviewInfoForSession']
        for bssKeyOne in bssKeys:
            bssExp = bsi[bssKeyOne]
            dsID = dsi[bssExp]['dataset_name'][0].replace(' ', '_')
            pdfName = dsID + '_' + bssExp['bss_hash'][0][:5]
            bss = bssExp.grabBinnedSpikes()[0]
            bss.labels['stimulusMainLabel'] = bss.labels['stimulusMainLabel'].astype('float64')


            cosTunCrvPrms = bss.computeCosTuningCurves(plot=True)
            cosTCFig = plt.gcf()

            mnMod = cosTunCrvPrms['modPerChan'].mean()

            plotChannelRespParams = plotOverviewParams['plotChannelRespParams']
#            mnChan = [np.argmin(np.abs(cosTunCrvPrms['modPerChan']-mnMod))]
            mnChan = [np.argmax(cosTunCrvPrms['modPerChan'])]

            if 'raster' in plotChannelRespParams:
                bssExp.generateDescriptivePlots(plotTypes = {'raster': plotChannelRespParams['raster']}, chPlot = [mnChan])
            elif 'psth' in plotChannelRespParams:
                bssExp.generateDescriptivePlots(plotTypes = {'psth': plotChannelRespParams['psth']}, chPlot = [mnChan])

            respFig = plt.gcf()

            plotGenericMetricParams = plotOverviewParams['plotGenericMetrics']
            supTitle = plotGenericMetricParams['supTitle']

            plotFiringRates(bss.convertUnitsTo('Hz'), dsID, supTitle = supTitle + " firing rates", cumulative = False)
            frFig = plt.gcf()
            mnVsStdFrSubplot = frFig.axes[2]

            plotFiringRates(bss.convertUnitsTo('count'), dsID, supTitle = supTitle + " count", cumulative = False)
            scFig = plt.gcf()
            mnVsStdCountSubplot = scFig.axes[2]

            naiveBayesAcc = bss.decode(labels=bss.labels['stimulusMainLabel'], trainFrac = 0.75)
            bss.timeAverage().pca(labels = bss.labels['stimulusMainLabel'], plot = {'supTitle' : "naive Bayes decoding accuracy of {:.2f}%".format(naiveBayesAcc*100)})
            pcaFig = plt.gcf()
#            MoveAxisToSubplot(mnVsStdCountSubplot, frFig, mnVsStdFrSubplot, removeOldFig = True)

#            overviewFig = plt.figure()#constrained_layout=True)
#            gs = overviewFig.add_gridspec(4,4)
#            sbPltFr = overviewFig.add_subplot(gs[0,:2])
#            sbPltCosTCParams = overviewFig.add_subplot(gs[1, :2])
#            sbPltOneChan = overviewFig.add_subplot(gs[:2, 2:])
#
#            saveFiguresToPdf(pdfname=pdfName+'init', analysisDescription="overviewPlot")
#
#            MoveFigureToSubplot(frFig, overviewFig, sbPltFr)
#            saveFiguresToPdf(pdfname=pdfName+'init1', analysisDescription="overviewPlot")
#            MoveFigureToSubplot(cosTCFig, overviewFig, sbPltCosTCParams)
#            saveFiguresToPdf(pdfname=pdfName+'init2', analysisDescription="overviewPlot")
#            MoveFigureToSubplot(respFig, overviewFig, sbPltOneChan)
#            saveFiguresToPdf(pdfname=pdfName+'init3', analysisDescription="overviewPlot")
#
#            overviewFig.tight_layout()
            saveFiguresToPdf(pdfname=pdfName+'new', analysisDescription="overviewPlot")

            plt.close('all')


#            outputFiguresRelativePath.append(saveFiguresToPdf(pdfname="{}{}".format(plotGenericMetricParams['pdfnameSt'],plotParams['analysisIdentifier'])))
#            plt.close('all')

    if 'plotKinematicDecoding' in plotParams and plotParams['plotKinematicDecoding']:
        plotKinParams = plotParams['plotKinematicDecoding']

        for bssKeyOne in bssKeys:
            bssExp = bsi[bssKeyOne]
            bss = bssExp.grabBinnedSpikes()[0]
            bss.labels['stimulusMainLabel'] = bss.labels['stimulusMainLabel'].astype('float64')
            dsID = dsi[bssExp]['dataset_name'][0].replace(' ', '_')
            matchedKinematics = bssExp.grabAlignedKinematics(kinBinning='binToMatch')
            # NOTE ONLY TRUE if this is the period POST cursor on...
            for trl, (kinTrl, bsTrl) in enumerate(zip(matchedKinematics, bss)):
                for ch, chResp in enumerate(bsTrl):
                    bss[trl,ch] = chResp[~np.isnan(kinTrl[:, 1])]
                matchedKinematics[trl] = kinTrl[~np.isnan(kinTrl[:, 1]), :]
        
            pred = [np.linalg.lstsq(np.stack(spks).T, kin, rcond=None)[0] for kin, spks in zip(matchedKinematics, bss)]
            
            trnFrac = 0.75
            numTrls = bss.shape[0]
            randomizedIndOrder = np.random.permutation(np.arange(numTrls))
            trainInds = randomizedIndOrder[:round(trnFrac*numTrls)]
            testInds = randomizedIndOrder[round(trnFrac*numTrls):]

            allSpkTrnTog = np.concatenate(bss[trainInds], axis=2).squeeze().T
            allSpkTrnTog = np.concatenate([np.array(allSpkTrnTog), np.ones_like(allSpkTrnTog[:,None,0])], axis=1)
            allKinTrnTog = np.concatenate(matchedKinematics[trainInds],axis=0)
            lstSqFit = np.linalg.lstsq(allSpkTrnTog, allKinTrnTog, rcond=None)[0]

            estimates = np.array([np.concatenate([np.array(np.stack(spk).T), np.ones_like(np.stack(spk).T[:,None,0])],axis=1) @ lstSqFit for spk in bss[testInds]])
            trueVals = matchedKinematics[testInds]


            unLb, lbTrl = np.unique(bss[testInds].labels['stimulusMainLabel'], return_inverse=True)

            fgLinEst, axs = plt.subplots(2, int(np.ceil(unLb.shape[0]/2)))
            fgLinEst.tight_layout()

            for uLb, ax in enumerate(axs.flatten()):
                trlH = lbTrl==uLb
                [ax.plot(pts[:, 0], pts[:, 1], 'k-') for pts in trueVals[trlH]]
                [ax.plot(pts[:, 0], pts[:, 1], 'r--') for pts in estimates[trlH]]
                ax.set_title(unLb[uLb])

            mnX = np.min([np.min(a.get_xlim()) for a in axs.flatten()])
            mxX = np.max([np.max(a.get_xlim()) for a in axs.flatten()])
            mnY = np.min([np.min(a.get_ylim()) for a in axs.flatten()])
            mxY = np.max([np.max(a.get_ylim()) for a in axs.flatten()])

            [a.set_xlim(mnX, mxX) for a in axs.flatten()]
            [a.set_ylim(mnY, mxY) for a in axs.flatten()]

            fgLinEst.suptitle('true vs estimated traj')

            pdfName = dsID + '_' + bssExp['bss_hash'][0][:5]
            saveFiguresToPdf(pdfname=pdfName, analysisDescription="kinematicPlots")
            plt.close('all')

            breakpoint()

def coincidenceOverview(datasetSqlFilter):

    dsgl = DatasetGeneralLoadParams()
    dsi = DatasetInfo()
    bsi = BinnedSpikeSetInfo()
    fsp = FilterSpikeSetParams()

    outputFiguresRelativePath = []

    if len(datasetSqlFilter)>0:
        dsiUse = dsi[datasetSqlFilter]
    else:
        dsiUse = dsi

    for dsKey in dsi[datasetSqlFilter].fetch('KEY'):
        ds = dsi[dsKey]
        suptitle = ds['dataset_name'][0]
        ds.createRawDataset().findCoincidentSpikeChannels(coincidenceTime=1, coincidenceThresh=0.2,checkNumTrls=1,plotResults=True)

        coincFig = plt.gcf()
        coincFig.suptitle(suptitle)

    saveFiguresToPdf(pdfname="coincidenceOverview", analysisDescription="coincidenceOverview")
    plt.close('all')



