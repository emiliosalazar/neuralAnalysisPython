"""
This function is responsible for outputting descriptive plots of datasets--useful for checking new datasets to see that they conform to what's expected;
Some of what's on here is like the olde HeadlessAnalysisRun4.py for now...
"""
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import plotly as ply
import plotly.express as px 
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
from setup.DataJointSetup import DatasetInfo, DatasetGeneralLoadParams, BinnedSpikeSetInfo, FilterSpikeSetParams, BinnedSpikeSetProcessParams
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
from methods.plotMethods.PlotUtils import MoveFigureToSubplot, MoveAxisToSubplot


# @saveCallsToDatabase
def binnedSpikesDescriptiveOverview(datasetSqlFilter, binnedSpikeSetGenerationParamsDict, plotParams):

    dsgl = DatasetGeneralLoadParams()
    dsi = DatasetInfo()
    bsi = BinnedSpikeSetInfo()
    fsp = FilterSpikeSetParams()
    bsp = BinnedSpikeSetProcessParams()

    colorsUse = BinnedSpikeSet.colorset

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

            bssExp.generateDescriptivePlots(plotTypes = plotChannelRespParams, stateName=binnedSpikeSetGenerationParamsDict['keyStateName'])
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
                bssExp.generateDescriptivePlots(plotTypes = {'raster': plotChannelRespParams['raster']}, stateName=binnedSpikeSetGenerationParamsDict['keyStateName'], chPlot = [mnChan])
            elif 'psth' in plotChannelRespParams:
                bssExp.generateDescriptivePlots(plotTypes = {'psth': plotChannelRespParams['psth']}, stateName=binnedSpikeSetGenerationParamsDict['keyStateName'], chPlot = [mnChan])

            respFig = plt.gcf()

            plotGenericMetricParams = plotOverviewParams['plotGenericMetrics']
            supTitle = plotGenericMetricParams['supTitle']

            plotFiringRates(bss.convertUnitsTo('Hz'), dsID, supTitle = supTitle + " firing rates", cumulative = False)
            frFig = plt.gcf()
            mnVsStdFrSubplot = frFig.axes[2]

            plotFiringRates(bss.convertUnitsTo('count'), dsID, supTitle = supTitle + " count", cumulative = False)
            scFig = plt.gcf()
            mnVsStdCountSubplot = scFig.axes[2]

            if plotChannelRespParams['decode']:
                decodeMethod = plotChannelRespParams['decode']['decodeMethod']
                decodeAcc = bss.decode(decodeType = decodeMethod, labels=bss.labels['stimulusMainLabel'], trainFrac = 0.75)[0]
                bss.timeAverage().pca(labels = bss.labels['stimulusMainLabel'], plot = {'supTitle' : "{} decoding accuracy of {:.2f}%".format(decodeMethod, decodeAcc*100)})
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

    if 'plotKinematics' in plotParams and plotParams['plotKinematics']:
        plotKinParams = plotParams['plotKinematics']

        for bssKeyOne in bssKeys:
            bssExp = bsi[bssKeyOne]
            bss = bssExp.grabBinnedSpikes()[0]
            bss.labels['stimulusMainLabel'] = bss.labels['stimulusMainLabel'].astype('float64')
            dsID = dsi[bssExp]['dataset_name'][0].replace(' ', '_')
            dsFilteredToBssTrls = bssExp.grabFilteredDataset()

            matchedKinematics = bssExp.grabAlignedKinematics(kinBinning='binToMatch')
            # NOTE ONLY TRUE if this is the period POST cursor on...
            for trl, (kinTrl, bsTrl) in enumerate(zip(matchedKinematics, bss)):
                for ch, chResp in enumerate(bsTrl):
                    bss[trl,ch] = chResp[~np.isnan(kinTrl[:, 1])]
                matchedKinematics[trl] = kinTrl[~np.isnan(kinTrl[:, 1]), :]

            unLb, lbTrl = np.unique(bss.labels['stimulusMainLabel'], return_inverse=True)

            fgIndCondKin, axsInd = plt.subplots(2, int(np.ceil(unLb.shape[0]/2)))
            fgTogTrajAndRT, axsOverall = plt.subplots(2, 2)
            axsRTandAT = axsOverall[:, 1]
            fgTogTrajAndRT.delaxes(axsOverall[0, 0])
            fgTogTrajAndRT.delaxes(axsOverall[1, 0])
            axsOverallTraj = fgTogTrajAndRT.add_subplot(1,2,1)


            trajInCombo = []
            for uLb, ax in enumerate(axsInd.flatten()):
                trlH = lbTrl==uLb
                [ax.plot(pts[:, 0], pts[:, 1], 'k-', alpha=0.05) for pts in matchedKinematics[trlH]]
                ax.set_title(unLb[uLb])

                [axsOverallTraj.plot(pts[:, 0], pts[:, 1], color=colorsUse[uLb, :], alpha=0.05) for pts in matchedKinematics[trlH]]
                # oneTrlPts = matchedKinematics[trlH.nonzero()[0][0]]
                # trajInCombo += axsOverallTraj.plot(oneTrlPts[:, 0], oneTrlPts[:, 1], color=colorsUse[uLb, :], alpha=0.05, label=unLb[uLb])
                trajInCombo += [Line2D([0],[0], color=colorsUse[uLb,:], label=unLb[uLb])]

            axsOverallTraj.legend(handles = trajInCombo,prop={'size':5},loc='lower left')
            axsOverallTraj.set_title('successful trajectories')
            axsOverallTraj.set_xlabel('pixels')
            axsOverallTraj.set_ylabel('pixels')

            mnX = np.min([np.min(a.get_xlim()) for a in axsInd.flatten()])
            mxX = np.max([np.max(a.get_xlim()) for a in axsInd.flatten()])
            mnY = np.min([np.min(a.get_ylim()) for a in axsInd.flatten()])
            mxY = np.max([np.max(a.get_ylim()) for a in axsInd.flatten()])

            [a.set_xlim(mnX, mxX) for a in axsInd.flatten()]
            [a.set_ylim(mnY, mxY) for a in axsInd.flatten()]

            fgIndCondKin.suptitle('successful trajectories')
            fgIndCondKin.tight_layout(rect=[0, 0.03, 1, 0.95])

            # histograms of reaction time
            stTm,endTm,stEndNm = dsFilteredToBssTrls.computeStateStartAndEnd(stateName = dsFilteredToBssTrls.keyStates['go cue'])
            endTm = np.array(endTm)
            stTm = np.array(stTm)
            rt = endTm - stTm
            binSize = 5 # ms
            binStart = np.floor(rt.min()/binSize)*binSize # this gets to the nearest binSize 
            histHands = []
            for uLb, uLbNm in enumerate(unLb):
                trlH = lbTrl==uLb
                histInfo =  axsRTandAT[0].hist(rt[trlH],bins = np.arange(rt.min(), rt.max()+binSize, binSize), color=colorsUse[uLb,:], alpha=2/len(unLb), label=uLbNm)
                histHands.append(histInfo[-1][0])

            axsRTandAT[0].legend(handles = histHands,prop={'size':5},loc='best')
            axsRTandAT[0].set_title('reaction times')
            axsRTandAT[0].set_xlabel('time (ms)')
            axsRTandAT[0].set_ylabel('count')

            # histograms of action time
            stTm = dsFilteredToBssTrls.timeOfState(dsFilteredToBssTrls.keyStates['action'])
            endTm = dsFilteredToBssTrls.timeOfState('CORRECT')
            stTm = np.array(stTm)
            endTm = np.array(endTm)
            at = endTm - stTm
            binSize = 20 # ms
            binStart = np.floor(at.min()/binSize)*binSize # this gets to the nearest binSize 
            histHands = []
            for uLb, uLbNm in enumerate(unLb):
                if uLb==2:
                    breakpoint()
                trlH = lbTrl==uLb
                histInfo =  axsRTandAT[1].hist(at[trlH],bins = np.arange(at.min(), at.max()+binSize, binSize), color=colorsUse[uLb,:], alpha=2/len(unLb), label=uLbNm)
                histHands.append(histInfo[-1][0])

            axsRTandAT[1].legend(handles = histHands,prop={'size':5},loc='best')
            axsRTandAT[1].set_title('movement times')
            axsRTandAT[1].set_xlabel('time (ms)')
            axsRTandAT[1].set_ylabel('count')
            fgTogTrajAndRT.tight_layout()




            pdfName = dsID + '_' + bssExp['bss_hash'][0][:5]
            saveFiguresToPdf(pdfname=pdfName, analysisDescription="kinematicPlots")
            plt.close('all')

    if 'plotKinematicDecoding' in plotParams and plotParams['plotKinematicDecoding']:
        plotKinParams = plotParams['plotKinematicDecoding']

        for bssKeyOne in bssKeys:
            bssExp = bsi[bssKeyOne]
            bss = bssExp.grabBinnedSpikes()[0]
            bss.labels['stimulusMainLabel'] = bss.labels['stimulusMainLabel'].astype('float64')
            dsID = dsi[bssExp]['dataset_name'][0].replace(' ', '_')
            matchedKinematicsPos = bssExp.grabAlignedKinematics(kinBinning='binToMatch')
            binSizeMs = bsp[bssExp]['bin_size']
            binSizeS = binSizeMs/1000
            # NOTE ONLY TRUE if this is the period POST cursor on...
            for trl, (kinTrl, bsTrl) in enumerate(zip(matchedKinematicsPos, bss)):
                for ch, chResp in enumerate(bsTrl):
                    bss[trl,ch] = chResp[~np.isnan(kinTrl[:, 1])]
                matchedKinematicsPos[trl] = kinTrl[~np.isnan(kinTrl[:, 1]), :]
            
            # pixels/ms velocity
            matchedKinematics = np.array([np.vstack([[0,0],np.diff(mK,axis=0)/binSizeS]) for mK in matchedKinematicsPos])
        
            # pred = [np.linalg.lstsq(np.stack(np.array(spks)).T, kin, rcond=None)[0] for kin, spks in zip(matchedKinematics, bss)]
            
            trnFrac = 0.75
            numTrls = bss.shape[0]
            randomizedIndOrder = np.random.permutation(np.arange(numTrls))
            trainInds = randomizedIndOrder[:round(trnFrac*numTrls)]
            testInds = randomizedIndOrder[round(trnFrac*numTrls):]

            # allSpkTrnTog = np.concatenate(bss[trainInds], axis=2).squeeze().T
            # allSpkTrnTog = np.concatenate([np.array(allSpkTrnTog), np.ones_like(allSpkTrnTog[:,None,0])], axis=1)
            # allKinTrnTog = np.concatenate(matchedKinematics[trainInds],axis=0)
            # lstSqFit = np.linalg.lstsq(allSpkTrnTog, allKinTrnTog, rcond=None)[0]

            # estimates = np.array([np.concatenate([np.array(np.stack(spk).T), np.ones_like(np.stack(spk).T[:,None,0])],axis=1) @ lstSqFit for spk in bss[testInds]])
            # trueVals = matchedKinematics[testInds]

            allSpksCurrStp = [np.stack(bsT)[:,1:] if bsT.size>0 else np.nan for bsT in bss]
            allSpksPrevStp = [np.stack(bsT)[:,:-1] if bsT.size>0 else np.nan for bsT in bss]
            allKinCurrStp = [aK[1:,:] if aK.size>0 else np.nan for aK in matchedKinematics]
            allKinPrevStp = [aK[:-1,:] if aK.size>0 else np.nan for aK in matchedKinematics]


            allKinCurrStpConcat = np.vstack(allKinCurrStp).T
            kinMean = np.mean(allKinCurrStpConcat,axis=1)
            allSpksCurrStpConcat = np.hstack(allSpksCurrStp)

            if 'kalman' in plotKinParams['decodeTypes']:
                A = np.eye(allKinCurrStpConcat.shape[0])
                Q = 100e3*np.eye(allKinCurrStpConcat.shape[0])

                from classes.FA import FA
                faMod = FA(allSpksCurrStpConcat[None, :, :], 1)
                _, faParams, allLatentProjInfo, *_ = faMod.runFa(numDim = 10)
                allLatentProj = allLatentProjInfo[0][0]['xorth']
                faParams = faParams[0]
                beta = faParams['beta']
                allLatentProj = beta @ (allSpksCurrStpConcat-faParams['d'][:,None])
                meanLatProj = np.mean(allLatentProj,axis=1)

                C = allLatentProj @ allKinCurrStpConcat.T @ np.linalg.inv(allKinCurrStpConcat @ allKinCurrStpConcat.T)
                Tall = allLatentProj.shape[1]
                R = 1/Tall * (allLatentProj - C @ allKinCurrStpConcat) @ (allLatentProj - C@allKinCurrStpConcat).T

                # get converged Kalman gain
                sigCurrGivenPrev = np.cov(allKinCurrStpConcat);
                muCurrGivenPrev = np.nanmean(allKinCurrStpConcat, axis=1);


                Kall = []
                for t in range(100):
                    Kcurr = sigCurrGivenPrev @ C.T @ np.linalg.inv(C @ sigCurrGivenPrev @ C.T + R)
                    sigCurrGivenCurr = sigCurrGivenPrev - Kcurr @ C @ sigCurrGivenPrev;
                    sigCurrGivenPrev = A @ sigCurrGivenCurr @ A.T + Q;
                    Kall.append(Kcurr)

                K = Kall[-1]

                M1 = A - K@C@A;
                M2 = K @ beta;
                # baseline takes care of accounting for mean offsets in training
                M0 = -M1 @ kinMean - M2@ faParams['d'] - K@meanLatProj
                


                unLb, lblTrl = np.unique(bss.labels['stimulusMainLabel'], return_inverse=True)
                figKalmanDecode = px.line()
                targDist = 300
                targRad = 50
                availableColors = px.colors.qualitative.Plotly
                [figKalmanDecode.add_shape(type='circle',
                        xref='x',yref='y',
                        x0=xTarg-targRad,y0=yTarg-targRad,
                        x1=xTarg+targRad,y1=yTarg+targRad,
                        fillcolor=availableColors[targNum],
                        # line_color=availableColors[targNum]
                        opacity=0.5,
                        row=1,
                        col=1,
                    ) for targNum, (xTarg, yTarg) in enumerate(zip(targDist*np.cos(unLb/180*np.pi), targDist*np.sin(unLb/180*np.pi)))];

                from copy import copy
                figTrueTraj = copy(figKalmanDecode)


                legendShown = {}
                [legendShown.update({tA : True}) for tA in unLb]
                boundMax = 0
                for trlSpks, trueTraj, lbl in zip(allSpksCurrStp, allKinCurrStp, lblTrl):
                    targAng = unLb[lbl]
                    targAngStr = str(targAng)
                    trueTraj = trueTraj.T
                    trlLat = beta @ (trlSpks - faParams['d'][:,None]) - meanLatProj[:,None]
                    
                    # muCurrGivenPrev = np.nanmean(allKinCurrStpConcat, axis=1)
                    # outVel = np.zeros_like(trueTraj)
                    # for t in np.arange(trueTraj.shape[1]):
                    #     muCurrGivenCurr = muCurrGivenPrev + K@(trlLat[:, t] - C@muCurrGivenPrev)
                    #     outVel[:,t] = muCurrGivenCurr
                    #     muCurrGivenPrev = A@(muCurrGivenCurr - kinMean)

                    outVel = np.zeros_like(trueTraj)
                    for t in np.arange(1, trueTraj.shape[1]):
                        outVel[:,t] = M0 + M1 @ outVel[:,t-1] + M2 @ trlSpks[:, t]

                    outPos = outVel.cumsum(axis=1)*binSizeS
                    figKalmanDecode.add_scatter(x=outPos[0,:].squeeze(), y=outPos[1,:].squeeze(),
                        opacity=0.5, legendgroup = 'group' + targAngStr, mode='lines',
                        line=dict(color=availableColors[lbl]), name=targAngStr,
                        showlegend=legendShown.pop(targAng, False), legendrank=lbl)
                    
                    trueTrajPos = trueTraj.cumsum(axis=1)*binSizeS
                    xMinTraj, yMinTraj = np.min(trueTrajPos, axis=1) 
                    xMaxTraj, yMaxTraj = np.max(trueTrajPos, axis=1) 
                    boundMax = 1.1*np.max(np.abs([boundMax, xMinTraj, yMinTraj, xMaxTraj, yMaxTraj]))
                    figTrueTraj.add_scatter(x=trueTrajPos[0,:].squeeze(), y=trueTrajPos[1,:].squeeze(),
                        opacity=0.5, legendgroup = 'group' + targAngStr, mode='lines',
                        line=dict(color=availableColors[lbl]), name=targAngStr,
                        showlegend=legendShown.pop(targAng, False), legendrank=lbl)


                outName = 'kalman decode'
                xMinTraj, yMinTraj = np.min(allKinCurrStpConcat, axis=1) 
                xMaxTraj, yMaxTraj = np.max(allKinCurrStpConcat, axis=1) 
                boundMax = 1.1*np.max(np.abs([xMinTraj, yMinTraj, xMaxTraj, yMaxTraj]))
                figKalmanDecode.update_layout(
                        title=outName,
                        xaxis=dict(range=[-boundMax, boundMax]),
                        yaxis=dict(range=[-boundMax, boundMax], scaleanchor = "x", scaleratio = 1)
                    );
                figTrueTraj.update_layout(
                        title="true trajectories",
                        xaxis=dict(range=[-boundMax, boundMax]),
                        yaxis=dict(range=[-boundMax, boundMax], scaleanchor = "x", scaleratio = 1)
                    );


                figKalmanDecode.show()
                figTrueTraj.show()

            breakpoint()
            # unLb, lbTrl = np.unique(bss[testInds].labels['stimulusMainLabel'], return_inverse=True)

            fgLinEst, axs = plt.subplots(2, int(np.ceil(unLb.shape[0]/2)))

            for uLb, ax in enumerate(axs.flatten()):
                trlH = lbTrl==uLb
                [ax.plot(pts[:, 0], pts[:, 1], 'k-', alpha=0.05) for pts in trueVals[trlH]]
                [ax.plot(pts[:, 0], pts[:, 1], 'r--', alpha=0.05) for pts in estimates[trlH]]
                ax.set_title(unLb[uLb])

            mnX = np.min([np.min(a.get_xlim()) for a in axs.flatten()])
            mxX = np.max([np.max(a.get_xlim()) for a in axs.flatten()])
            mnY = np.min([np.min(a.get_ylim()) for a in axs.flatten()])
            mxY = np.max([np.max(a.get_ylim()) for a in axs.flatten()])

            [a.set_xlim(mnX, mxX) for a in axs.flatten()]
            [a.set_ylim(mnY, mxY) for a in axs.flatten()]

            fgLinEst.suptitle('true vs estimated traj')
            fgLinEst.tight_layout(rect=[0, 0.03, 1, 0.95])

            pdfName = dsID + '_' + bssExp['bss_hash'][0][:5]
            saveFiguresToPdf(pdfname=pdfName, analysisDescription="decodeKinematics")
            plt.close('all')

            breakpoint()

def datasetDescriptiveOverview(datasetSqlFilter, plotParams):

    dsgl = DatasetGeneralLoadParams()
    dsi = DatasetInfo()
    bsi = BinnedSpikeSetInfo()
    fsp = FilterSpikeSetParams()

    outputFiguresRelativePath = []

    if len(datasetSqlFilter)>0:
        dsiUse = dsi[datasetSqlFilter]
    else:
        dsiUse = dsi

    if 'coincidenceOverview' in plotParams and plotParams['coincidenceOverview']:
        coincidenceOverviewParams = plotParams['coincidenceOverview']

        for dsKey in dsi[datasetSqlFilter].fetch('KEY'):
            ds = dsi[dsKey]
            suptitle = ds['dataset_name'][0]
            ds.createRawDataset().findCoincidentSpikeChannels(coincidenceTime=1, coincidenceThresh=0.2,checkNumTrls=1,plotResults=True)

            coincFig = plt.gcf()
            coincFig.suptitle(suptitle)

        saveFiguresToPdf(pdfname="coincidenceOverview", analysisDescription="coincidenceOverview")
        plt.close('all')

    if 'keyStateOverview' in plotParams and plotParams['keyStateOverview']:
        keyStateParams = plotParams['keyStateOverview']

        for dsKey in dsi[datasetSqlFilter].fetch('KEY'):
            dsInfo = dsi[dsKey]
            ds = dsInfo.grabDatasets()[0]
            suptitle = dsInfo['dataset_name'][0]
            if keyStateParams['onlySuccessfulTrials']:
                ds, _ = ds.successfulTrials()

            ds.plotKeyStateInfo()
            keyStateFig = plt.gcf()
            keyStateFig.suptitle(suptitle)

        saveFiguresToPdf(pdfname="keyStateOverview", analysisDescription="keyStateOverview")
        plt.close('all')



