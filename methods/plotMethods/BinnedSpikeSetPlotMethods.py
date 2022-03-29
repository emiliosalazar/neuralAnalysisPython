"""
Here I will have methods for plotting binned spike sets, so they can be separate from codey-code
"""
from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path

from setup.DataJointSetup import DatasetInfo, GpfaAnalysisInfo, GpfaAnalysisParams, FilterSpikeSetParams
from methods.GpfaMethods import crunchGpfaResults
from methods.plotMethods.GpfaPlotMethods import visualizeGpfaResults, plotDimensionsOverTrials

def plotResponseOverTime(binnedSpikes, datasetNames, plotTypeInfo, stateName = 'Delay', chPlot = None):
    from methods.BinnedSpikeSetListMethods import generateLabelGroupStatistics as genBSLLabGrp
    plotSegments = plotTypeInfo['plotSegments'] if 'plotSegments' in plotTypeInfo else True
    binsAroundAlign = plotTypeInfo['ptsAroundAlBin'] if 'ptsAroundAlBin' in plotTypeInfo else 4
    plotMethod = plotTypeInfo['plotMethod'] if 'plotMethod' in plotTypeInfo else 'plot'
    
    if chPlot is not None and len(chPlot) != len(binnedSpikes):
        raise Exception("One channel list per binned spike set must be provided if specifying channels to plot")

    for idx, (bnSp, dsName) in enumerate(zip(binnedSpikes, datasetNames)):
        numAlignmentBins = bnSp.alignmentBins[0].shape[0]
        numCols = 3
        numRows = 3
        
        if plotSegments:
            groupedSpikes = []
            grpSpkTrlAvgSem = []
            zeroBins = []
            chanTmAvgs = []
            for seg in range(numAlignmentBins):
                if bnSp.dtype == 'object':
                    alBins = bnSp.alignmentBins[:, seg].astype(int)
                    tpTots = np.stack([bnT[0].shape[0] for bnT in bnSp])
                    segStarts = np.maximum(alBins-binsAroundAlign, 0)
                    segEnds = np.minimum(alBins+binsAroundAlign, tpTots)
                    segLen = segEnds-segStarts
                    if not np.all(segLen==segLen[0]):
                        minSegLen = segLen.min()
                        maxSegLen = segLen.max()
                        if (maxSegLen-minSegLen) > 1:
                            breakpoint() # not sure why there's more than an off-by-one difference
                        else:
                            segEnds = segStarts + minSegLen
                    # NOTE that the enumerate approach is needed in
                    # order to keep labels and the like along...
                    bnSpSeg = np.stack([bnSp[trl,:,sS:sE] for trl, (sS, sE) in enumerate(zip(segStarts, segEnds))])
#                    breakpoint()

                    # NOTE: the trl:trl+1 prevents squashing the first
                    # dimension with timeAverage()...
                    if seg == 0:
                        bnSpSegB4Avg = np.vstack([bnSp[trl:trl+1,:,:aB].timeAverage() for trl, aB in enumerate(alBins)])
                    else:
                        alBinsB4 = bnSp.alignmentBins[:, seg-1].astype(int)
                        bnSpSegB4Avg = np.vstack([bnSp[trl:trl+1,:,aBb4:aB].timeAverage() for trl, (aBb4, aB) in enumerate(zip(alBinsB4, alBins))])

                else:
                    alBins = np.unique(bnSp.alignmentBins, axis=0)[0].astype(int)
                    if alBins.ndim>1 and alBins.shape[0]>1:
                        raise Exception("Do the same as you would for an object dtype here...")
                    else:
                        alB = alBins[seg]
                    segSt = np.maximum(alB-binsAroundAlign, 0) # can't go less than the bss start heh...
                    segEnd = np.minimum(alB+binsAroundAlign, bnSp.shape[2]) # can't go over bss end...
                    bnSpSeg = bnSp[:, :, int(segSt):int(segEnd)]
                    if seg == 0:
                        bnSpSegB4Avg = bnSp[:,:,:int(alB)].timeAverage()
                    else:
                        alBb4 = alBins[seg-1]
                        bnSpSegB4Avg = bnSp[:,:,int(alBb4):int(alB)].timeAverage()

                grpSpkTrlAvgSemHere, groupedSpikesHere, grpLabels = genBSLLabGrp([bnSpSeg], labelUse='stimulusMainLabel')
                grpSpkTrlAvgSem.append(grpSpkTrlAvgSemHere[0])
                groupedSpikes.append(groupedSpikesHere[0])
#                zeroBins.append(binsAroundAlign)
                if seg==0:
                    zeroBins.append(alBins[0])
                else:
                    zeroBins.append(int(np.minimum(alBins[seg], binsAroundAlign)))

                grpSpkTrlAvgSemAll, _, _ = genBSLLabGrp([bnSpSegB4Avg], labelUse='stimulusMainLabel')
                chanTmAvgs.append(grpSpkTrlAvgSemAll[0][0])

            if bnSp.dtype=="object":
                # NOTE: the trl:trl+1 prevents squashing the first
                # dimension...
                bnSpSegAftAvg = np.vstack([bnSp[trl:trl+1,:,aB:].timeAverage() for trl, aB in enumerate(alBins)])
            else:
                bnSpSegAftAvg = bnSp[:,:,int(alB):].timeAverage()
            grpSpkTrlAvgSemAll, _, _ = genBSLLabGrp([bnSpSegAftAvg], labelUse='stimulusMainLabel')
            chanTmAvgs.append(grpSpkTrlAvgSemAll[0][0])
        else:
            # note that this pops out as a list from the function, much
            # like the main if, so we don't need to change anything!
            numAlignmentBins = 1
            zeroBins = bnSp.alignmentBins[0].astype(int)
            grpSpkTrlAvgSem, groupedSpikes, grpLabels = genBSLLabGrp([bnSp], labelUse='stimulusMainLabel')
            chanTmAvgs = []
            if len(grpSpkTrlAvgSem[0]):
                chanTmAvgs.append(grpSpkTrlAvgSem[0][0][:,:,:zeroBins[0]].timeAverage())
                chanTmAvgs.append(grpSpkTrlAvgSem[0][0][:,:,zeroBins[0]:].timeAverage())
            else:
                chanTmAvgs.append(np.array([]))


        # we'll have this be based on the entire period... at least
        # for now; also we time average before splitting into
        # different groups to allow for binned spikes with
        # different length trials to be used...
        grpSpkTrlAvgSemAll, _, _ = genBSLLabGrp([bnSp.timeAverage()], labelUse='stimulusMainLabel')
        chanTmAvgAll = grpSpkTrlAvgSemAll[0][0]

        grpLabels = grpLabels[0]
        if chPlot is not None:
            chans = chPlot[idx]
        else:
            chans = np.arange(bnSp.shape[1])
        binSizeMs = bnSp.binSize

        grpLabels = grpLabels.astype('float64')
        if grpLabels.astype('float64').max()>2*np.pi:
            sbpltAngs = np.arange(135, -225, -45)
            modulusFullRot = 360
            grpLblPlotFactor = np.pi/180
        else:
            sbpltAngs = np.arange(3*np.pi/4, -5*np.pi/4, -np.pi/4)
            modulusFullRot = 2*np.pi
            grpLblPlotFactor = 1

        figTuneRaw = plt.figure()
        figTuneRaw.tight_layout()
        figTuneRaw.suptitle(dsName)
        tuningCurveParams = bnSp.computeCosTuningCurves()
        tuningCurves = tuningCurveParams['tuningCurves'].T
        tuningCurveAngs = tuningCurveParams['tuningCurveAngs']
        axTun = []
        pltNum=0
        totPltRow = 2
        pltNum+=1
        axTun.append(plt.subplot(totPltRow, 1, pltNum))
        axTun[-1].set_ylabel('FR (Hz)')
        axTun[-1].set_xlim([0, 2*np.pi])
        axTun[-1].set_xticklabels('')
        axTun[-1].xaxis.set_visible(False)
        axTun[-1].spines['bottom'].set_visible(False)
        axTun[-1].spines['top'].set_visible(False)
        axTun[-1].spines['right'].set_visible(False)
        pltAngsUnsort = ((sbpltAngs+180)*np.pi/180) % (2*np.pi)
        srtInds = np.argsort(pltAngsUnsort)
        pltAngs = pltAngsUnsort[srtInds][::-1]
        pltAngs = np.hstack([2*np.pi, pltAngs])
        chanTmAvgPlt = chanTmAvgAll[np.hstack([srtInds[-1],srtInds])]
        axTun[-1].plot(pltAngs, chanTmAvgPlt)
        axTun[-1].set_title('raw data')
        pltNum+=1
        axTun.append(plt.subplot(totPltRow, 1, pltNum))
        axTun[-1].set_ylabel('FR (Hz)')
        axTun[-1].set_xlim([0, 2*np.pi])
        axTun[-1].set_xticklabels('')
        axTun[-1].xaxis.set_visible(False)
        axTun[-1].spines['bottom'].set_visible(False)
        axTun[-1].spines['top'].set_visible(False)
        axTun[-1].spines['right'].set_visible(False)
        axTun[-1].plot(tuningCurveAngs, tuningCurves)
        axTun[-1].set_title('raw tuning curves')
        figTuneRaw.tight_layout()

        figTuneNorm = plt.figure()
        figTuneNorm.tight_layout()
        figTuneNorm.suptitle(dsName)
        pltNum = 0
        axTun = []
        totPltRow = 2
        pltNum+=1
        axTun.append(plt.subplot(totPltRow, 1, pltNum))
        axTun[-1].set_ylabel('$\Delta$FR from baseline (Hz)')
        axTun[-1].set_xlim([0, 2*np.pi])
        axTun[-1].set_xticklabels('')
        axTun[-1].xaxis.set_visible(False)
        axTun[-1].spines['bottom'].set_visible(False)
        axTun[-1].spines['top'].set_visible(False)
        axTun[-1].spines['right'].set_visible(False)
        tuningCurveBaselines = tuningCurveParams['bslnPerChan']
        axTun[-1].plot(tuningCurveAngs, tuningCurves-tuningCurveBaselines)
        axTun[-1].set_title('tuning curves, baseline subtracted')
        pltNum+=1
        axTun.append(plt.subplot(totPltRow, 1, pltNum))
        tuningCurveModulations = tuningCurveParams['modPerChan']
        axTun[-1].plot(tuningCurveAngs, (tuningCurves-tuningCurveBaselines)/tuningCurveModulations)
        axTun[-1].set_title('tuning curves, normalized modulation')
        axTun[-1].set_xlabel('angle (degrees)')
        axTun[-1].set_xlim([0, 2*np.pi])
        axTun[-1].set_xticks(np.linspace(0, 2*np.pi, 9))
        axTun[-1].set_xticklabels(np.linspace(0, 360, 9, dtype=int))
        axTun[-1].spines['top'].set_visible(False)
        axTun[-1].spines['right'].set_visible(False)
        figTuneNorm.tight_layout()

        # Add nan to represent center for tuning polar curve...
        sbpltAngs = np.concatenate((sbpltAngs[0:3], sbpltAngs[[-1]], np.expand_dims(np.asarray(np.nan),axis=0), sbpltAngs[[3]], np.flip(sbpltAngs[4:-1])), axis=0)


        for chan in chans:
            # Prep figure
            plt.figure()
            plt.suptitle(dsName + ': channel ' + str(chan))
            axMn = None
            axVals = np.empty((0,4))
            axes = []

            for alB in range(numAlignmentBins):
                # Prep parameters to plot
                grpSpkTrlAvgSemHere = grpSpkTrlAvgSem[alB]
                if plotMethod == 'plot':
                    chanRespMean = np.squeeze(grpSpkTrlAvgSemHere[0][:,[chan]])
                    chanRespSem = np.squeeze(grpSpkTrlAvgSemHere[1][:,[chan]])

                    if chanRespMean.ndim == 1: 
                        # the squeeze can get rid of the condition dimension if
                        # there was only one condition...
                        chanRespMean = chanRespMean[None, :]
                        chanRespSem = chanRespSem[None, :]


                colSt = bnSp.colorset[alB, :]
                colEnd = bnSp.colorset[alB+1, :]

                for idx in range(0, len(grpLabels)):
                    chanSpkBinsByTrial = groupedSpikes[alB][idx][:,chan]
                    if grpLabels.shape[1]==1:
                        subplotChooseCond = np.where([np.allclose(grpLabels.astype('float64')[idx] % (modulusFullRot), sbpltAngs[i] % (modulusFullRot)) for i in range(0,len(sbpltAngs))])[0]
                    else:
                        grpLocs = np.arange(modulusFullRot, step=modulusFullRot/grpLabels.shape[0])
                        subplotChooseCond = np.where([np.allclose(grpLocs.astype('float64')[idx] % (modulusFullRot), sbpltAngs[i] % (modulusFullRot)) for i in range(0,len(sbpltAngs))])[0]

                    if not subplotChooseCond.size:
                        subplotChooseCond = np.where(sbpltAngs==modulusFullRot/2*(grpLabels[idx]-2))[0]
                    subplotChoose = subplotChooseCond[0]*numAlignmentBins+alB
                    numColsAll = numCols*numAlignmentBins if plotSegments else 3
                    axes.append(plt.subplot(numRows, numColsAll, subplotChoose+1))
                    

                    zeroBin = zeroBins[alB]
                    if plotMethod == 'plot':
                        numTp = chanRespMean.shape[1]
    #                        tmValStart = np.arange(timeBeforeAndAfterStart[0]+binSizeMs/2, timeBeforeAndAfterStart[1]+binSizeMs/2, binSizeMs)
                        binAll = np.arange(numTp)
                        tmVals = (binAll - zeroBin)*binSizeMs
                        # plot pre-alignment bin
                        plt.plot(tmVals[:zeroBin+1], chanRespMean[idx, :zeroBin+1], color=colSt)
                        plt.fill_between(tmVals[:zeroBin+1], chanRespMean[idx, :zeroBin+1]-chanRespSem[idx,:zeroBin+1], chanRespMean[idx, :zeroBin+1]+chanRespSem[idx,:zeroBin+1], alpha=0.2, color=colSt)

                        # plot post-alignment bin
                        plt.plot(tmVals[zeroBin:], chanRespMean[idx, zeroBin:], color=colEnd)
                        plt.fill_between(tmVals[zeroBin:], chanRespMean[idx, zeroBin:]-chanRespSem[idx,zeroBin:], chanRespMean[idx, zeroBin:]+chanRespSem[idx,zeroBin:], alpha=0.2, color=colEnd)
                    elif plotMethod == 'eventplot':
                        allAlBins = chanSpkBinsByTrial.alignmentBins
                        # plot pre-alignment bin
                        chPltEvts = [np.where(ch==1)[0]-zeroBin for ch in chanSpkBinsByTrial[:, :zeroBin+1]]
                        plt.eventplot(chPltEvts, color=colSt)

                        # plot post-alignment bin
                        if numAlignmentBins == 1 and allAlBins.shape[1] > 1:
                            for colNm, (alBinNumFirst, alBinNumNext) in enumerate(zip(allAlBins[:,:-1].T, allAlBins[:,1:].T)):
                                colCurr = bnSp.colorset[colNm+1, :]
                                chPltEvts = [np.where(ch[int(alFst):int(alNxt)+1]==1)[0]+int(alFst)-zeroBin for ch, alFst, alNxt in zip(chanSpkBinsByTrial[:, :], alBinNumFirst, alBinNumNext)]
                                plt.eventplot(chPltEvts, color=colCurr)
                            colCurr = bnSp.colorset[colNm+2, :]
                            chPltEvts = [np.where(ch[int(alNxt):]==1)[0] + int(alNxt) - zeroBin for ch, alNxt in zip(chanSpkBinsByTrial[:, :], alBinNumNext)]
                            plt.eventplot(chPltEvts, color=colCurr)
                        else:
                            chPltEvts = [np.where(ch==1)[0] for ch in chanSpkBinsByTrial[:, zeroBin:]]
                            plt.eventplot(chPltEvts, color=colEnd)


                    axVals = np.append(axVals, np.array(plt.axis())[None, :], axis=0)

                    ax = axes[-1]
                    if alB != 0:
                        ax.yaxis.set_visible(False)
                        ax.spines['left'].set_visible(False)
                    else:
                        ax.tick_params(direction='in')

                    if subplotChooseCond[0] == 0 and alB == 0:
                        ax.set_ylabel(bnSp.units)
                    elif subplotChooseCond[0] == (numCols*numRows - np.ceil(numCols/2)) and alB == np.ceil(numAlignmentBins/2):
                        ax.set_xlabel('Time Around %s (ms)' % stateName)

                    if subplotChooseCond[0] % numCols:
                        ax.yaxis.set_ticklabels('')
                    else:
                        ax.tick_params(direction='inout')

                    if subplotChooseCond[0] < (numCols*numRows - numCols):
                        ax.xaxis.set_ticklabels('')
                        ax.xaxis.set_visible(False)
                        ax.spines['bottom'].set_visible(False)

                # plot the average before the bin
                if axMn is None:
                    axMn = plt.subplot(numRows, numCols, np.where(np.isnan(sbpltAngs))[0][0]+1, projection='polar')
                if chanTmAvgs[alB].size>1:
                    ptch = axMn.fill(grpLabels*grpLblPlotFactor, chanTmAvgs[alB][:,[chan]])
                    ptch[0].set_fill(False)
                    ptch[0].set_edgecolor(colSt)

            # plot the average after the last bin
            # plt.subplot(numRows, numCols, np.where(np.isnan(sbpltAngs))[0][0]+1, projection='polar')
            if chanTmAvgs[alB+1].size>1:
                ptch = axMn.fill(grpLabels*grpLblPlotFactor, chanTmAvgs[alB+1][:,[chan]])
                ptch[0].set_fill(False)
                ptch[0].set_edgecolor(colEnd)

            ymin = np.min(np.append(0, np.min(axVals, axis=0)[2]))
            ymax = np.max(axVals, axis=0)[3]
            for ax in axes:
                ax.set_ylim(bottom = ymin, top = ymax )
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                plt.axes(ax)
                plt.axvline(x=0, linestyle='--')

            # plot the overall average
            # plt.subplot(numRows, numCols, np.where(np.isnan(sbpltAngs))[0][0]+1, projection='polar')
            if chanTmAvgAll.size>1:
                ptch = axMn.fill(grpLabels*grpLblPlotFactor, chanTmAvgAll[:,[chan]])
                ptch[0].set_fill(False)
                ptch[0].set_edgecolor('k')


# def plotGpfaResults(bssExp, gpfaRes, useFa, crossvalidateNumFolds = 4, timeBeforeAndAfterStart = None, timeBeforeAndAfterEnd = None):
def plotGpfaResults(gpfaResAll, descriptionsAll, brainAreas, timeBeforeAndAfterStart = None, timeBeforeAndAfterEnd = None, bssExp = None):
    dsi = DatasetInfo()
    fsp = FilterSpikeSetParams()
    gai = GpfaAnalysisInfo()
    gap = GpfaAnalysisParams()

    plotInfo = {}
    # might have to go back on these lines for how to define ncols...
    gpPathPer = [[Path(gpH[0]).parent for gpH in gp.keys()] for gp in gpfaResAll]
    gpUnPathNum = [len(np.unique(gpP)) for gpP in gpPathPer]
    ncols = np.max(gpUnPathNum) # len(bssExp)
    # sorry for the hard code
    # brainAreas = np.unique(dsi[bssExp]['brain_area'])
    dimArBest = [[],[],[]]
    ncols = len(brainAreas)
    # axs = figErr.subplots(nrows=3, ncols=ncols, squeeze=False, constrained_layout=True)
    figErr, axs = plt.subplots(nrows=3, ncols=ncols, squeeze=False, constrained_layout=True)
    figErr.suptitle('dimensionality vs (GP)FA log likelihood/cumulative/individual variance accounted for')
    
    if bssExp is not None:
        dsParents = [dsi[bE] for bE in bssExp]
        trlUsedForGp = [bE.trialAndChannelFilterFromParent(ds)[0] for bE,ds in zip(bssExp, dsParents)]

    # gpfaResAll = []
    # gpfaInfoAll = []
    # for subExp in bssExp:
    #     gpfaResH, gpfaInfoH = gai[subExp][gap['num_folds_crossvalidate={}'.format(crossvalidateNumFolds)]].grabGpfaResults(returnInfo=True, useFa=useFa)
        
    #     gpfaResAll.append(gpfaResH)
    #     gpfaInfoAll.append(gpfaInfoH)
    
    cvApproach = "logLikelihood"
    shCovThresh = 0.95
    lblLLErr = 'LL err over folds'
    lblLL= 'LL mean over folds'
    crossValsUse = [0]

    for gpResInd, (gpfaResHere, descriptions) in enumerate(zip(gpfaResAll, descriptionsAll)):
        gpfaCrunchedResults = crunchGpfaResults(gpfaResHere, cvApproach = cvApproach, shCovThresh = shCovThresh)
        
        dimExpAllTest = [d['xDimBestAll'] for _, d in gpfaCrunchedResults.items()]
        normScoreAllTest = [d['normalGpfaScoreAll'] for _, d in gpfaCrunchedResults.items()]
        gpfaOutDimAllTest = [d['dimResults'] for _, d in gpfaCrunchedResults.items()]
        gpfaTestIndsOutAllTest = [d['testInds'] for _, d in gpfaCrunchedResults.items()]
        gpfaBinSizeAllTest = [d['binSize'] for _, d in gpfaCrunchedResults.items()]
        gpfaCondLabelsAllTest = [d['condLabel'] for _, d in gpfaCrunchedResults.items()]
        gpfaAlignmentBinsAllTest = [d['alignmentBins'] for _, d in gpfaCrunchedResults.items()]
        gpfaDimsTestedAllTest = [d['dimsTest'] for _, d in gpfaCrunchedResults.items()]

        if bssExp is not None:
            # this is going to be used to understand what trials were used/what their timing was
            spikesForGpfa =  bssExp[gpResInd].grabBinnedSpikes()[0].convertUnitsTo('count')
            dsParent = dsParents[gpResInd].grabDataset()
            trlFromDs = trlUsedForGp[gpResInd]
            spikeBinStartInTrial = bssExp[gpResInd]['start_time'][0]

            spikesNorm = ((spikesForGpfa - spikesForGpfa.mean(axis=0))/spikesForGpfa.std(axis=0)).squeeze()

        for idx, (description, dimsTest, testIndsCondAll, dimResult, normalGpfaScoreAll, binSize, condLabels, alignmentBins) in enumerate(zip(descriptions, gpfaDimsTestedAllTest, gpfaTestIndsOutAllTest, gpfaOutDimAllTest, normScoreAllTest, gpfaBinSizeAllTest, gpfaCondLabelsAllTest, gpfaAlignmentBinsAllTest)):

            if bssExp is not None:
                unLab = np.unique(condLabels[0], axis = 0)
                if unLab.shape[0]>1:
                    # NOTE: this is for when conditions are combined; think
                    # it'll be easy to do just haven't written the code yet
                    labelUse = 'stimulusMainLabel'
                    trialsInBssWithLabel = np.hstack([(np.all(spikesForGpfa.labels[labelUse] == uL, axis=1)).nonzero()[0] for uL in unLab])
                    # NOTE: you do NOT want to sort trialsInBssWithLabel. The
                    # simplest way to give the reason is that before FA/GPFA was
                    # run, when all the conditions were combined, the trials for
                    # each condition were first grouped together. This means
                    # that the trial index refers to the index of the trial not
                    # ordered by when it was presented in the session, but
                    # ordered first by the condition, and then *within* the
                    # condition by when it was presented. By not sorting
                    # trialsInBssWithLabel, I effectively replicate that
                    # ordering here, so now testInds below is referring to the
                    # same trials as indexed by trialsInBssWithLabel
                    # trialsInBssWithLabel.sort()
                    trialsForLabel = trlFromDs[trialsInBssWithLabel]
                    trlTimes = dsParent.trialStartTimeInSession()[trialsForLabel] 
                    strtTimesInTrial = spikeBinStartInTrial[trialsInBssWithLabel]
                    spikeStartTimesInSession = trlTimes + strtTimesInTrial
                else:
                    # note that eventually I might want to be flexible in using
                    # another label... but for the moment this is hardcoded
                    labelUse = 'stimulusMainLabel'
                    trialsInBssWithLabel = np.all(spikesForGpfa.labels[labelUse] == unLab, axis=1)
                    trialsForLabel = trlFromDs[trialsInBssWithLabel]
                    trlTimes = dsParent.trialStartTimeInSession()[trialsForLabel] 
                    strtTimesInTrial = spikeBinStartInTrial[trialsInBssWithLabel]
                    spikeStartTimesInSession = trlTimes + strtTimesInTrial


            # grab first cval--they'll be the same for each cval, which is what
            # these lists store
            dimTest = dimsTest[0]
            binSize = binSize[0]
            testInds = testIndsCondAll[0]

            axChs = np.array([description.find(bA)!=-1 for bA in brainAreas])
            dimArBest[axChs.nonzero()[0][0]] += dimExpAllTest
            axScore = axs[0][axChs][0]
            axScore.set_title(np.array(brainAreas)[axChs][0])
            axCumulDim = axs[1][axChs][0]
            axByDim = axs[2][axChs][0]
            plotInfo['axScore'] = axScore
            plotInfo['axCumulDim'] = axCumulDim
            plotInfo['axByDim'] = axByDim
            # breakpoint()

            if timeBeforeAndAfterStart is not None:
                tmValsStartBest = np.arange(-timeBeforeAndAfterStart[0], timeBeforeAndAfterStart[1], binSize)
            else:
                tmValsStartBest = np.ndarray((0,0))
                
            if timeBeforeAndAfterEnd is not None:
                tmValsEndBest = np.arange(-timeBeforeAndAfterEnd[0], timeBeforeAndAfterEnd[1], binSize)  
            else:
                tmValsEndBest = np.ndarray((0,0))

            plotInfo['lblLL'] = lblLL
            plotInfo['lblLLErr'] = lblLLErr
            plotInfo['description'] = description
            tmVals = [tmValsStartBest, tmValsEndBest]
            condLabelsInt = [idx]
            visualizeGpfaResults(plotInfo, dimResult,  tmVals, cvApproach, normalGpfaScoreAll, dimTest, testInds, shCovThresh, binSize, condLabelsInt, alignmentBins, crossValsUse, trlTimes=spikeStartTimesInSession, spikesUsedNorm = spikesNorm[trialsInBssWithLabel])
    
    for axCumul, axByDim, dmB in zip(axs[1], axs[2], dimArBest):
        axCumul.set_xlim(1, np.max(dmB))
        axByDim.set_xlim(1, np.max(dmB))
        axCumul.set_ylim(0, 1.05)
        axByDim.set_ylim(0, 1.05)

def plotSlowDriftVsFaSpace(spikesUsedNorm, trlInds, trlTimes, binSize, pcSlowDrift, pcs, evalsPca, description, condLabel, dimResult):

    # plot dimension over trials!
    figOverTime = plt.figure()
    figOverTime.suptitle(description)# + " cond " + str(condLabel) + "")

    figOverTimePC = plt.figure()
    figOverTimePC.suptitle(description)#+ " cond " + str(condLabel) + "")

    pltNum = 1
    rowsPlt = 2
    axesOverTime = []
    axesOverTimePC = []
    axVals = np.empty((0,4))
    
    dimsComp = list(dimResult.keys())
    if len(dimsComp) > 1:
        breakpoint() # this only works if one dim has been computed! The best hopefully
    else:
        xDimScoreBest = dimsComp[0]
    seqTestUse = dimResult[xDimScoreBest]['seqsTestNew']
    if len(seqTestUse) > 1:
        print('only using the first crossvalidation of the GPFA/FA result for slow drift comparison!')
    cValUse = 0
    seqTestUse = seqTestUse[cValUse]
    allSeqsTog = [sq['xorth'] for sq in seqTestUse]
    # else:
    #     allSeqsTog = [sq['xsm'] for sq in seqTestUse]

    allSeqsByDim = list(zip(*allSeqsTog))
    colsOverTimePlt = np.ceil(len(allSeqsByDim) / rowsPlt)
    C = dimResult[xDimScoreBest]['allEstParams'][cValUse]['C']
    Corth, egsOrth, _ = np.linalg.svd(C)
    Corth = Corth[:, :egsOrth.size]
    CorthScaled = Corth @ np.diag(egsOrth)
    R = dimResult[xDimScoreBest]['allEstParams'][cValUse]['R']
    svPerLatent = [np.mean(np.diag(C[:,None] @ C[None,:]) / np.diag(CorthScaled @ CorthScaled.T + R)) for C in CorthScaled.T]

    # calculated dshared...
    shEigs = egsOrth**2
    percAcc = np.cumsum(shEigs)/np.sum(shEigs)
    shCovThresh = 0.95
    if percAcc.size>0:
        xDimBest = np.where(percAcc>shCovThresh)[0][0]+1
    else:
        xDimBest = 0

    latentVarExpBySlowDrift = []
    
    for dimNum, dimSeqs in enumerate(allSeqsByDim):
        trajTms = np.hstack(trlTimes[trlInds])
        trajAll = np.hstack(dimSeqs)
        intercept = np.ones_like(trajTms)
        coeffs = np.vstack([trajTms,intercept]).T
        modelParams, resid = np.linalg.lstsq(coeffs, trajAll, rcond=None)[:2]
        slp,intcpt = modelParams
        linearFit = slp*np.sort(trajTms) + intcpt
        r2latent = 1-resid/(trajAll.var()*trajAll.size)
        latentSv = svPerLatent[dimNum]
        try:
            trajReprojToSlowDrift = trajAll[:,None] @ Corth[:,[dimNum]].T @ pcSlowDrift
            varInSlowDriftDim = trajReprojToSlowDrift.var(ddof=0)
            varInShLatent = trajAll.var(ddof=0)
            varExpBySlowDrift = varInSlowDriftDim/varInShLatent
        except ValueError:
            breakpoint()
            varExpBySlowDrift = -0.01
        plotAnnot = {
            '%sv' : 100*latentSv,
            'R^2' : r2latent[0],
            '%latent var exp by slow drift pc' : 100*varExpBySlowDrift,
        }
        latentVarExpBySlowDrift.append(100*varExpBySlowDrift)
        colorUse = [0.5,0.5,0.5]

        dimNum = dimNum+1 # for the plot, 1 is 1-dimensional, not zero-dimensinoal
        axUsed = plotDimensionsOverTrials(figOverTime, pltNum, axesOverTime, rowsPlt, colsOverTimePlt, dimSeqs, trlInds, trlTimes, binSize, dimNum, xDimBest, colorUse, linearFit = linearFit, plotAnnot = plotAnnot, linewidth=0.2, alpha = 0.5)
        axVals = np.append(axVals, np.array(axUsed.axis())[None, :], axis=0)
        pltNum += 1

    axesOverTimePC = []


    numPCUse = 4
    colsOverTimePltPC = np.ceil(numPCUse/rowsPlt)
    spikesProjNorm = [pcs.T @ sp[:, None]  for sp in spikesUsedNorm[trlInds]]
    spikesProjNormByDim = list(zip(*spikesProjNorm))
    dimsExpVar = evalsPca/evalsPca.sum()
    totalVarTrls = np.cov(spikesUsedNorm.T, ddof=0).trace()
    dimsExpVarTrls = np.array([np.array(spProj).var() for spProj in spikesProjNormByDim])/totalVarTrls


    pltNum = 1
    trajTms = np.hstack(trlTimes)
    intercept = np.ones_like(trajTms)
    coeffs = np.vstack([trajTms,intercept]).T
    numOverSessPcPlot = 4
    slowDriftVarExpTrials = 100*dimsExpVarTrls[0]
    slowDriftVarExpRA = 100*dimsExpVar[0]
    for dimNum, dimVals in enumerate(spikesProjNormByDim[:numOverSessPcPlot]):
        trajAll = np.hstack(dimVals)
        # linear fit
        modelParams, resid = np.linalg.lstsq(coeffs, trajAll, rcond=None)[:2]
        slp,intcpt = modelParams
        linearFit = slp*np.sort(trajTms) + intcpt
        # r^2 computation
        r2latent = 1-resid/(trajAll.var()*trajAll.size)
        # grab explained variance
        dimExpVar = dimsExpVar[dimNum]
        dimExpVarTrls = dimsExpVarTrls[dimNum]
        # project to shared space to see explained variance
        try:
            trajInHighDim = trajAll[:,None] @ pcs[:, [dimNum]].T
            trajInShSpace = trajInHighDim @ Corth
            varInPC = np.var(trajAll, axis=0, ddof=0)
            varInShSpace = np.var(trajInShSpace, axis=0, ddof=0).sum()
            varExpByShSpace = varInShSpace/varInPC
            if dimNum == 0:
                slowDriftVarExpSharedSpace = 100*varExpByShSpace
        except ValueError:
            breakpoint()
            varExpByShSpace = -0.01
        plotAnnot = {
            '%ev of rolling avg' : 100*dimExpVar,
            '%ev of ind trials' : 100*dimExpVarTrls,
            '%pc var exp by sh space' : 100*varExpByShSpace,
            'R^2' : r2latent[0],
        }
        colorUse = [0.5,0.5,0.5]

        dimNum = dimNum+1 # for the plot, 1 is 1-dimensional, not zero-dimensinoal
        axUsed = plotDimensionsOverTrials(figOverTimePC, pltNum, axesOverTimePC, rowsPlt, colsOverTimePltPC, dimVals, trlInds, trlTimes, binSize, dimNum, xDimBest, colorUse, linearFit = linearFit, plotAnnot = plotAnnot, linewidth=0.2, alpha = 0.5)
        axVals = np.append(axVals, np.array(axUsed.axis())[None, :], axis=0)
        pltNum += 1
    # figOverTimePC.tight_layout()

    if axVals.size:
        ymin = np.min(np.append(0, np.min(axVals, axis=0)[2]))
        ymax = np.max(axVals, axis=0)[3]

        for ax in axesOverTime:
            from matplotlib import text
            annot = [child for child in ax.get_children() if isinstance(child, text.Text) and child.get_text().find('=') != -1]
            annot[0].set_y(ymax)

            ax.set_ylim(bottom = ymin, top = ymax )
            ax.axvline(x=0, linestyle=':', color='black')
            ax.axhline(y=0, linestyle=':', color='black')
#                        xl = ax.get_xlim()
#                        yl = ax.get_ylim()
#                        ax.axhline(y=0,xmin=xl[0],xmax=xl[1], linestyle = ':', color='black')
#                        ax.axvline(x=0,ymin=xl[0],ymax=xl[1], linestyle = ':', color='black')
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)

        for ax in axesOverTimePC:
            from matplotlib import text
            annot = [child for child in ax.get_children() if isinstance(child, text.Text) and child.get_text().find('=') != -1]
            annot[0].set_y(ymax)

            ax.set_ylim(bottom = ymin, top = ymax )
            ax.axvline(x=0, linestyle=':', color='black')
            ax.axhline(y=0, linestyle=':', color='black')
#                        xl = ax.get_xlim()
#                        yl = ax.get_ylim()
#                        ax.axhline(y=0,xmin=xl[0],xmax=xl[1], linestyle = ':', color='black')
#                        ax.axvline(x=0,ymin=xl[0],ymax=xl[1], linestyle = ':', color='black')
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
    
    return latentVarExpBySlowDrift, slowDriftVarExpTrials, slowDriftVarExpRA, slowDriftVarExpSharedSpace



