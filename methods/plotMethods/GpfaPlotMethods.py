#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 17:05:23 2020

@author: emilio
"""
from matplotlib import pyplot as plt
import numpy as np
from matplotlib.axis import Axis as AxisObj

    
# Note that indsUse can include a None if we want to plot to the end (or start) of a given dimension
def plotDimensionsAgainstEachOther(axUse, dimensionVals, tmVals, tpIndUse, lblStTraj, lblEndTraj, lblsAlign, colorset, idx, axTtl, axLabel, colorAlignPts):
    
    valsPlot = dimensionVals[:, tpIndUse[0]:tpIndUse[1]]
    axUse.plot(*valsPlot, color = colorset[idx,:], linewidth = 0.4, marker='.' if valsPlot.shape[1] == 1 else None)
    
    # axUse.plot(sq['xorth'][0,:tmValsStart.shape[0]], sq['xorth'][1,:tmValsStart.shape[0]], sq['xorth'][2,:tmValsStart.shape[0]],
               # color=colorset[idx,:], linewidth=0.4)
               
    # no need to mark beginning and end if we don't have a trajectory...
    if valsPlot.shape[1] > 1:
        valsPlot = dimensionVals[:, tpIndUse[0], None]
        axUse.plot(*valsPlot, 'o', color = colorset[idx,:], linewidth=0.4, label=lblStTraj, markeredgecolor='black')
                            
        valsPlot = dimensionVals[:, tpIndUse[1]-1, None]
        axUse.plot(*valsPlot, '>', color = colorset[idx,:], linewidth=0.4, label=lblEndTraj, markeredgecolor='black')
        
    # marking the alignment point here
    if type(tmVals) is not list:
        tmVals = [tmVals]
        lblsAlign = [lblsAlign]
        colorAlignPts = [colorAlignPts]
        
    for tmV, lbAl, colAlPt in zip(tmVals, lblsAlign, colorAlignPts):
        if np.min(tmVals) <= 0 and np.max(tmVals) >= 0:
            # alignX = np.interp(0, tmVals, sq['xorth'][0,tpIndUse[0]:tpIndUse[1]])
            # alignY = np.interp(0, tmVals, sq['xorth'][1,tpIndUse[0]:tpIndUse[1]])
            # alignZ = np.interp(0, tmVals, sq['xorth'][2,tpIndUse[0]:tpIndUse[1]])
            
#            alignPt = [np.interp(0,tmV, dimVal[dm,tpIndUse[0]:tpIndUse[1]]) for dm, dimVal in enumerate(dimensionVals)]
            alignPt = [[np.interp(0,tmV, dimVal[tpIndUse[0]:tpIndUse[1]])] for dimVal in dimensionVals]
            axUse.plot(*alignPt, '*', color=colAlPt, label = lbAl[0])
        else:
            axUse.plot([np.nan], [np.nan], '*', color=colAlPt, label = lbAl[1])
    
    axUse.set_title(axTtl)
    
    axisObj = [aObj for aObj in axUse.get_children() if isinstance(aObj, AxisObj)]
    axObjName = [aObj.axis_name for aObj in axisObj]
    sortedAx = np.argsort(axObjName)
    [sA.set_label_text(aL) for sA, aL in zip(np.array(axisObj)[sortedAx], axLabel)]
    # axUse.set_xlabel('gpfa 1')
    # axUse.set_ylabel('gpfa 2')
    # axUse.set_zlabel('gpfa 3')
    if lblStTraj is not None and lblEndTraj is not None and lbAl[0] is not None and lbAl[1] is not None:
        axUse.legend()

# the biggest chunk of this function is just finding which plot to plot into...
def plotDimensionsAgainstTime(figSep, pltNum, pltListNum, axesPlotList, axesOtherList, rowsPlt, colsPlt, tmValsPlot, dimVals, dimNum, xDimBest, colorUse, labelTraj = "trajectory", linewidth=0.4, alpha = 1, axTitle="AlPoint"):
    if len(axesPlotList) + len(axesOtherList) < rowsPlt*colsPlt:
        axesHere = figSep.add_subplot(rowsPlt, colsPlt, pltNum)
        axesPlotList.append(axesHere)
        if pltNum == 1: 
            axesHere.set_title("dim " + str(dimNum) + " peri" + axTitle)
        else:
            axesHere.set_title("d"+str(dimNum)+axTitle[0])
    
        if pltNum <= (xDimBest - colsPlt):
            axesHere.set_xticklabels('')
            axesHere.xaxis.set_visible(False)
            axesHere.spines['bottom'].set_visible(False)
        else:  
            axesHere.set_xlabel('time (ms)')
    
        if (pltNum % colsPlt) != 1 and colsPlt != 1:
            axesHere.set_yticklabels('')
            axesHere.yaxis.set_visible(False)
            axesHere.spines['left'].set_visible(False)
    else:        
        axesHere = axesPlotList[pltListNum]    
        plt.axes(axesHere)
    
    plt.plot(tmValsPlot, dimVals, color=colorUse, linewidth=linewidth, label=labelTraj, marker='.' if tmValsPlot.size==1 else '')
    
    return axesHere

def plotDimensionsOverTrials(figOverTime, pltNum, axesOverTime, rowsPlt, colsPlt, allDimTraj, trlInds, startTimesInSession, binSize, dimNum, xDimBest, colorUse, linearFit=None, plotAnnot=None, labelTraj = "trajectory", linewidth=0.4, alpha = 1):

    if len(axesOverTime) < rowsPlt*colsPlt:
        axesHere = figOverTime.add_subplot(rowsPlt, colsPlt, pltNum)

        if pltNum == 1: 
            axesHere.set_title("dim " + str(dimNum) + " over session" )
            axesHere.set_ylabel("dim score")
        else:
            axesHere.set_title("d"+str(dimNum))

        if pltNum <= (xDimBest - colsPlt):
            axesHere.set_xticklabels('')
            axesHere.xaxis.set_visible(False)
            axesHere.spines['bottom'].set_visible(False)
        else:  
            axesHere.set_xlabel('time (ms)')
    
        if (pltNum % colsPlt) != 1 and colsPlt != 1:
            axesHere.set_yticklabels('')
            axesHere.yaxis.set_visible(False)
            axesHere.spines['left'].set_visible(False)
        
        axesHere.spines['right'].set_visible(False)
        axesHere.spines['top'].set_visible(False)
    else:
        axesHere = axesOverTime[pltNum]

    axesOverTime.append(axesHere)


    if plotAnnot is not None:
        annotList = ['{} = {:0.2f}'.format(k,v) for k, v in plotAnnot.items()]
        annotStr = '\n'.join(annotList)

    # meanTrajLen = np.mean([traj.shape[0] for traj in allDimTraj])
    # sortingInds = np.argsort(trlInds)
    for traj, trlInd in zip(allDimTraj, trlInds):
    # for indSorted in sortingInds:
    #     traj = allDimTraj[indSorted]
    #     trlInd = trlInds[indSorted]
        # this accounts for trials that aren't plotted (i.e. because they were
        # training trials) between the current trial and the previous plotted
        # trials: if sequential trials were computed, then this will be zero and
        # nothing will shift
        tmptSt = startTimesInSession[trlInd] + binSize/2
        tmptUse = np.arange(tmptSt, tmptSt + len(traj)*binSize, binSize)
        if traj.size == 1:
            marker = '.'
        else:
            marker = None
        axesHere.plot(tmptUse, traj, color='k', marker = marker)

    if plotAnnot is not None:
        xlms = axesHere.get_xlim()
        ylms = axesHere.get_ylim()
        axesHere.text(xlms[-1], ylms[-1], annotStr, ha='right', va='top', fontsize='x-small')

    if linearFit is not None:
        startTmsTog = np.sort(np.hstack(startTimesInSession[trlInds]))
        axesHere.plot(startTmsTog, linearFit, linestyle='--', color='r')

        return axesHere
   

def visualizeGpfaResults(plotInfo, dimResults, tmVals, cvApproach, gpfaScoreAll, xDimTest, testInds, shCovThresh, crossValsUse, binSize, condLabels, alignmentBins,
            baselineSubtract = False, # setting to false default because I think I want to get rid of it entirely...
            computeShuffles = False, sqrtSpikes = False):

   from classes.BinnedSpikeSet import BinnedSpikeSet
   colorset = BinnedSpikeSet.colorset

   for idx, dimResult in enumerate(dimResults):
        print("** Plotting GPFA for condition %d/%d **" % (idx+1, len(dimResults)))
        gpfaScore = gpfaScoreAll[idx]
        # normalizedGpfaScore = (gpfaScore - np.min(gpfaScore, axis=0))/(np.max(gpfaScore,axis=0)-np.min(gpfaScore,axis=0))

        # breakpoint()
        gpfaScoreMn = gpfaScore.mean(axis=1)
        gpfaScoreErr = gpfaScore.std(axis=1)

        if cvApproach is "logLikelihood":
            xDimScoreBest = xDimTest[np.argmax(gpfaScoreMn)]
        elif cvApproach is "squaredError":
            xDimScoreBest = xDimTest[np.argmin(gpfaScoreMn)]
        
        if xDimScoreBest == 0:
            continue
        
        Cparams = [prm['C'] for prm in dimResult[xDimScoreBest]['allEstParams']]
        shEigs = [np.flip(np.sort(np.linalg.eig(C.T @ C)[0])) for C in Cparams]
        percAcc = np.stack([np.cumsum(eVals)/np.sum(eVals) for eVals in shEigs])
        meanPercAcc = np.mean(percAcc, axis=0)
        stdPercAcc = np.std(percAcc, axis = 0)
        accByDim = np.stack([eVals/np.sum(eVals) for eVals in shEigs])
        meanAccByDim = np.mean(accByDim, axis=0)
        stdAccByDim = np.std(accByDim, axis = 0)
        xDimBest = np.where(meanPercAcc>shCovThresh)[0][0]+1
        
        axScore = plotInfo['axScore']
        
        lblLL = plotInfo['lblLL']
        lblLLErr = plotInfo['lblLLErr']
        description = plotInfo['description']

        if cvApproach is "logLikelihood":
            axScore.plot(xDimTest,gpfaScoreMn, label = lblLL)
            axScore.fill_between(xDimTest, gpfaScoreMn-gpfaScoreErr, gpfaScoreMn+gpfaScoreErr, alpha=0.2, label =lblLLErr)
            # axScore.set_title(description)
        elif cvApproach is "squaredError":
            axScore.plot(xDimTest,gpfaScoreMn, label = 'Summed GPFA Error Over Folds')
            axScore.plot(np.arange(len(reducedGpfaScore))+1, reducedGpfaScore, label='Summed Reduced GPFA Error Over Folds')
            # axScore.set_title(description)
        # axScore.legend(loc='upper right')
        
        axCumulDim = plotInfo['axCumulDim']
        axByDim = plotInfo['axByDim']
        axByDim.set_xlabel("num dims")
        
        axCumulDim.plot(np.arange(len(meanPercAcc))+1,meanPercAcc, '.' if len(meanPercAcc)==1 else '-')
        axCumulDim.fill_between(np.arange(len(meanPercAcc))+1, meanPercAcc-stdPercAcc,meanPercAcc+stdPercAcc,alpha=0.2)
        axByDim.plot(np.arange(len(meanAccByDim))+1,meanAccByDim, '.' if len(meanAccByDim)==1 else '-')
        axByDim.fill_between(np.arange(len(meanAccByDim))+1, meanAccByDim-stdAccByDim,meanAccByDim+stdAccByDim,alpha=0.2)
        
        axCumulDim.axhline(shCovThresh, linestyle='--')
        axCumulDim.axhline(0, linestyle='--', color='k')
        
        axScore.axvline(xDimBest, linestyle='--')

        xlD = axCumulDim.get_xlim()
        xlS = axScore.get_xlim()
        axScore.set_xticks(xDimTest)
        axScore.set_xticklabels(xDimTest)
        axScore.set_xlim(xmin=np.min([xlD[0],xlS[0]]), xmax=np.max([xlD[1],xlS[1]]))
        axScore.spines['right'].set_visible(False)
        axScore.spines['top'].set_visible(False)

        axCumulDim.xaxis.set_ticklabels('')
        axCumulDim.xaxis.set_visible(False)
        axCumulDim.spines['bottom'].set_visible(False)
        axCumulDim.spines['right'].set_visible(False)
        axCumulDim.spines['top'].set_visible(False)
        axByDim.set_xticks(xDimTest)
        axByDim.set_xticklabels(xDimTest)
        axByDim.spines['right'].set_visible(False)
        axByDim.spines['top'].set_visible(False)

       
        alBins = alignmentBins[idx]
        
        tmValsStartBest = tmVals[0]
        tmValsEndBest = tmVals[1]

        for cValUse in [0]:# range(crossvalidateNum):
            condLabel = condLabels[idx]
            if baselineSubtract:
                # NOTE: This should probably be pulled out of this function; will let it fail for now because labelMeans isn't defined
                # meanTraj = gpfaPrep.projectTrajectory(gpfaPrep.dimOutput[xDimScoreBest]['allEstParams'][cValUse], labelMeans[idx][None,:,:], sqrtSpks = sqrtSpikes)
                meanTraj = projectTrajectory(dimResult[xDimScoreBest]['allEstParams'][cValUse], np.stack(labelMeans))
                
            if computeShuffles:
                # NOTE this will fail for the moment... I'm going to ignore it until it's needed again, though
                # this has nothing to do with baseline subtracting... I'm just being lazy
                shuffTraj = shuffleGpfaControl(dimResult[xDimScoreBest]['allEstParams'][cValUse], cvalTest = cValUse, sqrtSpks = sqrtSpikes)

            rowsPlt = 2
            if tmValsStartBest.size and tmValsEndBest.size:
                colsPlt = np.ceil(xDimBest/rowsPlt)*2 # align to start and to end...
            else:
                colsPlt = np.ceil(xDimBest/rowsPlt) # aligning to only one of them...
            axesStart = []
            axesEnd = []
            axVals = np.empty((0,4))
            figSep = plt.figure()
            figSep.suptitle(description + " cond " + str(condLabel) + "")

            if xDimBest>2:
                figTraj = plt.figure()
                axStartTraj = plt.subplot(1,3,1,projection='3d')
                axEndTraj = plt.subplot(1,3,2,projection='3d')
                axAllTraj = plt.subplot(1,3,3,projection='3d')
                figTraj.suptitle(description + " cond " + str(condLabel) + "")
            elif xDimBest>1:
                figTraj = plt.figure()
                axStartTraj = plt.subplot(1,3,1)
                axEndTraj = plt.subplot(1,3,2)
                axAllTraj = plt.subplot(1,3,3)
                figTraj.suptitle(description + " cond " + str(condLabel) + "")
        
            plt.figure()
            plt.imshow(np.abs(dimResult[xDimScoreBest]['allEstParams'][cValUse]['C']),aspect="auto")
            plt.title('C matrix (not orth)')
            plt.colorbar()


            # seqTrainNewAll = gpfaPrep.dimOutput[xDimScoreBest]['seqsTrainNew'][cValUse]
            # seqTrainOrthAll = [sq['xorth'] for sq in seqTrainNewAll]
            # seqTrainConcat = np.concatenate(seqTrainOrthAll, axis = 1)
            # plt.figure()
            # plt.plot(seqTrainConcat.T)



            seqTestUse = dimResult[xDimScoreBest]['seqsTestNew'][cValUse] # just use the first one...
            lblStartTraj = 'traj start'
            lblEndTraj = 'traj end'
            lblDelayStart = 'delay start'
            lblDelayEnd = 'delay end'
            lblNoDelayStart = 'delay start outside traj'
            lblNoDelayEnd = 'delay end outside traj'
            lblNeuralTraj = 'neural trajectories'
            trainInds = [sq['trialId'] for sq in dimResult[xDimScoreBest]['seqsTrainNew'][0]]
            mxInd = np.max(np.hstack([trainInds, testInds[cValUse]]))
            for k, (sq, tstInd) in enumerate(zip(seqTestUse,testInds[cValUse])):
                # if k>5:
                    # break
                if alBins.shape[0] > 1:
                    alB = alBins[tstInd]
                else:
                    # might happen if there's the same alignment bin for all trials...
                    alB = alBins[0]
                # sq = {}
                # sq['xorth'] = np.concatenate((sq2['xorth'][1:], sq2['xorth'][:1]), axis=0)
                # gSp = grpSpks[tstInd]
                # print(gSp.alignmentBins.shape)
                if tmValsStartBest.size:
                    startZeroBin = alB[0]
                    fstBin = 0
                    tmBeforeStartZero = (fstBin-startZeroBin)*binSize
                    tmValsStart = tmValsStartBest[tmValsStartBest>=tmBeforeStartZero]
                else:
                    tmValsStart = np.ndarray((0,0))
                    
                if tmValsEndBest.size:
                    # Only plot what we have data for...
                    endZeroBin = alB[1]
                    # going into gSp[0] because gSp might be dtype object instead of the float64,
                    # so might have trials within the array, not accessible without
                    lastBin = sq['xorth'].shape[1]
                    timeAfterEndZero = (lastBin-endZeroBin)*binSize
                    tmValsEnd = tmValsEndBest[tmValsEndBest<timeAfterEndZero]
                else:
                    tmValsEnd = np.ndarray((0,0))
                    
                if xDimBest>2:
                    plt.figure(figTraj.number)
                    if tmValsStart.size:
                        plotDimensionsAgainstEachOther(axStartTraj, sq['xorth'][:3], tmValsStart, [0,tmValsStart.shape[0]], lblStartTraj, lblEndTraj, [lblDelayStart, lblNoDelayStart], colorset, idx, axTtl = 'Start', axLabel = ['gpfa 1', 'gpfa 2', 'gpfa 3'], colorAlignPts = 'green')
                        # axStartTraj.plot(sq['xorth'][0,:tmValsStart.shape[0]], sq['xorth'][1,:tmValsStart.shape[0]], sq['xorth'][2,:tmValsStart.shape[0]],
                        #        color=colorset[idx,:], linewidth=0.4)                            
                        # axStartTraj.plot([sq['xorth'][0,0]], [sq['xorth'][1,0]], [sq['xorth'][2,0]], 'o',
                        #        color=colorset[idx,:], linewidth=0.4, label=lblStartTraj, markeredgecolor='black')
                        # axStartTraj.plot([sq['xorth'][0,tmValsStart.shape[0]-1]], [sq['xorth'][1,tmValsStart.shape[0]-1]], [sq['xorth'][2,tmValsStart.shape[0]-1]], '>',
                        #        color=colorset[idx,:], linewidth=0.4, label=lblEndTraj, markeredgecolor='black')
                        
                        # # marking the alginment point here
                        # if np.min(tmValsStart) <= 0 and np.max(tmValsStart) >= 0:
                        #     alignX = np.interp(0, tmValsStart, sq['xorth'][0,:tmValsStart.shape[0]])
                        #     alignY = np.interp(0, tmValsStart, sq['xorth'][1,:tmValsStart.shape[0]])
                        #     alignZ = np.interp(0, tmValsStart, sq['xorth'][2,:tmValsStart.shape[0]])
                        #     axStartTraj.plot([alignX], [alignY], [alignZ], '*', color='green', label = 'delay start alignment')
                        # else:
                        #     axStartTraj.plot([np.nan], [np.nan], '*', color='green', label =lblNoDelayStart)
                        
                        # axStartTraj.set_title('Start')
                        # axStartTraj.set_xlabel('gpfa 1')
                        # axStartTraj.set_ylabel('gpfa 2')
                        # axStartTraj.set_zlabel('gpfa 3')
                        # axStartTraj.legend()
                    
                    if tmValsEnd.size:
                        plotDimensionsAgainstEachOther(axEndTraj, sq['xorth'][:3], tmValsEnd, [-tmValsEnd.shape[0],sq['xorth'].shape[1]], lblStartTraj, lblEndTraj, [lblDelayEnd, lblNoDelayEnd], colorset, idx, axTtl = 'End', axLabel = ['gpfa 1', 'gpfa 2', 'gpfa 3'], colorAlignPts = 'red')
                        # axEndTraj.plot(sq['xorth'][0,-tmValsEnd.shape[0]:], sq['xorth'][1,-tmValsEnd.shape[0]:], sq['xorth'][2,-tmValsEnd.shape[0]:],
                        #            color=colorset[idx,:], linewidth=0.4)
                        # axEndTraj.plot([sq['xorth'][0,-tmValsEnd.shape[0]]], [sq['xorth'][1,-tmValsEnd.shape[0]]], [sq['xorth'][2,-tmValsEnd.shape[0]]], 'o',
                        #        color=colorset[idx,:], linewidth=0.4, label=lblEndTraj, markeredgecolor='black')
                        # axEndTraj.plot([sq['xorth'][0,-1]], [sq['xorth'][1,-1]], [sq['xorth'][2,-1]], '>',
                        #        color=colorset[idx,:], linewidth=0.4, label=lblEndTraj, markeredgecolor='black')
                        
                        # # marking the alginment point here
                        # if np.min(tmValsEnd) <= 0 and np.max(tmValsEnd) >= 0:
                        #     alignX = np.interp(0, tmValsEnd, sq['xorth'][0,-tmValsEnd.shape[0]:])
                        #     alignY = np.interp(0, tmValsEnd, sq['xorth'][1,-tmValsEnd.shape[0]:])
                        #     alignZ = np.interp(0, tmValsEnd, sq['xorth'][2,-tmValsEnd.shape[0]:])
                        #     axEndTraj.plot([alignX], [alignY], [alignZ], '*', color='red', label = 'delay end alignment')
                        # else:
                        #     axEndTraj.plot([np.nan], [np.nan], '*', color='red', label =lblNoDelayEnd)
                        
                        # axEndTraj.set_title('End')
                        # axEndTraj.set_xlabel('gpfa 1')
                        # axEndTraj.set_ylabel('gpfa 2')
                        # axEndTraj.set_zlabel('gpfa 3')
                        axEndTraj.legend()
                        
                    if tmValsStart.size and tmValsEnd.size:
                        # the binSize and start are detailed here, so we can find the rest of time
                        lenT = sq['xorth'].shape[1]
                        allTimesAlignStart = np.arange(tmValsStart[0], tmValsStart[0]+binSize*lenT, binSize)
                        # gotta play some with the end to make sure 0 is aligned from the end as in tmValsEnd
                        allTimesAlignEnd = np.arange(tmValsEnd[-1]-binSize*(lenT-1), tmValsEnd[-1]+binSize/10, binSize)
                        
                        plotDimensionsAgainstEachOther(axAllTraj, sq['xorth'][:3], [allTimesAlignStart, allTimesAlignEnd], [0,sq['xorth'].shape[1]], lblStartTraj, lblEndTraj, [[lblDelayStart, lblNoDelayStart],[lblDelayEnd, lblNoDelayEnd]], colorset, idx, axTtl = 'All', axLabel = ['gpfa 1', 'gpfa 2', 'gpfa 3'], colorAlignPts = ['green', 'red'])
                        # axAllTraj.plot(sq['xorth'][0,:], sq['xorth'][1,:], sq['xorth'][2,:],
                        #            color=colorset[idx,:], linewidth=0.4)
                        # axAllTraj.plot([sq['xorth'][0,0]], [sq['xorth'][1,0]], [sq['xorth'][2,0]], 'o',
                        #            color=colorset[idx,:], linewidth=0.4, label=lblStartTraj, markeredgecolor='black')
                        # axAllTraj.plot([sq['xorth'][0,-1]], [sq['xorth'][1,-1]], [sq['xorth'][2,-1]], '>',
                        #            color=colorset[idx,:], linewidth=0.4, label=lblEndTraj, markeredgecolor='black')
                        
                        
                        # # marking the alginment points here
                        # # the binSize and start are detailed here, so we can find the rest of time
                        # lenT = sq['xorth'].shape[1]
                        # allTimesAlignStart = np.arange(tmValsStart[0], tmValsStart[0]+binSize*lenT, binSize)
                        # if np.min(allTimesAlignStart) <= 0 and np.max(allTimesAlignStart) >= 0:
                        #     alignXStart = np.interp(0, allTimesAlignStart, sq['xorth'][0,:])
                        #     alignYStart = np.interp(0, allTimesAlignStart, sq['xorth'][1,:])
                        #     alignZStart = np.interp(0, allTimesAlignStart, sq['xorth'][2,:])
                        #     axAllTraj.plot([alignXStart], [alignYStart], [alignZStart], '*', color='green', label =lblDelayStart)
                        # else:
                        #     axAllTraj.plot([np.nan], [np.nan], '*', color='green', label =lblNoDelayStart)
                            
                        # # gotta play some with the end to make sure 0 is aligned from the end as in tmValsEnd
                        # allTimesAlignEnd = np.arange(tmValsEnd[-1]-binSize*(lenT-1), tmValsEnd[-1]+binSize/10, binSize)
                        # if np.min(allTimesAlignEnd) <= 0 and np.max(allTimesAlignEnd) >= 0:
                        #     alignXEnd = np.interp(0, allTimesAlignEnd, sq['xorth'][0,:])
                        #     alignYEnd = np.interp(0, allTimesAlignEnd, sq['xorth'][1,:])
                        #     alignZEnd = np.interp(0, allTimesAlignEnd, sq['xorth'][2,:])
                        #     axAllTraj.plot([alignXEnd], [alignYEnd], [alignZEnd], '*', color='red', label = lblDelayEnd)
                        # else:
                        #     axAllTraj.plot([np.nan], [np.nan], '*', color='red', label =lblNoDelayEnd)
                            
                                
                        
                        # axAllTraj.set_title('All')
                        # axAllTraj.set_xlabel('gpfa 1')
                        # axAllTraj.set_ylabel('gpfa 2')
                        # axAllTraj.set_zlabel('gpfa 3')
                        # axAllTraj.legend()

                elif xDimBest>1:
                    plt.figure(figTraj.number)
                    if tmValsStart.size:
                        plotDimensionsAgainstEachOther(axStartTraj, sq['xorth'][:2], tmValsStart, [0,tmValsStart.shape[0]], lblStartTraj, lblEndTraj, [lblDelayStart, lblNoDelayStart], colorset, idx, axTtl = 'Start', axLabel = ['gpfa 1', 'gpfa 2'], colorAlignPts = 'green')

                        # axStartTraj.plot(sq['xorth'][0,:tmValsStart.shape[0]], sq['xorth'][1,:tmValsStart.shape[0]],
                        #        color=colorset[idx,:], linewidth=0.4)
                        # axStartTraj.plot(sq['xorth'][0,0], sq['xorth'][1,0], 'o',
                        #        color=colorset[idx,:], linewidth=0.4, label='traj start', markeredgecolor='black')
                        # axStartTraj.plot(sq['xorth'][0,tmValsStart.shape[0]-1], sq['xorth'][1,tmValsStart.shape[0]-1], '>',
                        #        color=colorset[idx,:], linewidth=0.4, label='traj end', markeredgecolor='black')
                        
                        # # marking the alginment point here
                        # if np.min(tmValsStart) <= 0 and np.max(tmValsStart) >= 0:
                        #     alignX = np.interp(0, tmValsStart, sq['xorth'][0,:tmValsStart.shape[0]])
                        #     alignY = np.interp(0, tmValsStart, sq['xorth'][1,:tmValsStart.shape[0]])
                        #     axStartTraj.plot(alignX, alignY, '*', color='green', label =lblDelayStart)
                        # else:
                        #     axStartTraj.plot(np.nan, '*', color='green', label =lblNoDelayStart)
                        # axStartTraj.set_title('Start')
                        # axStartTraj.set_xlabel('gpfa 1')
                        # axStartTraj.set_ylabel('gpfa 2')
                        # axStartTraj.legend()
                    
                    if tmValsEnd.size:
                        plotDimensionsAgainstEachOther(axEndTraj, sq['xorth'][:2], tmValsEnd, [-tmValsEnd.shape[0],sq['xorth'].shape[1]], lblStartTraj, lblEndTraj, [lblDelayEnd, lblNoDelayEnd], colorset, idx, axTtl = 'End', axLabel = ['gpfa 1', 'gpfa 2'], colorAlignPts = 'red')

                        # axEndTraj.plot(sq['xorth'][0,-tmValsEnd.shape[0]:], sq['xorth'][1,-tmValsEnd.shape[0]:],
                        #            color=colorset[idx,:], linewidth=0.4)
                        # axEndTraj.plot(sq['xorth'][0,-tmValsEnd.shape[0]], sq['xorth'][1,-tmValsEnd.shape[0]], 'o',
                        #        color=colorset[idx,:], linewidth=0.4, label='traj start', markeredgecolor='black')
                        # axEndTraj.plot(sq['xorth'][0,-1], sq['xorth'][1,-1], '>',
                        #        color=colorset[idx,:], linewidth=0.4, label='traj end', markeredgecolor='black')
                        
                        # # marking the alginment point here
                        # if np.min(tmValsEnd) <= 0 and np.max(tmValsEnd) >= 0:
                        #     alignX = np.interp(0, tmValsEnd, sq['xorth'][0,-tmValsEnd.shape[0]:])
                        #     alignY = np.interp(0, tmValsEnd, sq['xorth'][1,-tmValsEnd.shape[0]:])
                        #     axEndTraj.plot(alignX, alignY, '*', color='red', label =lblDelayEnd)
                        # else:
                        #     axEndTraj.plot(np.nan, '*', color='red', label =lblNoDelayEnd)

                        # axEndTraj.set_title('End')
                        # axEndTraj.set_xlabel('gpfa 1')
                        # axEndTraj.set_ylabel('gpfa 2')
                        # axEndTraj.legend()
                    
                    if tmValsStart.size and tmValsEnd.size:
                        # the binSize and start are detailed here, so we can find the rest of time
                        lenT = sq['xorth'].shape[1]
                        allTimesAlignStart = np.arange(tmValsStart[0], tmValsStart[0]+binSize*lenT, binSize)
                        # gotta play some with the end to make sure 0 is aligned from the end as in tmValsEnd
                        allTimesAlignEnd = np.arange(tmValsEnd[-1]-binSize*(lenT-1), tmValsEnd[-1]+binSize/10, binSize)
                        
                        plotDimensionsAgainstEachOther(axAllTraj, sq['xorth'][:2], [allTimesAlignStart, allTimesAlignEnd], [0,sq['xorth'].shape[1]], lblStartTraj, lblEndTraj, [[lblDelayStart, lblNoDelayStart],[lblDelayEnd, lblNoDelayEnd]], colorset, idx, axTtl = 'All', axLabel = ['gpfa 1', 'gpfa 2'], colorAlignPts = ['green', 'red'])

                        # axAllTraj.plot(sq['xorth'][0,:], sq['xorth'][1,:],
                        #            color=colorset[idx,:], linewidth=0.4)
                        # axAllTraj.plot(sq['xorth'][0,0], sq['xorth'][1,0], 'o',
                        #            color=colorset[idx,:], linewidth=0.4, label=lblStartTraj, markeredgecolor='black')
                        # axAllTraj.plot(sq['xorth'][0,-1], sq['xorth'][1,-1], '>',
                        #            color=colorset[idx,:], linewidth=0.4, label=lblEndTraj, markeredgecolor='black')
                        
                        # # marking the alginment points here
                        # # the binSize and start are detailed here, so we can find the rest of time
                        # lenT = sq['xorth'].shape[1]
                        # allTimesAlignStart = np.arange(tmValsStart[0], tmValsStart[0]+binSize*lenT, binSize)
                        # if np.min(allTimesAlignStart) <= 0 and np.max(allTimesAlignStart) >= 0:
                        #     alignXStart = np.interp(0, allTimesAlignStart, sq['xorth'][0,:])
                        #     alignYStart = np.interp(0, allTimesAlignStart, sq['xorth'][1,:])
                        #     axAllTraj.plot(alignXStart, alignYStart, '*', color='green', label =lblDelayStart)
                        # else:
                        #     axAllTraj.plot(np.nan, '*', color='green', label =lblNoDelayStart)
                            
                        # # gotta play some with the end to make sure 0 is aligned from the end as in tmValsEnd
                        # allTimesAlignEnd = np.arange(tmValsEnd[-1]-binSize*(lenT-1), tmValsEnd[-1]+binSize/10, binSize)
                        # if np.min(allTimesAlignEnd) <= 0 and np.max(allTimesAlignEnd) >= 0:
                        #     alignXEnd = np.interp(0, allTimesAlignEnd, sq['xorth'][0,:])
                        #     alignYEnd = np.interp(0, allTimesAlignEnd, sq['xorth'][1,:])
                        #     axAllTraj.plot(alignXEnd, alignYEnd, '*', color='red', label =lblDelayEnd)
                        # else:
                        #     axAllTraj.plot(np.nan, '*', color='red', label =lblNoDelayEnd)
                            
                                
                        
                        # axAllTraj.set_title('All')
                        # axAllTraj.set_xlabel('gpfa 1')
                        # axAllTraj.set_ylabel('gpfa 2')
                        # axAllTraj.legend()

                
                if True:
                    pltNum = 1
                    pltListNum = 0
                    plt.figure(figSep.number)
                    plt.suptitle(description + " cond " + str(condLabel) + "")
                    try:
                        zip(sq['xorth'])
                    except TypeError:
                        sq['xorth'] = [sq['xorth']]
                    for dimNum, dim in enumerate(sq['xorth']):
                        dimNum = dimNum+1 # first is 1-dimensional, not zero-dimensinoal
                        #we'll only plot the xDimBest dims...
                        if dimNum > xDimBest:
                            break

                        try:
                            dim[0]
                        except TypeError:
                            dim = [dim]
                        
                        mxColorInd = 255
                        thisColorInd = np.round(tstInd/mxInd*255)
                        timeLineColor = plt.cm.viridis(int(thisColorInd))
                        if tmValsStart.size:
                            axUsed = plotDimensionsAgainstTime(figSep, pltNum, pltListNum, axesStart, axesEnd, rowsPlt, colsPlt, tmValsStart, dim[:tmValsStart.shape[0]], dimNum, xDimBest, timeLineColor, lblNeuralTraj, axTitle = 'Start')
                            # if len(axesStart) + len(axesEnd) <rowsPlt*colsPlt:
                            #     axesHere = figSep.add_subplot(rowsPlt, colsPlt, pltNum)
                            #     axesStart.append(axesHere)
                            #     if pltNum == 1: 
                            #         axesHere.set_title("dim " + str(dimNum) + " periStart")
                            #     else:
                            #         axesHere.set_title("d"+str(dimNum)+"S")

                            #     if pltNum <= (xDimBest - colsPlt):
                            #         axesHere.set_xticklabels('')
                            #         axesHere.xaxis.set_visible(False)
                            #         axesHere.spines['bottom'].set_visible(False)
                            #     else:  
                            #         axesHere.set_xlabel('time (ms)')

                            #     if (pltNum % colsPlt) != 1:
                            #         axesHere.set_yticklabels('')
                            #         axesHere.yaxis.set_visible(False)
                            #         axesHere.spines['left'].set_visible(False)
                            # else:
                            #     if tmValsEnd.size:
                            #         axesHere = axesStart[int((pltNum-1)/2)]
                            #     else:
                            #         axesHere = axesStart[int(pltNum-1)]    
                            #     plt.axes(axesHere)
                        
                            # plt.plot(tmValsStart, dim[:tmValsStart.shape[0]], color=colorset[idx,:], linewidth=0.4, label=lblNeuralTraj)
                        
                        
                        
                            axVals = np.append(axVals, np.array(axUsed.axis())[None, :], axis=0)
                            pltNum += 1
                        
                        if tmValsEnd.size:
                            axUsed = plotDimensionsAgainstTime(figSep, pltNum, pltListNum, axesEnd, axesStart, rowsPlt, colsPlt, tmValsEnd, dim[-tmValsEnd.shape[0]:], dimNum, xDimBest, timeLineColor, lblNeuralTraj, axTitle = 'End')
                            # if len(axesStart) + len(axesEnd) <rowsPlt*colsPlt:
                            #     axesHere = figSep.add_subplot(rowsPlt, colsPlt, pltNum)
                            #     axesEnd.append(axesHere)
                            #     if pltNum == colsPlt:
                            #         axesHere.set_title("dim " + str(dimNum) + " periEnd")
                            #     else:
                            #         axesHere.set_title("d"+str(dimNum)+"E")

                            #     if pltNum <= (xDimBest - colsPlt):
                            #         axesHere.set_xticklabels('')
                            #         axesHere.xaxis.set_visible(False)
                            #         axesHere.spines['bottom'].set_visible(False)
                            #     else:  
                            #         axesHere.set_xlabel('time (ms)')

                            #     if (pltNum % colsPlt) != 1:
                            #         axesHere.set_yticklabels('')
                            #         axesHere.yaxis.set_visible(False)
                            #         axesHere.spines['left'].set_visible(False)
                            # else:
                            #     if tmValsStart.size:
                            #         axesHere = axesEnd[int(pltNum/2-1)]
                            #     else:
                            #         axesHere = axesEnd[int(pltNum-1)]
                            #     plt.axes(axesHere)
                    
                            # plt.plot(tmValsEnd, dim[-tmValsEnd.shape[0]:], color=colorset[idx,:], linewidth=0.4, label=lblNeuralTraj)

                            axVals = np.append(axVals, np.array(axUsed.axis())[None, :], axis=0)
                            pltNum += 1
                            
                        pltListNum += 1
                        
                    axUsed.legend()

                lblStartTraj = None
                lblEndTraj = None
                lblDelayStart = None
                lblDelayEnd = None
                lblNoDelayStart = None
                lblNoDelayEnd = None
                lblNeuralTraj = None

            m = plt.cm.ScalarMappable(cmap=plt.cm.viridis)
            m.set_array(testInds)
            figSep.colorbar(m)
            
            # now I'm gonna plot the shuffles
            if computeShuffles:
                lblShuffle = 'shuffles'
                for sq in shuffTraj:
                    pltNum = 1
                    pltListNum = 0
                    plt.figure(figSep.number)
                    for dimNum, dim in enumerate(sq['xorth']):
                        dimNum = dimNum+1 # first is 1-dimensional, not zero-dimensinoal
                        #we'll only plot the xDimBest dims...
                        if dimNum > xDimBest:
                            break
                        
                        if tmValsStart.size:
                            axUsed = plotDimensionsAgainstTime(figSep, pltNum, pltListNum, axesStart, axesEnd, rowsPlt, colsPlt, tmValsStart, dim[:tmValsStart.shape[0]], dimNum, xDimBest, colorUse = [0.5,0.5,0.5], labelTraj = lblShuffle, linewidth=0.2, alpha = 0.5, axTitle = 'Start')
                            # if len(axesStart) + len(axesEnd) <rowsPlt*colsPlt:
                            #     axesHere = figSep.add_subplot(rowsPlt, colsPlt, pltNum)
                            #     axesStart.append(axesHere)
                            #     if pltNum == 1:
                            #         axesHere.set_title("dim " + str(dimNum) + " periStart")
                            #     else:
                            #         axesHere.set_title("d"+str(dimNum)+"S")

                            #     if pltNum <= (xDimBest - colsPlt):
                            #         axesHere.set_xticklabels('')
                            #         axesHere.xaxis.set_visible(False)
                            #         axesHere.spines['bottom'].set_visible(False)
                            #     else:  
                            #         axesHere.set_xlabel('time (ms)')

                            #     if (pltNum % colsPlt) != 1:
                            #         axesHere.set_yticklabels('')
                            #         axesHere.yaxis.set_visible(False)
                            #         axesHere.spines['left'].set_visible(False)
                            # else:
                            #     if tmValsEnd.size:
                            #         axesHere = axesStart[int((pltNum-1)/2)]
                            #     else:
                            #         axesHere = axesStart[int(pltNum-1)]    
                            #     plt.axes(axesHere)
                        
                            # plt.plot(tmValsStart, dim[:tmValsStart.shape[0]], color=[0.5,0.5,0.5], linewidth=0.2, alpha=0.5, label=lblShuffle)
                        
                        
                        
                            axVals = np.append(axVals, np.array(axUsed.axis())[None, :], axis=0)
                            pltNum += 1
                        
                        if tmValsEnd.size:
                            axUsed = plotDimensionsAgainstTime(figSep, pltNum, pltListNum, axesEnd, axesStart, rowsPlt, colsPlt, tmValsEnd, dim[-tmValsEnd.shape[0]:], dimNum, xDimBest, colorUse = [0.5,0.5,0.5], labelTraj = lblShuffle, linewidth=0.2, alpha = 0.5, axTitle = 'End')
                            # if len(axesStart) + len(axesEnd) <rowsPlt*colsPlt:
                            #     axesHere = figSep.add_subplot(rowsPlt, colsPlt, pltNum)
                            #     axesEnd.append(axesHere)
                            #     if pltNum == colsPlt:
                            #         axesHere.set_title("dim " + str(dimNum) + " periEnd")
                            #     else:
                            #         axesHere.set_title("d"+str(dimNum)+"E")

                            #     if pltNum <= (xDimBest - colsPlt):
                            #         axesHere.set_xticklabels('')
                            #         axesHere.xaxis.set_visible(False)
                            #         axesHere.spines['bottom'].set_visible(False)
                            #     else:  
                            #         axesHere.set_xlabel('time (ms)')

                            #     if (pltNum % colsPlt) != 1:
                            #         axesHere.set_yticklabels('')
                            #         axesHere.yaxis.set_visible(False)
                            #         axesHere.spines['left'].set_visible(False)
                            # else:
                            #     if tmValsStart.size:
                            #         axesHere = axesEnd[int(pltNum/2-1)]
                            #     else:
                            #         axesHere = axesEnd[int(pltNum-1)]
                            #     plt.axes(axesHere)
                    
                            # plt.plot(tmValsEnd, dim[-tmValsEnd.shape[0]:], color=[0.5,0.5,0.5], linewidth=0.2, alpha=0.5, label=lblShuffle)                 
                            axVals = np.append(axVals, np.array(axUsed.axis())[None, :], axis=0)
                            pltNum += 1
                        
                        pltListNum += 1

                    axUsed.legend()
                    lblShuffle = None


            # just keeps getting uglier, but this is for the mean trajectory...
            if baselineSubtract:
                lblMn = 'mean traj per cond'
                for condNum, mnTraj in enumerate(meanTraj):
                    pltNum = 1
                    pltListNum = 0
                    plt.figure(figSep.number)
                    for dimNum, dim in enumerate(mnTraj['xorth']):
                        dimNum = dimNum+1 # first is 1-dimensional, not zero-dimensinoal
                        #we'll only plot the xDimBest dims...
                        if dimNum > xDimBest:
                            break

                        
                        if tmValsStart.size:
                            axUsed = plotDimensionsAgainstTime(figSep, pltNum, pltListNum, axesStart, axesEnd, rowsPlt, colsPlt, tmValsStart, dim[:tmValsStart.shape[0]], dimNum, xDimBest, colorset[condNum,:], lblMn, linewidth=1, axTitle = 'Start')
                            # if len(axesStart) + len(axesEnd) <rowsPlt*colsPlt:
                            #     axesHere = figSep.add_subplot(rowsPlt, colsPlt, pltNum)
                            #     axesStart.append(axesHere)
                            #     if pltNum == 1:
                            #         axesHere.set_title("dim " + str(dimNum) + " periStart")
                            #     else:
                            #         axesHere.set_title("d"+str(dimNum)+"S")

                            #     if pltNum <= (xDimBest - colsPlt):
                            #         axesHere.set_xticklabels('')
                            #         axesHere.xaxis.set_visible(False)
                            #         axesHere.spines['bottom'].set_visible(False)
                            #     else:  
                            #         axesHere.set_xlabel('time (ms)')

                            #     if (pltNum % colsPlt) != 1:
                            #         axesHere.set_yticklabels('')
                            #         axesHere.yaxis.set_visible(False)
                            #         axesHere.spines['left'].set_visible(False)
                            # else:
                            #     if tmValsEnd.size:
                            #         axesHere = axesStart[int((pltNum-1)/2)]
                            #     else:
                            #         axesHere = axesStart[int(pltNum-1)]    
                            #     plt.axes(axesHere)
                        
#                                plt.plot(tmValsStart, dim[:tmValsStart.shape[0]], color=colorset[idx,:][0,0,0], linewidth=1,label=lblMn)
                            # plt.plot(tmValsStart, dim[:tmValsStart.shape[0]], color=colorset[condNum,:], linewidth=1,label=lblMn)
                        
                        
                        
                            axVals = np.append(axVals, np.array(axUsed.axis())[None, :], axis=0)
                            pltNum += 1
                        
                        if tmValsEnd.size:
                            axUsed = plotDimensionsAgainstTime(figSep, pltNum, pltListNum, axesEnd, axesStart, rowsPlt, colsPlt, tmValsEnd, dim[-tmValsEnd.shape[0]:], dimNum, xDimBest, colorset[condNum,:], lblMn, linewidth=1, axTitle = 'End')
                            # if len(axesStart) + len(axesEnd) <rowsPlt*colsPlt:
                            #     axesHere = figSep.add_subplot(rowsPlt, colsPlt, pltNum)
                            #     axesEnd.append(axesHere)
                            #     if pltNum == colsPlt:
                            #         axesHere.set_title("dim " + str(dimNum) + " periEnd")
                            #     else:
                            #         axesHere.set_title("d"+str(dimNum)+"E")

                            #     if pltNum <= (xDimBest - colsPlt):
                            #         axesHere.set_xticklabels('')
                            #         axesHere.xaxis.set_visible(False)
                            #         axesHere.spines['bottom'].set_visible(False)
                            #     else:  
                            #         axesHere.set_xlabel('time (ms)')

                            #     if (pltNum % colsPlt) != 1:
                            #         axesHere.set_yticklabels('')
                            #         axesHere.yaxis.set_visible(False)
                            #         axesHere.spines['left'].set_visible(False)

                            # else:
                            #     if tmValsStart.size:
                            #         axesHere = axesEnd[int(pltNum/2-1)]
                            #     else:
                            #         axesHere = axesEnd[int(pltNum-1)]
                            #     plt.axes(axesHere)
                    
                            # plt.plot(tmValsEnd, dim[-tmValsEnd.shape[0]:], color=[0,0,0], linewidth=1, label=lblMn)                 
                            axVals = np.append(axVals, np.array(axUsed.axis())[None, :], axis=0)
                            pltNum += 1
                            
                        pltListNum += 1
                    axesHere.legend() #legend on the last plot...
                    lblMn = None
            
            ymin = np.min(np.append(0, np.min(axVals, axis=0)[2]))
            ymax = np.max(axVals, axis=0)[3]
            for ax in axesStart:
                ax.set_ylim(bottom = ymin, top = ymax )
                plt.axes(ax)
                plt.axvline(x=0, linestyle=':', color='black')
                plt.axhline(y=0, linestyle=':', color='black')
#                        xl = ax.get_xlim()
#                        yl = ax.get_ylim()
#                        ax.axhline(y=0,xmin=xl[0],xmax=xl[1], linestyle = ':', color='black')
#                        ax.axvline(x=0,ymin=xl[0],ymax=xl[1], linestyle = ':', color='black')
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                
            for ax in axesEnd:
                ax.set_ylim(bottom = ymin, top = ymax )
                plt.axes(ax)
                plt.axvline(x=0, linestyle=':', color='black')
                plt.axhline(y=0, linestyle=':', color='black')
#                        xl = ax.get_xlim()
#                        yl = ax.get_ylim()
#                        ax.axhline(y=0,xmin=xl[0],xmax=xl[1], linestyle = ':', color='black')
#                        ax.axvline(x=0,ymin=xl[0],ymax=xl[1], linestyle = ':', color='black')
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)

def projectTrajectory(estParams, trajectory):
    from classes import GPFA
    seqTrajDict = GPFA.binSpikesToGpfaInputDict([], binnedSpikes = trajectory)
    from multiprocessing import Pool
    with Pool() as poolHere:
        res = []

        res.append(poolHere.apply_async(GPFA.projectTrajectory, (seqTrajDict,estParams)))

        resultsTraj = [rs.get() for rs in res]

    return resultsTraj[0]
