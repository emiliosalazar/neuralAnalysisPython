#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 17:05:23 2020

@author: emilio
"""
from matplotlib import pyplot as plt
import numpy as np
import scipy as sp
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
   

def visualizeGpfaResults(plotInfo, dimResults, tmVals, cvApproach, gpfaScoreAll, xDimTest, testInds, shCovThresh, binSize, condLabels, alignmentBins,
            crossValsUse=[0], trlTimes = None, spikesUsedNorm = None,
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
        
        
        Cparams = [prm['C'] for prm in dimResult[xDimScoreBest]['allEstParams']]
        shEigs = [np.flip(np.sort(np.linalg.eig(C.T @ C)[0])) for C in Cparams]
        percAcc = np.stack([np.cumsum(eVals)/np.sum(eVals) for eVals in shEigs])
        meanPercAcc = np.mean(percAcc, axis=0)
        stdPercAcc = np.std(percAcc, axis = 0)
        accByDim = np.stack([eVals/np.sum(eVals) for eVals in shEigs])
        meanAccByDim = np.mean(accByDim, axis=0)
        stdAccByDim = np.std(accByDim, axis = 0)
        if meanPercAcc.size>0:
            xDimBest = np.where(meanPercAcc>shCovThresh)[0][0]+1
        else:
            xDimBest = 0
        
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

        if xDimScoreBest == 0:
            continue

        for cValUse in crossValsUse:
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

                # kinda wanna plot all dimensions against each other in 2D, though...
                # square grid where rows will be one dim and cols the other
                # NOTE: there won't be a start/end distinction syoo... hope that
                # doesn't break things
                sqSize = np.minimum(6, xDimBest)
                figDimsVsEachOther, axDimsVsEachOther = plt.subplots(sqSize, sqSize)
                figDimsVsEachOther.tight_layout()
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
            trainInds = [sq['trialId'] for sq in dimResult[xDimScoreBest]['seqsTrainNew'][cValUse]]
            mxInd = np.max(np.hstack([trainInds, testInds[cValUse]]))
            for k, (sq, tstInd) in enumerate(zip(seqTestUse,testInds[cValUse])):
                if 'xorth' not in sq:
                    # this feels kind of dangerous... but I think the only time
                    # 'xorth' won't exist is if this is FA and the
                    # orthonormalized dimensions were saved in xsm. I think?
                    sq['xorth'] = sq['xsm']
                    # continue # doing this instead of wrapping all the for loop in an if statement >.>
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
                    
                # 3D plots
                if xDimBest>2:
                    plt.figure(figTraj.number)
                    if tmValsStart.size:
                        plotDimensionsAgainstEachOther(axStartTraj, sq['xorth'][:3], tmValsStart, [0,tmValsStart.shape[0]], lblStartTraj, lblEndTraj, [lblDelayStart, lblNoDelayStart], colorset, idx, axTtl = 'Start', axLabel = ['gpfa 1', 'gpfa 2', 'gpfa 3'], colorAlignPts = 'green')
                        axStartTraj.legend()
                    
                    if tmValsEnd.size:
                        plotDimensionsAgainstEachOther(axEndTraj, sq['xorth'][:3], tmValsEnd, [-tmValsEnd.shape[0],sq['xorth'].shape[1]], lblStartTraj, lblEndTraj, [lblDelayEnd, lblNoDelayEnd], colorset, idx, axTtl = 'End', axLabel = ['gpfa 1', 'gpfa 2', 'gpfa 3'], colorAlignPts = 'red')
                        axEndTraj.legend()
                        
                    if tmValsStart.size and tmValsEnd.size:
                        # the binSize and start are detailed here, so we can find the rest of time
                        lenT = sq['xorth'].shape[1]
                        allTimesAlignStart = np.arange(tmValsStart[0], tmValsStart[0]+binSize*lenT, binSize)
                        # gotta play some with the end to make sure 0 is aligned from the end as in tmValsEnd
                        allTimesAlignEnd = np.arange(tmValsEnd[-1]-binSize*(lenT-1), tmValsEnd[-1]+binSize/10, binSize)
                        
                        plotDimensionsAgainstEachOther(axAllTraj, sq['xorth'][:3], [allTimesAlignStart, allTimesAlignEnd], [0,sq['xorth'].shape[1]], lblStartTraj, lblEndTraj, [[lblDelayStart, lblNoDelayStart],[lblDelayEnd, lblNoDelayEnd]], colorset, idx, axTtl = 'All', axLabel = ['gpfa 1', 'gpfa 2', 'gpfa 3'], colorAlignPts = ['green', 'red'])

                    # plot all dimensions vs each other in addition to the 3D
                    # plot
                    # plt.figure(figDimsVsEachOther.number)
                    from itertools import product
                    dimsVsDim = product(np.arange(sqSize), np.arange(sqSize))
                    for dim1, dim2 in dimsVsDim:
                        # turns out at the these can also be used to index the axes...
                        if dim2 > dim1:
                            plotDimensionsAgainstEachOther(axDimsVsEachOther[dim1, dim2], sq['xorth'][[dim1, dim2]], tmValsStart, [0, tmValsStart.shape[0]], None, None, [None, None], colorset, idx, axTtl = 'Dims', axLabel=['gpfa {}'.format(dim1), 'gpfa {}'.format(dim2)], colorAlignPts = 'green')
                    figDimsVsEachOther.tight_layout()

                # 2D plots
                elif xDimBest>1:
                    plt.figure(figTraj.number)
                    if tmValsStart.size:
                        plotDimensionsAgainstEachOther(axStartTraj, sq['xorth'][:2], tmValsStart, [0,tmValsStart.shape[0]], lblStartTraj, lblEndTraj, [lblDelayStart, lblNoDelayStart], colorset, idx, axTtl = 'Start', axLabel = ['gpfa 1', 'gpfa 2'], colorAlignPts = 'green')

                    
                    if tmValsEnd.size:
                        plotDimensionsAgainstEachOther(axEndTraj, sq['xorth'][:2], tmValsEnd, [-tmValsEnd.shape[0],sq['xorth'].shape[1]], lblStartTraj, lblEndTraj, [lblDelayEnd, lblNoDelayEnd], colorset, idx, axTtl = 'End', axLabel = ['gpfa 1', 'gpfa 2'], colorAlignPts = 'red')

                    
                    if tmValsStart.size and tmValsEnd.size:
                        # the binSize and start are detailed here, so we can find the rest of time
                        lenT = sq['xorth'].shape[1]
                        allTimesAlignStart = np.arange(tmValsStart[0], tmValsStart[0]+binSize*lenT, binSize)
                        # gotta play some with the end to make sure 0 is aligned from the end as in tmValsEnd
                        allTimesAlignEnd = np.arange(tmValsEnd[-1]-binSize*(lenT-1), tmValsEnd[-1]+binSize/10, binSize)
                        
                        plotDimensionsAgainstEachOther(axAllTraj, sq['xorth'][:2], [allTimesAlignStart, allTimesAlignEnd], [0,sq['xorth'].shape[1]], lblStartTraj, lblEndTraj, [[lblDelayStart, lblNoDelayStart],[lblDelayEnd, lblNoDelayEnd]], colorset, idx, axTtl = 'All', axLabel = ['gpfa 1', 'gpfa 2'], colorAlignPts = ['green', 'red'])


                
                # plot dimension vs time!
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
                    # plot start of time segment
                    if tmValsStart.size:
                        axUsed = plotDimensionsAgainstTime(figSep, pltNum, pltListNum, axesStart, axesEnd, rowsPlt, colsPlt, tmValsStart, dim[:tmValsStart.shape[0]], dimNum, xDimBest, timeLineColor, lblNeuralTraj, axTitle = 'Start')
                    
                    
                        axVals = np.append(axVals, np.array(axUsed.axis())[None, :], axis=0)
                        pltNum += 1
                    
                    # plot end of time segment
                    if tmValsEnd.size:
                        axUsed = plotDimensionsAgainstTime(figSep, pltNum, pltListNum, axesEnd, axesStart, rowsPlt, colsPlt, tmValsEnd, dim[-tmValsEnd.shape[0]:], dimNum, xDimBest, timeLineColor, lblNeuralTraj, axTitle = 'End')

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
                        
                            axVals = np.append(axVals, np.array(axUsed.axis())[None, :], axis=0)
                            pltNum += 1
                        
                        if tmValsEnd.size:
                            axUsed = plotDimensionsAgainstTime(figSep, pltNum, pltListNum, axesEnd, axesStart, rowsPlt, colsPlt, tmValsEnd, dim[-tmValsEnd.shape[0]:], dimNum, xDimBest, colorUse = [0.5,0.5,0.5], labelTraj = lblShuffle, linewidth=0.2, alpha = 0.5, axTitle = 'End')

                            axVals = np.append(axVals, np.array(axUsed.axis())[None, :], axis=0)
                            pltNum += 1
                        
                        pltListNum += 1

                    axUsed.legend()
                    lblShuffle = None


            # ** These are computations of slow drift **
            # Doing here because the PC is needed for the overlap calculation 
            plotSlowDrift = False
            axesOverTime = []
            axesOverTimePC = []
            if plotSlowDrift:
                figOverTime = plt.figure()
                figOverTime.suptitle(description + " cond " + str(condLabel) + "")

                figOverTimePC = plt.figure()
                figOverTimePC.suptitle(description + " cond " + str(condLabel) + "")

                minTime = 0 # ms
                binSizeForAvg = 500 # ms
                maxTime = binSizeForAvg * np.ceil(trlTimes.max()/binSizeForAvg) # ms
                binnedSpikesOverSession, binEdges, binNum = sp.stats.binned_statistic(trlTimes, spikesUsedNorm.T, statistic='sum', bins=np.arange(minTime, maxTime+binSizeForAvg/2, binSizeForAvg))
                maskRecordedVals = np.zeros_like(binnedSpikesOverSession[0], dtype='bool')
                maskRecordedVals[binNum-1] = True # binNum is 1-indexed for these purposes...
                windowSizeForMeanMinutes = 20 # minutes
                windowSizeForMeanInBinSizes = np.round(windowSizeForMeanMinutes*60*1000/binSizeForAvg).astype(int)
                boxAverageFilter = np.ones(windowSizeForMeanInBinSizes, dtype=int)
                maskedVals = np.where(maskRecordedVals, binnedSpikesOverSession, 0)
                numValsExist = np.convolve(maskRecordedVals, boxAverageFilter, mode='valid')
                boxedAvgOut = [np.convolve(spkChan, boxAverageFilter, mode='valid')/numValsExist for spkChan in maskedVals]
                firstFullVal = boxAverageFilter.shape[0]
                binsToUse = np.sort(binNum[binNum>firstFullVal])-firstFullVal
                binnedSpikeRunningAvgOverSession = np.stack(boxedAvgOut)[:, binsToUse]
                binnedSpikeRunningAvgOverSession = binnedSpikeRunningAvgOverSession.T
                pcs, evalsPca, _ = np.linalg.svd(np.cov(binnedSpikeRunningAvgOverSession.T,ddof=0))
                slowDriftDim = 0
                pcSlowDrift = pcs[:, [slowDriftDim]]
                # ** slow drift computations over **

                # plot dimension over trials!
                pltNum = 1
                plt.figure(figOverTime.number)
                figOverTime.tight_layout()
                # if 'xorth' in seqTestUse[0]:
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

                for dimNum, dimSeqs in enumerate(allSeqsByDim):
                    trlInds = testInds[cValUse]
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
                        varExpBySlowDrift = -0.01
                    plotAnnot = {
                        '%sv' : 100*latentSv,
                        'R^2' : r2latent[0],
                        '%latent var exp by slow drift pc' : 100*varExpBySlowDrift,
                    }
                    colorUse = [0.5,0.5,0.5]

                    dimNum = dimNum+1 # for the plot, 1 is 1-dimensional, not zero-dimensinoal
                    axUsed = plotDimensionsOverTrials(figOverTime, pltNum, axesOverTime, rowsPlt, colsOverTimePlt, dimSeqs, trlInds, trlTimes, binSize, dimNum, xDimBest, colorUse, linearFit = linearFit, plotAnnot = plotAnnot, linewidth=0.2, alpha = 0.5)
                    axVals = np.append(axVals, np.array(axUsed.axis())[None, :], axis=0)
                    pltNum += 1

                plt.figure(figOverTimePC.number)
                figOverTimePC.tight_layout()


                numPCUse = 4
                colsOverTimePltPC = np.ceil(numPCUse/rowsPlt)
                spikesProjNorm = [pcs.T @ sp[:, None]  for sp in spikesUsedNorm[trlInds]]
                spikesProjNormByDim = list(zip(*spikesProjNorm))
                dimsExpVar = evalsPca/evalsPca.sum()
                totalVarTrls = np.cov(spikesUsedNorm.T, ddof=0).trace()
                dimsExpVarTrls = np.array([np.array(spProj).var() for spProj in spikesProjNormByDim])/totalVarTrls


                pltNum = 1
                trajTms = np.hstack(trlTimes[trlInds])
                trlInds = testInds[cValUse]
                intercept = np.ones_like(trajTms)
                coeffs = np.vstack([trajTms,intercept]).T
                numOverSessPcPlot = 4
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
                    except ValueError:
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
                figOverTimePC.tight_layout()

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
                        
                            axVals = np.append(axVals, np.array(axUsed.axis())[None, :], axis=0)
                            pltNum += 1
                        
                        if tmValsEnd.size:
                            axUsed = plotDimensionsAgainstTime(figSep, pltNum, pltListNum, axesEnd, axesStart, rowsPlt, colsPlt, tmValsEnd, dim[-tmValsEnd.shape[0]:], dimNum, xDimBest, colorset[condNum,:], lblMn, linewidth=1, axTitle = 'End')

                            axVals = np.append(axVals, np.array(axUsed.axis())[None, :], axis=0)
                            pltNum += 1
                            
                        pltListNum += 1
                    axUsed.legend() #legend on the last plot...
                    lblMn = None
            
            # if there are no axVals, probably none of the below plots were plotted...
            if axVals.size:
                ymin = np.min(np.append(0, np.min(axVals, axis=0)[2]))
                ymax = np.max(axVals, axis=0)[3]
                for ax in axesStart:
                    ax.set_ylim(bottom = ymin, top = ymax )
                    ax.axvline(x=0, linestyle=':', color='black')
                    ax.axhline(y=0, linestyle=':', color='black')
    #                        xl = ax.get_xlim()
    #                        yl = ax.get_ylim()
    #                        ax.axhline(y=0,xmin=xl[0],xmax=xl[1], linestyle = ':', color='black')
    #                        ax.axvline(x=0,ymin=xl[0],ymax=xl[1], linestyle = ':', color='black')
                    ax.spines['right'].set_visible(False)
                    ax.spines['top'].set_visible(False)
                    
                for ax in axesEnd:
                    ax.set_ylim(bottom = ymin, top = ymax )
                    ax.axvline(x=0, linestyle=':', color='black')
                    ax.axhline(y=0, linestyle=':', color='black')
    #                        xl = ax.get_xlim()
    #                        yl = ax.get_ylim()
    #                        ax.axhline(y=0,xmin=xl[0],xmax=xl[1], linestyle = ':', color='black')
    #                        ax.axvline(x=0,ymin=xl[0],ymax=xl[1], linestyle = ':', color='black')
                    ax.spines['right'].set_visible(False)
                    ax.spines['top'].set_visible(False)

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

def projectTrajectory(estParams, trajectory):
    from classes import GPFA
    seqTrajDict = GPFA.binSpikesToGpfaInputDict([], binnedSpikes = trajectory)
    from multiprocessing import Pool
    with Pool() as poolHere:
        res = []

        res.append(poolHere.apply_async(GPFA.projectTrajectory, (seqTrajDict,estParams)))

        resultsTraj = [rs.get() for rs in res]

    return resultsTraj[0]
