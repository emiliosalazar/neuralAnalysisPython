"""
Here I will have methods for plotting binned spike sets, so they can be separate from codey-code
"""
from methods.BinnedSpikeSetListMethods import generateLabelGroupStatistics as genBSLLabGrp
from matplotlib import pyplot as plt
import numpy as np

def plotResponseOverTime(binnedSpikes, datasetNames, plotTypeInfo, chPlot = None):
    plotSegments = plotTypeInfo['plotSegments'] if 'plotSegments' in plotTypeInfo else True
    binsAroundAlign = plotTypeInfo['ptsAroundAlBin'] if 'ptsAroundAlBin' in plotTypeInfo else 4
    plotMethod = plotTypeInfo['plotMethod'] if 'plotMethod' in plotTypeInfo else 'plot'
    
    if chPlot is not None and len(chPlot) != len(binnedSpikes):
        raise Exception("One channel list per binned spike set must be provided if specifying channels to plot")

    for idx, (bnSp, dsName) in enumerate(zip(binnedSpikes, datasetNames)):
        numAlignmentBins = bnSp.alignmentBins[0].shape[0]
        numCols = 3
        numRows = 3
        stateName = 'Delay'
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
                        bnSpSegB4Avg = np.stack([bnSp[trl:trl+1,:,:aB].timeAverage() for trl, aB in enumerate(alBins)])
                    else:
                        alBinsB4 = bnSp.alignmentBins[:, seg-1].astype(int)
                        bnSpSegB4Avg = np.stack([bnSp[trl:trl+1,:,aBb4:aB].timeAverage() for trl, (aBb4, aB) in enumerate(zip(alBinsB4, alBins))])

                else:
                    alBins = np.unique(bnSp.alignmentBins, axis=0)
                    if alBins.shape[0]>1:
                        raise Exception("Do the same as you would for an object dtype here...")
                    else:
                        alB = alBins[seg]
                    segSt = np.max(alB-binsAroundAlign, 0) # can't go less than the bss start heh...
                    segEnd = np.min(alB+binsAroundAlign, bnSp.shape[2]) # can't go over bss end...
                    bnSpSeg = bnSp[:, :, segSt:segEnd]
                    if seg == 0:
                        bnSpSegB4Avg = bnSp[:,:,:alB].timeAverage()
                    else:
                        alBb4 = alBins[seg-1].astype(int)
                        bnSpSegB4Avg = bnSp[:,:,alBb4:alB].timeAverage()

                grpSpkTrlAvgSemHere, groupedSpikesHere, grpLabels = genBSLLabGrp([bnSpSeg], labelUse='stimulusMainLabel')
                grpSpkTrlAvgSem.append(grpSpkTrlAvgSemHere[0])
                groupedSpikes.append(groupedSpikesHere[0])
#                zeroBins.append(binsAroundAlign)
                if seg==0:
                    zeroBins.append(alBins[0])
                else:
                    zeroBins.append(binsAroundAlign)

                grpSpkTrlAvgSemAll, _, _ = genBSLLabGrp([bnSpSegB4Avg], labelUse='stimulusMainLabel')
                chanTmAvgs.append(grpSpkTrlAvgSemAll[0][0])

            if bnSp.dtype=="object":
                # NOTE: the trl:trl+1 prevents squashing the first
                # dimension...
                bnSpSegAftAvg = np.stack([bnSp[trl:trl+1,:,aB:].timeAverage() for trl, aB in enumerate(alBins)])
            else:
                bnSpSegAftAvg = bnSp[:,:,alB:].timeAverage()
            grpSpkTrlAvgSemAll, _, _ = genBSLLabGrp([bnSpSegAftAvg], labelUse='stimulusMainLabel')
            chanTmAvgs.append(grpSpkTrlAvgSemAll[0][0])
        else:
            # note that this pops out as a list from the function, much
            # like the main if, so we don't need to change anything!
            numAlignmentBins = 1
            zeroBins = bnSp.alignmentBins[0]
            grpSpkTrlAvgSem, groupedSpikes, grpLabels = genBSLLabGrp([bnSp], labelUse='stimulusMainLabel')

        grpLabels = grpLabels[0]
        if chPlot is not None:
            chans = chPlot[idx]
        else:
            chans = np.arange(bnSp.shape[1])
        binSizeMs = bnSp.binSize
        for chan in chans:
            # Prep figure
            plt.figure()
            plt.suptitle(dsName + ': channel ' + str(chan))
            grpLabels = grpLabels.astype('float64')
            if grpLabels.astype('float64').max()>2*np.pi:
                sbpltAngs = np.arange(135, -225, -45)
                modulusFullRot = 360
                grpLblPlotFactor = np.pi/180
            else:
                sbpltAngs = np.arange(3*np.pi/4, -5*np.pi/4, -np.pi/4)
                modulusFullRot = 2*np.pi
                grpLblPlotFactor = 1
            # Add nan to represent center for tuning polar curve...
            sbpltAngs = np.concatenate((sbpltAngs[0:3], sbpltAngs[[-1]], np.expand_dims(np.asarray(np.nan),axis=0), sbpltAngs[[3]], np.flip(sbpltAngs[4:-1])), axis=0)
            axVals = np.empty((0,4))
            axes = []

            # we'll have this be based on the entire period... at least
            # for now; also we time average before splitting into
            # different groups to allow for binned spikes with
            # different length trials to be used...
            grpSpkTrlAvgSemAll, _, _ = genBSLLabGrp([bnSp.timeAverage()], labelUse='stimulusMainLabel')
            chanTmAvg = grpSpkTrlAvgSemAll[0][0][:,[chan]]
            for alB in range(numAlignmentBins):
                # Prep parameters to plot
                grpSpkTrlAvgSemHere = grpSpkTrlAvgSem[alB]
                chanRespMean = np.squeeze(grpSpkTrlAvgSemHere[0][:,[chan]])
                chanRespSem = np.squeeze(grpSpkTrlAvgSemHere[1][:,[chan]])

                colSt = bnSp.colorset[alB, :]
                colEnd = bnSp.colorset[alB+1, :]

                for idx in range(0, len(grpLabels)):
                    chanSpkBinsByTrial = groupedSpikes[alB][idx][:,chan]
                    subplotChooseCond = np.where([np.allclose(grpLabels.astype('float64')[idx] % (modulusFullRot), sbpltAngs[i] % (modulusFullRot)) for i in range(0,len(sbpltAngs))])[0]

                    if not subplotChooseCond.size:
                        subplotChooseCond = np.where(sbpltAngs==modulusFullRot/2*(grpLabels[idx]-2))[0]
                    subplotChoose = subplotChooseCond[0]*numAlignmentBins+alB
                    numColsAll = numCols*numAlignmentBins if plotSegments else 3
                    axes.append(plt.subplot(numRows, numColsAll, subplotChoose+1))
                    
                    numTp = chanRespMean.shape[1]
#                        tmValStart = np.arange(timeBeforeAndAfterStart[0]+binSizeMs/2, timeBeforeAndAfterStart[1]+binSizeMs/2, binSizeMs)
                    binAll = np.arange(numTp)
                    zeroBin = zeroBins[alB]
                    tmVals = (binAll - zeroBin)*binSizeMs

                    if plotMethod == 'plot':
                        # plot pre-alignment bin
                        plt.plot(tmVals[:zeroBin+1], chanRespMean[idx, :zeroBin+1], color=colSt)
                        plt.fill_between(tmVals[:zeroBin+1], chanRespMean[idx, :zeroBin+1]-chanRespSem[idx,:zeroBin+1], chanRespMean[idx, :zeroBin+1]+chanRespSem[idx,:zeroBin+1], alpha=0.2, color=colSt)

                        # plot post-alignment bin
                        plt.plot(tmVals[zeroBin:], chanRespMean[idx, zeroBin:], color=colEnd)
                        plt.fill_between(tmVals[zeroBin:], chanRespMean[idx, zeroBin:]-chanRespSem[idx,zeroBin:], chanRespMean[idx, zeroBin:]+chanRespSem[idx,zeroBin:], alpha=0.2, color=colEnd)
                    elif plotMethod == 'eventplot':
                        # plot pre-alignment bin
                        chPltEvts = [np.where(ch)[0]-zeroBin for ch in chanSpkBinsByTrial[:, :zeroBin+1]]
                        plt.eventplot(chPltEvts, color=colSt)

                        # plot post-alignment bin
                        chPltEvts = [np.where(ch)[0] for ch in chanSpkBinsByTrial[:, zeroBin:]]
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
                plt.subplot(numRows, numCols, np.where(np.isnan(sbpltAngs))[0][0]+1, projection='polar')
                ptch = plt.fill(grpLabels*grpLblPlotFactor, chanTmAvgs[alB][:,[chan]])
                ptch[0].set_fill(False)
                ptch[0].set_edgecolor(colSt)

            # plot the average after the last bin
            plt.subplot(numRows, numCols, np.where(np.isnan(sbpltAngs))[0][0]+1, projection='polar')
            ptch = plt.fill(grpLabels*grpLblPlotFactor, chanTmAvgs[alB+1][:,[chan]])
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
            plt.subplot(numRows, numCols, np.where(np.isnan(sbpltAngs))[0][0]+1, projection='polar')
            ptch = plt.fill(grpLabels*grpLblPlotFactor, chanTmAvg)
            ptch[0].set_fill(False)
            ptch[0].set_edgecolor('k')
