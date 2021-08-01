"""
For plot methods I don't know where to put... may this list be short and in
flux as I move thing to their final destinations...
"""
import numpy as np
from classes.BinnedSpikeSet import BinnedSpikeSet
from methods.plotMethods.ScatterBar import scatterBar

from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from itertools import product,chain

from methods.plotMethods.LegendUtils import TransitionHandler

def plotAllVsAll(descriptions, metricDict, labelForCol, labelForMarker, supTitle=""):
    colorsUse = BinnedSpikeSet.colorset
    # this variable will use matplotlib's maker of markers to combine polygon
    # numbers and angles
    polygonSides = range(2,8)
    rotDegBtShapes = 20
    # 0 is to make this a polygon
    polyAng = [product([polySd], [0], np.arange(0, np.lcm(np.int(np.round(360/polySd)), rotDegBtShapes) if np.lcm(np.int(np.round(360/polySd)), rotDegBtShapes) < 360 else 360, rotDegBtShapes)) for polySd in polygonSides]
    markerCombos = np.stack(chain.from_iterable(polyAng))
    # sort by angle--this ends up ensuring that sequential plots have different
    # polygons
    markerCombos = markerCombos[markerCombos[:,2].argsort()]
    # back in tuple form
    markerCombos = tuple(tuple(mc) for mc in markerCombos)
    markerCombos = computeMarkerCombos()
    for metricNum, (metricName, metricVal) in enumerate(metricDict.items()):
        for metric2Num, (metric2Name, metric2Val) in enumerate(metricDict.items()):
            try:
                mVlConc = np.concatenate(metricVal, axis=0)
            except ValueError:
                mVlConc = np.concatenate(metricVal, axis=1)

            try:
                mVl2Conc = np.concatenate(metric2Val, axis=0)
            except ValueError:
                mVl2Conc = np.concatenate(metric2Val, axis=1)

            if mVlConc.size != mVl2Conc.size:
                # some small cases have values broken down even more than just
                # by-condition, so they don't line up well with the
                # by-condition ones; I'm skipping these for now
                pass
            elif metric2Num > metricNum:
                plt.figure()
                plt.title('%s vs %s' % ( metricName, metric2Name ))
                plt.suptitle(supTitle)
                
                unLabForCol, colNum = np.unique(labelForCol, return_inverse=True, axis=0)
                unLabForTup, mcNum = np.unique(labelForMarker, return_inverse=True, axis=0)
                count = np.unique(labelForMarker,return_counts=1)[1]
                angle = grp_range(count)[np.argsort(labelForMarker).argsort()]

                if (metricName.find('factor load') != -1 and metric2Name.find('%sv') != -1) or (metric2Name.find('factor load') != -1 and metricName.find('%sv') != -1):
                    ax = plt.subplot(1,1,1, projection='polar')
                    if metricName.find('factor load') != -1:
                        metricValTh = [np.pi/2 - mV/1*np.pi/2 for mV in metricVal] # stretching it from 0-1 to be pi/2-0 (so the angle)
                        metricValR = metric2Val

                        metricNameTh = metricName
                        metricNameR = metric2Name
                    elif metric2Name.find('factor load') != -1:            
                        metricValTh = [np.pi/2 - mV/1*np.pi/2 for mV in metric2Val] # stretching it from 0-1 to be pi/2-0 (so the angle)
                        metricValR = metricVal

                        metricNameTh = metric2Name
                        metricNameR = metricName
                    
                    # breakpoint()
                    scPts = [ax.scatter(m1, m2, label=desc, color=colorsUse[colN,:], marker=markerCombos[mcN]) for m1, m2, desc, colN, mcN in zip(metricValTh, metricVal, descriptions, colNum, mcNum)]

                else:        
                    ax=plt.subplot(1,1,1)
                    scPts = [ax.scatter(m1, m2, label=desc, color=colorsUse[colN,:], marker=markerCombos[mcN]) for m1, m2, desc, colN, mcN in zip(metricVal, metric2Val, descriptions, colNum, mcNum)]

                if len(metricVal) > 2*unLabForCol.shape[0]:
                    colLegElem = [Patch(facecolor=colorsUse[colN, :], label=unLC) for colN, unLC in enumerate(unLabForCol)]
                else:
                    colLegElem = []

                if len(metricVal) > 2*unLabForTup.shape[0]:
                    colGray = [0.5,0.5,0.5]
                    mrkLegElem = [Line2D([0], [0], marker=markerCombos[mcN], label=unLM, color='w', markerfacecolor=colGray, markersize=7) for mcN, unLM in enumerate(unLabForTup)]
                else:
                    mrkLegElem = []

                if metricName.find('r_{sc}') != -1 or metricName.find('%sv') != -1 or metricName.find('sh pop') != -1:
                    minMet1Val = np.nanmin(mVlConc)
                    maxMet1Val = np.nanmax(mVlConc)

                    if metric2Name.find('r_{sc}') != -1 or metric2Name.find('%sv') != -1 or metric2Name.find('sh pop') != -1:
                        minMet2Val = np.nanmin(mVl2Conc)
                        maxMet2Val = np.nanmax(mVl2Conc)
                        min2Val = np.array([minMet2Val, 0]).min()

                        minVal = np.array([minMet1Val, minMet2Val, 0]).min()
                        maxVal = np.array([maxMet1Val, maxMet2Val]).max()
                        breathingRoom = 0.1*(maxVal-minVal)
                        ax.set_xlim(left=minVal - breathingRoom, right=maxVal + breathingRoom)
                        ax.set_ylim(bottom=minVal - breathingRoom, top=maxVal + breathingRoom)
                        ax.axhline(0, color='black', linestyle='--')
                        ax.axvline(0, color='black', linestyle='--')
                
                if hasattr(ax, 'PolarTransform'):
                    minMetRVal = np.nanmin(np.hstack(metricValR))
                    maxMetRVal = np.nanmax(np.hstack(metricValR))
                    breathingRoom = 0.1*(maxMetRVal-minMetRVal)
                    ax.set_rlim(minMetRVal, maxMetRVal+breathingRoom)
                    ax.set_thetalim(0, np.pi/2)
                    _, labels = ax.set_thetagrids(np.linspace(0, 90, 3, endpoint=True), labels=[1,metricNameTh,0])
                    labels[1].set_rotation(-45)
                    # ax.set_xtick_labels([0,0.5,1])
                    ax.set_xlabel(metricNameR)
                else:
                    ax.set_xlabel(metricName)
                    ax.set_ylabel(metric2Name)
                axBx = ax.get_position()
                ax.set_position([axBx.x0, axBx.y0, axBx.width * 0.8, axBx.height])
#                ax.legend(colLegElem + scPts, [elm.get_label() for elm in colLegElem+scPts],prop={'size':5},loc='center left', bbox_to_anchor=(1, 0.5))
                lgGen = ax.legend(handles=colLegElem + mrkLegElem,prop={'size':5},loc='upper left', bbox_to_anchor=(1, 1))
                ax.legend(handles=scPts,prop={'size':5},loc='lower left', bbox_to_anchor=(1, 0))
                ax.add_artist(lgGen)

def plotMetricVsExtractionParams(descriptions, metricDict, splitNameDesc, labelForPair, labelForSplit, labelForCol, labelForMarker, supTitle=""):
    colorsUse = BinnedSpikeSet.colorset
    try:
        splitNmPart, colSplitTemp = np.unique(labelForSplit, return_inverse=True, axis=0)
        unLabForSplit, colSplit = np.unique(colSplitTemp, return_inverse=True, axis=0)
        splitNames = [splitNameDesc + str(sNT) for sNT in splitNmPart]
    except TypeError:
        # allows us to use string objects for the label here... (note that the
        # above allows us to use string *lists*... don't ask me why np.unique
        # doesn't work well on object arrays of strings, it's dumb
        splitNmPart, colSplitTemp = np.unique(labelForSplit, return_inverse=True)
        unLabForSplit, colSplit = np.unique(colSplitTemp, return_inverse=True) # converts the strings in unLabForSplit to integers!
        splitNames = [splitNameDesc + str(sNT) for sNT in splitNmPart]

    try:
        unLabForPair, colPair, nmPerPair = np.unique(labelForPair, return_inverse=True, return_counts=True, axis=0)
    except TypeError:
        # allows us to use string objects for the label here... (note that the
        # above allows us to use string *lists*... don't ask me why np.unique
        # doesn't work well on object arrays of strings, it's dumb
        pairNames, colPairTemp = np.unique(labelForPair, return_inverse=True)
        unLabForPair, colPair, nmPerPair = np.unique(colPairTemp, return_inverse=True, return_counts=True) # converts the strings in unLabForSplit to integers!

    if np.any(nmPerPair>2):
        raise Exception("Splits need to be PAIRED or SOLO--I've found at least one 'pair' that comes with more than two inputs!")

    pairItemSplit1 = colPair[colSplit == unLabForSplit[0]]
    pairItemSplit2 = colPair[colSplit == unLabForSplit[1]]

    if unLabForSplit.size!=2:
        raise Exception("Currently can only plot a 2-way split!")
    if np.sum(colSplit==unLabForSplit[0]) != np.sum(colSplit==unLabForSplit[1]):
        print("Some values are not paired! Putting those along the axes...")

    # here we're kind of assuming everything is organized in order... which...
    # may not be the most correct assumption, hmm...
#    descriptions = list(np.array(descriptions)[colSplit==unLabForSplit[0]])
#    labelForCol = labelForCol[colSplit==unLabForSplit[0]]
#    labelForMarker = labelForMarker[colSplit==unLabForSplit[0]]
    
#    from itertools import product,chain
#    # this variable will use matplotlib's maker of markers to combine polygon
#    # numbers and angles
#    polygonSides = range(3,8)
#    rotDegBtShapes = 20
#    # 0 is to make this a polygon
#    polyAng = [product([polySd], [0], np.arange(0, np.lcm(np.int(np.round(360/polySd)), rotDegBtShapes) if np.lcm(np.int(np.round(360/polySd)), rotDegBtShapes) < 360 else 360, rotDegBtShapes)) for polySd in polygonSides]
#    markerCombos = np.stack(chain.from_iterable(polyAng))
#    # sort by angle--this ends up ensuring that sequential plots have different
#    # polygons
#    markerCombos = markerCombos[markerCombos[:,2].argsort()]
#    # back in tuple form
#    markerCombos = tuple(tuple(mc) for mc in markerCombos)
    markerCombos = computeMarkerCombos()

    for metricNum, (metricName, metricVal) in enumerate(metricDict.items()):
        metricVal = [np.array(m).squeeze() for m in metricVal]
        metricValSplit1Init = list(np.array(metricVal)[colSplit==unLabForSplit[0]])
        splitName1 = splitNames[unLabForSplit[0]]
        metricValSplit2Init = list(np.array(metricVal)[colSplit==unLabForSplit[1]])
        splitName2 = splitNames[unLabForSplit[1]]

#        if len(metricValSplit1Init) > len(metricValSplit2Init):
#            metricValSplit1 = metricValSplit1Init
#            metricValSplit2 = [metricValSplit2Init[(pairItemSplit2 == pI1).nonzero()[0][0]] if np.any(pairItemSplit2 == pI1) else np.full_like(metricValSplit1[(pairItemSplit1==pI1).nonzero()[0][0]], np.nan, dtype=np.double) for pI1 in pairItemSplit1]
#            labelForColSp = labelForCol[colSplit==unLabForSplit[0]]
#            labelForMarkerSp = labelForMarker[colSplit==unLabForSplit[0]]
#        else:
#            metricValSplit2 = metricValSplit2Init
#            metricValSplit1 = [metricValSplit1Init[(pairItemSplit1 == pI2).nonzero()[0][0]] if np.any(pairItemSplit1 == pI2) else np.full_like(metricValSplit2[(pairItemSplit2==pI2).nonzero()[0][0]], np.nan, dtype=np.double) for pI2 in pairItemSplit2]
#            labelForColSp = labelForCol[colSplit==unLabForSplit[1]]
#            labelForMarkerSp = labelForMarker[colSplit==unLabForSplit[1]]
#
        grpSplits, grpSort = splitMetricVals(metricVal, colSplit, unLabForSplit, [pairItemSplit1,pairItemSplit2])
        labelForColSp = np.array(labelForCol)[colSplit==unLabForSplit[grpSort]]
        labelForMarkerSp = np.array(labelForMarker)[colSplit==unLabForSplit[grpSort]]
        descriptionsSp = np.array(descriptions)[colSplit==unLabForSplit[grpSort]]
        metricValSplit1 = grpSplits[0]
        metricValSplit2 = grpSplits[1]
        
        plt.figure()
        plt.title('{}'.format( metricName))
        supTitleAll = supTitle + ' {} vs {}'.format( splitNames[0], splitNames[1] )
        plt.suptitle(supTitleAll)
        ax=plt.subplot(1,1,1)
        unLabForCol, colNum = np.unique(labelForColSp, return_inverse=True, axis=0)
        unLabForTup, mcNum = np.unique(labelForMarkerSp, return_inverse=True, axis=0)

        scPts = [ax.scatter(m1, m2, label=desc, color=colorsUse[colN,:], marker=markerCombos[mcN]) if m1.size==m2.size else None for m1, m2, desc, colN, mcN in zip(metricValSplit1, metricValSplit2, descriptionsSp, colNum, mcNum)]

        # remove what effectively wasn't plotted
        scPts = list(np.array(scPts)[np.array(scPts)!=None])

        ax.set_xlabel(splitName1)
        ax.set_ylabel(splitName2)

        if len(metricValSplit1) > 2*unLabForCol.shape[0]:
            colLegElem = [Patch(facecolor=colorsUse[colN, :], label=unLC) for colN, unLC in enumerate(unLabForCol)]
        else:
            colLegElem = []

        if len(metricValSplit1) > 2*unLabForTup.shape[0]:
            colGray = [0.5,0.5,0.5]
            mrkLegElem = [Line2D([0], [0], marker=markerCombos[mcN], label=unLM, color='w', markerfacecolor=colGray, markersize=7) for mcN, unLM in enumerate(unLabForTup)]
        else:
            mrkLegElem = []

        # make space for a legend to the right of the plot
        axBx = ax.get_position()
        ax.set_position([axBx.x0, axBx.y0, axBx.width * 0.8, axBx.height])
        ax.axis('equal')
        
        # draw a dashed line representing y=x line
        axXlim = ax.get_xlim()
        axYlim = ax.get_ylim()
        newMin = np.min([axXlim[0],axYlim[0]])
        newMax = np.max([axXlim[1],axYlim[1]])
        ax.set_xlim(newMin,newMax)
        ax.set_ylim(newMin,newMax)
        ax.plot(np.array([newMin,newMax]), np.array([newMin,newMax]), 'k--')

        lgGen = ax.legend(handles=colLegElem + mrkLegElem,prop={'size':5},loc='upper left', bbox_to_anchor=(1, 1))
        ax.legend(handles=scPts,prop={'size':5},loc='lower left', bbox_to_anchor=(1, 0))
        ax.add_artist(lgGen)


def grp_range(a):
    idx = a.cumsum()
    id_arr = np.ones(idx[-1],dtype=int)
    id_arr[0] = 0
    id_arr[idx[:-1]] = -a[:-1]+1
    return id_arr.cumsum()

def computeMarkerCombos():
# this variable will use matplotlib's maker of markers to combine polygon
# numbers and angles
    polygonSides = range(3,8)
    rotDegBtShapes = 20
# 0 is to make this a polygon
    polyAng = [product([polySd], [0], np.arange(0, np.lcm(np.int(np.round(360/polySd)), rotDegBtShapes) if np.lcm(np.int(np.round(360/polySd)), rotDegBtShapes) < 360 else 360, rotDegBtShapes)) for polySd in polygonSides]
    markerCombos = np.stack(chain.from_iterable(polyAng))
# sort by angle--this ends up ensuring that sequential plots have different
# polygons
    markerCombos = markerCombos[markerCombos[:,2].argsort()]
# back in tuple form
    markerCombos = tuple(tuple(mc) for mc in markerCombos)

    return markerCombos

def splitMetricVals(metricVal, colSplit, unLabForSplit, groupItemSplit):
    metricVal = [np.array(m).flatten() for m in metricVal]
    metricValSplitInit = [list(np.array(metricVal)[colSplit==uLfS]) for uLfS in unLabForSplit]

    unGroupItemSplits = np.unique(groupItemSplit)
    whereUnGrp = [np.nonzero([np.any(gpSp==unG) for gpSp in groupItemSplit])[0][0] for unG in unGroupItemSplits]
    groupSplitLens = np.array([len(mVSI) for mVSI in metricValSplitInit])
    groupLargest = groupSplitLens.argmax()
    unGroupItemLargest = np.unique(groupItemSplit[groupLargest])
    if np.array_equal(unGroupItemLargest,unGroupItemSplits):
        metricValSplit = [[mVSplit[(gISplit == gI).nonzero()[0][0]] if np.any(gISplit == gI) else np.full_like(metricValSplitInit[groupLargest][(groupItemSplit[groupLargest]==gI).nonzero()[0][0]], np.nan, dtype=np.double) for gI in groupItemSplit[groupLargest]] for mVSplit, gISplit in zip(metricValSplitInit, groupItemSplit)]
        sortingOfGroups = groupLargest
    else:
        metricValSplit = [[mVSplit[(gISplit == gI).nonzero()[0][0]] if np.any(gISplit == gI) else np.full_like(metricValSplitInit[wUG][(groupItemSplit[wUG]==gI).nonzero()[0][0]], np.nan, dtype=np.double) for gI, wUG in zip(unGroupItemSplits, whereUnGrp)] for mVSplit, gISplit in zip(metricValSplitInit, groupItemSplit)]
        sortingOfGroups = whereUnGrp
        
#    if len(metricValSplit1Init) > len(metricValSplit2Init):
#        metricValSplit1 = metricValSplit1Init
#        metricValSplit2 = [metricValSplit2Init[(pairItemSplit2 == pI1).nonzero()[0][0]] if np.any(pairItemSplit2 == pI1) else np.full_like(metricValSplit1[(pairItemSplit1==pI1).nonzero()[0][0]], np.nan, dtype=np.double) for pI1 in pairItemSplit1]
#    else:
#        metricValSplit2 = metricValSplit2Init
#        metricValSplit1 = [metricValSplit1Init[(pairItemSplit1 == pI2).nonzero()[0][0]] if np.any(pairItemSplit1 == pI2) else np.full_like(metricValSplit2[(pairItemSplit2==pI2).nonzero()[0][0]], np.nan, dtype=np.double) for pI2 in pairItemSplit2]
#
    return metricValSplit, sortingOfGroups



def plotTimeEvolution(descriptions, timeShiftMetricDict, labelForMarkers, labelForColors, supTitle = ''):

    times = []

    timeShiftMetricMats = {}
    for tm, tmMetricDict in timeShiftMetricDict.items():
        times.append(tmMetricDict.pop('timeCenterPoint'))
        for metricName, metricVal in tmMetricDict.items():
            # I'm actually not sure what squeeze() does here... it must be
            # useful for some situations and I ported it from above
            metricVal = np.array([np.array(m).squeeze() if len(np.array(m).squeeze().shape)>0 else np.array(m).squeeze()[None] for m in metricVal])
            # metricVal = np.array([np.array(m) for m in metricVal])
            mxLen = np.max([m.shape[0] for m in metricVal])
            metricValNanPad = np.array([np.pad(m.astype('float'), ((0,mxLen-m.shape[0])), constant_values=(np.nan,)) for m in metricVal])
            if metricName in timeShiftMetricMats:
                currVals = timeShiftMetricMats[metricName]
                if currVals.shape[1] < metricValNanPad.shape[1]:
                    nanInit = np.zeros_like(currVals)*np.nan
                    numNans = metricValNanPad.shape[1] - currVals.shape[1]
                    nanAppend = nanInit[:, :numNans]
                    currVals = np.concatenate([currVals, nanAppend],axis=1)
                elif currVals.shape[1] > metricValNanPad.shape[1]:
                    nanInit = np.zeros_like(metricValNanPad)*np.nan
                    numNans = currVals.shape[1] - metricValNanPad.shape[1] 
                    nanAppend = nanInit[:, :numNans]
                    metricValNanPad = np.concatenate([metricValNanPad, nanAppend],axis=1)
                # timeShiftMetricMats[metricName] = np.insert(currVals, currVals.shape[1], metricValNanPad, axis=2)
                timeShiftMetricMats[metricName] = np.dstack([currVals, metricValNanPad])
            else:
                timeShiftMetricMats[metricName] = metricValNanPad[:,:,None] # make this a column vector

    markerCombos = computeMarkerCombos()

    unLabForCol, colNum = np.unique(labelForColors, return_inverse=True, axis=0)
    unLabForTup, mcNum = np.unique(labelForMarkers, return_inverse=True, axis=0)

    for metricName, metricTmMat in timeShiftMetricMats.items():
        plt.figure()
        plt.title('%s over time' % ( metricName ))
        plt.suptitle(supTitle)
        ax=plt.subplot(1,1,1)
        
        colorsUse = plt.cm.Set3(np.arange(len(unLabForCol)))
        for dset, labDesc, colN, mcN in zip(metricTmMat, descriptions, colNum, mcNum):
            # plot one for the label...
            ax.plot(times[0], np.nan, label=labDesc, color=colorsUse[colN, :], marker=markerCombos[mcN], linestyle='-')
            ax.plot(times, dset.T, color=colorsUse[colN, :], marker=markerCombos[mcN], linestyle='-', alpha=0.5)
        
        ax.set_xlabel('time (ms)')
        ax.set_ylabel(metricName)
        axBx = ax.get_position()
        ax.set_position([axBx.x0, axBx.y0, axBx.width * 0.8, axBx.height])
        ax.legend(prop={'size':5},loc='center left', bbox_to_anchor=(1, 0.5))
        
        # dashed line at zero-point alignment
        axXlim = ax.get_xlim()
        axYlim = ax.get_ylim()
        newXMin = np.min([axXlim[0], 0])
        newXMax = np.max([axXlim[1], 0])
        ax.set_xlim(newXMin,newXMax)
        ax.plot(np.array([0,0]), axYlim, 'k--')


def plotMetricsBySeparation(metricDict, descriptions, separationName, labelForSeparation, labelForColors, labelForMarkers, supTitle = '', secondOrderGrouping = None, splitSecondOrderByLabel = None):
    colorsUse = BinnedSpikeSet.colorset

    # note that labelForSeparation is ideally a *list* (not np.array) of string
    # labels, so that it can be used for the x-axis as well!
    try:
        unLabForSep, colSep = np.unique(labelForSeparation, return_inverse=True, axis=0)
        sepNames = unLabForSep
    except TypeError:
        # allows us to use string objects for the label here... (note that the
        # above allows us to use string *lists*... don't ask me why np.unique
        # doesn't work well on object arrays of strings, it's dumb
        sepNames, colSepTemp = np.unique(labelForSeparation, return_inverse=True)
        unLabForSep, colSep = np.unique(colSepTemp, return_inverse=True) # converts the strings in unLabForSep to integers!

    if secondOrderGrouping is not None and splitSecondOrderByLabel is not None:
        try:
            splitNmPart, colSplitTemp = np.unique(secondOrderGrouping, return_inverse=True, axis=0)
#            unLabForSplit, colSplit = np.unique(colSplitTemp, return_inverse=True, axis=0)
#            splitNames = [splitNameDesc + str(sNT) for sNT in splitNmPart]
        except TypeError:
            # allows us to use string objects for the label here... (note that the
            # above allows us to use string *lists*... don't ask me why np.unique
            # doesn't work well on object arrays of strings, it's dumb
            splitNmPart, colSplitTemp = np.unique(secondOrderGrouping, return_inverse=True)
#            unLabForSplit, colSplit = np.unique(colSplitTemp, return_inverse=True) # converts the strings in unLabForSplit to integers!
#            splitNames = [splitNameDesc + str(sNT) for sNT in splitNmPart]
        orderForLabels = np.stack([(sNP == np.array(splitSecondOrderByLabel)).nonzero()[0] for sNP in splitNmPart])

    unLabForCol, colNumAll = np.unique(labelForColors, return_inverse=True, axis=0)
    unLabForTup, mcNumAll = np.unique(labelForMarkers, return_inverse=True, axis=0)

    markerCombos = computeMarkerCombos()
    for metricName, metricVal in metricDict.items():
        metricVal = [np.array(m).squeeze() for m in metricVal]
        
        plt.figure()
        plt.title('%s split by %s' % ( metricName, separationName))
        plt.suptitle(supTitle)
        ax=plt.subplot(1,1,1)
        
        xTickLocs = []
        scPts = []
        for labelNum, lbl in enumerate(unLabForSep):
            # here we're saying to grab the metric vals whose col (index?) for
            # separation is equal to the index we're at, which is labelNum... 
            metricValSep = list(np.array(metricVal)[colSep==labelNum])
            colNum = colNumAll[colSep==labelNum]
            mcNum = mcNumAll[colSep==labelNum]
            descHere = list(np.array(descriptions)[colSep==labelNum])
            if secondOrderGrouping is None:
                xloc = labelNum


                # get ready for this lovable error catching cascade because I'm
                # so confused on how numpy concatenates shit
                try:
                    # returns a [ 1x# vals (ALL vals) ] when all arrays have
                    # the same number of dimensions, and the dimenson along
                    # axis 0 is the same
                    mVlConc = np.concatenate(metricValSep, axis=0)
                except ValueError as err:
                    try:
                        # returns a [ 1x# vals (ALL vals) ] when all arrays have
                        # the same number of dimensions, but its the dimenson
                        # along axis 1 that is the same, instead of axis 1
                        mVlConc = np.concatenate(metricValSep, axis=1)
                    except ValueError as err2:
                        try:
                            # Hey! What if all the arrays DON'T have the same
                            # number of dimensions, but you could feasibly
                            # match their first dimension (say a (1,) array
                            # with a (3,) array and a () array
                            #
                            # alright, what if you explicitly have a bunch of
                            # 1D or 0D arrays?
                            mVlConc = np.hstack(metricValSep)
                        except ValueError as err3:
                            # Hell, why not?
                            mVlConc = np.vstack(metricValSep)
            
                meanVal = [mVlConc.mean()]
                numSepValsPerGroup = [mVal.shape[0] if len(mVal.shape)>0 else 1 for mVal in metricValSep]
                groupPartitions = [np.cumsum(numSepValsPerGroup)] # need to make a list to match how each group is a list below
            else:

                unGroups = np.unique(secondOrderGrouping)
                groupingInGroup = list(np.array(secondOrderGrouping)[colSep==labelNum])
                xloc = labelNum*unGroups.size

                indsToGroup = np.unique(groupingInGroup, return_inverse=True)[1]
#                metricValSep = [mV[indsToGroup==secGrp] for secGrp in range(np.max(indsToGroup)+1) for mV in np.array(metricVal)[colSep==labelNum]]
#                colNum = list(np.hstack([colNumAll[colSep==labelNum][indsToGroup==secGrp] for secGrp in range(np.max(indsToGroup)+1)]))
                colNum = list(np.hstack([colNumAll[colSep==labelNum][indsToGroup==secGrp] for secGrp in orderForLabels]))
#                colNum = colNumAll[colSep==labelNum]
#                mcNum = list(np.hstack([mcNumAll[colSep==labelNum][indsToGroup==secGrp] for secGrp in range(np.max(indsToGroup)+1)]))
                mcNum = list(np.hstack([mcNumAll[colSep==labelNum][indsToGroup==secGrp] for secGrp in orderForLabels]))
#                mcNum = mcNumAll[colSep==labelNum]
#                descHere = list(np.hstack([np.array(descriptions)[colSep==labelNum][indsToGroup==secGrp] for secGrp in range(np.max(indsToGroup)+1)]))
                descHere = list(np.hstack([np.array(descriptions)[colSep==labelNum][indsToGroup==secGrp] for secGrp in orderForLabels]))
#                descHere = list(np.array(descriptions)[colSep==labelNum])

#                numSepValsPerGroup = [[mVal.shape[0] if len(mVal.shape)>0 else 1 for mVal in np.array(metricValSep)[indsToGroup==secGrpSep]] for secGrpSep in range(np.max(indsToGroup)+1) ]
                numSepValsPerGroup = [[mVal.shape[0] if len(mVal.shape)>0 else 1 for mVal in np.array(metricValSep)[indsToGroup==secGrpSep]] for secGrpSep in orderForLabels ]
                groupPartitions = [np.cumsum(spVlGrp) for spVlGrp in numSepValsPerGroup]

#                mVlConc = [np.hstack([mVal for mVal in np.array(metricValSep)[indsToGroup==secGrpSep]]) for secGrpSep in range(np.max(indsToGroup)+1) ]
                mVlConc = [np.hstack([mVal for mVal in np.array(metricValSep)[indsToGroup==secGrpSep]] if secGrpSep in indsToGroup else [np.array([])]) for secGrpSep in orderForLabels ]

#                mVlConc = [mVlConc[i == indsToGroup] for i in range(np.max(indsToGroup)+1)]
                meanVal = [mVl.mean() for mVl in mVlConc]

                indsToGroup = [[np.repeat(secGrpSep, mVal.shape[0] if len(mVal.shape)>0 else 1)] for secGrpSep in orderForLabels for mVal in np.array(metricValSep)[indsToGroup==secGrpSep]  ]
                indsToGroup = np.hstack(indsToGroup).squeeze()
            
            scatterXY, scatterOrigInds = scatterBar(mVlConc)

            # scatterOrigInds comes out as a numPts x numBars array, but here
            # we're only doing one bar for each group and the 0 index serves to
            # squeeze it for appropriate indexing; honestly, the same goes for
            # why the third dimension of scatterXY has a zero index as well
#            scatterXY = scatterXY[scatterOrigInds[:,0].astype('int'),:,0]
            scatterXY = [scatterXY[scatterOrigInds[:,col].astype('int'), :, col] for col in range(scatterOrigInds.shape[1])]
            scatterXY = np.transpose(np.stack(scatterXY), axes=(1,2,0))
            
            metricValSepWithXPos = [np.split(scatterXY[:, :, grp], groupPartitions[grp])[:-1] for grp in range(len(groupPartitions))]
#            metricValSepWithXPos = np.split(scatterXY, groupPartitions)
            # metricValSepWithXPos = np.vstack([np.vstack(mV) if len(mV)>0 else np.ndarray((0,2)) for mV in metricValSepWithXPos])
            # metricValSepWithXPos = [np.vstack(mV) if len(mV)>0 else np.ndarray((0,2)) for mV in metricValSepWithXPos]
            metricValSepWithXPos = [mV for mVs in metricValSepWithXPos for mV in mVs]
             
            # mtrcWX has two columns with the x and y positions; we transpose
            # to make them two rows, which numpy can represent as the first two
            # elements of a list, which the asterisk then unpacks
            scPtsInit = [ax.scatter(*((mtrcWX + [xloc,0]).T), label=desc, color=colorsUse[colN,:], marker=markerCombos[mcN]) for mtrcWX, desc, colN, mcN in zip(metricValSepWithXPos, descHere, colNum, mcNum)]
            _, unInd = np.unique(descHere, return_index=True)
            scPts += list(np.array(scPtsInit)[unInd])

            # realized I can hard code the width to be 0.25 around the
            # center... trying to make it based on the x-coordinate of the
            # points sometimes caused problems when all the points ended
            # up in the same x location heh
            minX = -0.125
            maxX = 0.125
            [ax.plot(np.array([minX,maxX]) + xloc + offshift, [mnVal,mnVal], 'k') for offshift, mnVal in enumerate(meanVal)]

            # this does the job of centering around the different splits
            xTickLocs.append(xloc + (len(meanVal)-1)/2)


        # apparently running np.array() on a list that has a mix of
        # strings and np.array(nan) types results in the nan being cast
        # to the string 'nan', so this is what I search for here to
        # remove the nans...
#        unLabForCol = unLabForCol[unLabForCol != 'nan']
        if len(scPts) > 2*unLabForCol.shape[0]:
            colLegElem = [Patch(facecolor=colorsUse[colN, :], label=unLC) for colN, unLC in enumerate(unLabForCol)]
        else:
            colLegElem = []

        # see above comment on unLabForCol
        # unLabForTup = unLabForTup[unLabForTup != 'nan']
        if len(scPts) > 2*unLabForTup.shape[0]:
            colGray = [0.5,0.5,0.5]
            mrkLegElem = [Line2D([0], [0], marker=markerCombos[lblOrd], label=unLabForTup[lblOrd], color='w', markeredgewidth=0.0, markerfacecolor=colGray, markersize=7) for lblOrd in orderForLabels.squeeze()]
        else:
            mrkLegElem = []
        
        ax.set_xticks(np.array(xTickLocs))
        ax.set_xticklabels(unLabForSep)
        ax.set_ylabel(metricName)
        ax.set_xlabel(separationName)
        
        # make space for a legend to the right of the plot
        axBx = ax.get_position()
        ax.set_position([axBx.x0, axBx.y0, axBx.width * 0.8, axBx.height])
        lgGen = ax.legend(handles=colLegElem + mrkLegElem,prop={'size':5},loc='upper left', 
#            handler_map = 
#                {type(colLegElem[0]) : HandlerPatch(),
#                type(mrkLegElem[0]) : TransitionHandler(stNamesInOrder = unLabForTup[orderForLabels.squeeze()], transitionColors=transitionColors)}
                bbox_to_anchor=(1, 1))
        ax.legend(handles=scPts,prop={'size':5},loc='lower left', bbox_to_anchor=(1, 0))
        ax.add_artist(lgGen)
#        ax.axis('equal')



def plotAllVsAllExtractionParamShift(descriptions, metricDict, splitNameDesc, labelForGrouping, labelForSplit, splitOrderByLabel, labelForCol, labelForMarker, supTitle=""):
    # splitOrderByLabel has only as many elements as unique labels, and those
    # elements are the unique elements in the order the arrow will point
    colorsUse = BinnedSpikeSet.colorset

    if labelForMarker != labelForSplit:
        breakpoint() 
        # not sure what this means in this function, and more importantly some marker legend code below assumes this

    try:
        splitNmPart, colSplitTemp = np.unique(labelForSplit, return_inverse=True, axis=0)
        unLabForSplit, colSplit = np.unique(colSplitTemp, return_inverse=True, axis=0)
        splitNames = [splitNameDesc + str(sNT) for sNT in splitNmPart]
    except TypeError:
        # allows us to use string objects for the label here... (note that the
        # above allows us to use string *lists*... don't ask me why np.unique
        # doesn't work well on object arrays of strings, it's dumb
        splitNmPart, colSplitTemp = np.unique(labelForSplit, return_inverse=True)
        unLabForSplit, colSplit = np.unique(colSplitTemp, return_inverse=True) # converts the strings in unLabForSplit to integers!
        splitNames = [splitNameDesc + str(sNT) for sNT in splitNmPart]

    orderForLabels = np.stack([(sNP == np.array(splitOrderByLabel)).nonzero()[0] for sNP in splitNmPart])

    try:
        unLabForGrouping, colGrouping, nmPerGrouping = np.unique(labelForGrouping, return_inverse=True, return_counts=True, axis=0)
    except TypeError:
        # allows us to use string objects for the label here... (note that the
        # above allows us to use string *lists*... don't ask me why np.unique
        # doesn't work well on object arrays of strings, it's dumb
        pairNames, colGroupingTemp = np.unique(labelForGrouping, return_inverse=True)
        unLabForGrouping, colGrouping, nmPerGrouping = np.unique(colGroupingTemp, return_inverse=True, return_counts=True) # converts the strings in unLabForSplit to integers!


    groupItemSplits = [colGrouping[colSplit == uLfS] for uLfS in unLabForSplit]

#    # here we're kind of assuming everything is organized in order... which...
#    # may not be the most correct assumption, hmm...
#    descriptions = list(np.array(descriptions)[colSplit==unLabForSplit[0]])
#    labelForCol = labelForCol[colSplit==unLabForSplit[0]]
#    labelForMarker = labelForMarker[colSplit==unLabForSplit[0]]
    
    markerCombos = computeMarkerCombos()

    splitName1 = splitNames[unLabForSplit[0]]
    splitName2 = splitNames[unLabForSplit[1]]

    for metric1Num, (metric1Name, metric1Val) in enumerate(metricDict.items()):
        for metric2Num, (metric2Name, metric2Val) in enumerate(metricDict.items()):

            try:
                mVl1Conc = np.concatenate(metric1Val, axis=0)
            except ValueError:
                mVl1Conc = np.concatenate(metric1Val, axis=1)

            try:
                mVl2Conc = np.concatenate(metric2Val, axis=0)
            except ValueError:
                mVl2Conc = np.concatenate(metric2Val, axis=1)

            # if metric1Name == 'ones dir on each latent' and (metric2Name == 'each factor load sim' or metric2Name == '%sv by latent'):
            #     breakpoint()
            #     # ext([0.82952093 0.0666976 ], [0.11390571 0.02941919], '')

            if mVl1Conc.size != mVl2Conc.size:
                continue

            if metric2Num > metric1Num:
                metric1ValSplitUnsorted, grpSort = splitMetricVals(metric1Val, colSplit, unLabForSplit, groupItemSplits)
                metric1ValSplit = np.array(metric1ValSplitUnsorted)[orderForLabels].squeeze()

                metric2ValSplitUnsorted, grp2Sort = splitMetricVals(metric2Val, colSplit, unLabForSplit, groupItemSplits)
                metric2ValSplit = np.array(metric2ValSplitUnsorted)[orderForLabels].squeeze()


                if grpSort != grp2Sort:
                    breakpoint() # erm...
#                labelForColSp = np.array(labelForCol)[colSplit == unLabForSplit[grpSort]]
#                labelForMarkerSp = np.array(labelForMarker)[colSplit == unLabForSplit[grpSort]]
                if np.array(grpSort).size==1:
                    descriptionsSp = np.array(descriptions)[colSplit == unLabForSplit[grpSort]]
                else:
                    unGrpItm,inds=np.unique(groupItemSplits,return_inverse=True)
                    whereUnGrp = [np.nonzero([np.any(gpSp==unG) for gpSp in groupItemSplits])[0][0] for unG in unGrpItm]
                    descriptionsSp = np.array([np.array(descriptions)[colSplit==wUG][groupItemSplits[wUG]==uG] for wUG, uG in zip(whereUnGrp, unGrpItm)])
#

                labelForColSpUnsort, _ = splitMetricVals(labelForCol, colSplit, unLabForSplit, groupItemSplits)
                labelForColsSp = np.array(labelForColSpUnsort)[orderForLabels].squeeze()

                labelForMarkerSpUnsort, _ = splitMetricVals(labelForMarker, colSplit, unLabForSplit, groupItemSplits)
                labelForMarkersSp = np.array(labelForMarkerSpUnsort)[orderForLabels].squeeze()


                plt.figure()
                plt.title('%s vs %s' % ( metric1Name, metric2Name ))
                plt.suptitle(supTitle)
                if metric1Name.find('factor load') != -1 or metric1Name.find('factor load') != -1:
                    if metric1Name.find('%sv') != -1 or metric2Name.find('%sv') != -1:
                        ax = plt.subplot(1,1,1, projection='polar')
                        if metric1Name.find('factor load'):
                            # in this case we're switching things so factor load is always theta (the second input) and the %sv is always the radius
                            metric2ValSplitTemp = metric1ValSplit
                            metric1ValSplit = np.pi/2-metric2ValSplit/1*np.pi/2 # stretching it from 0-1 to be pi/2-0 (so the angle)
                            metric2ValSplit = metric2ValSplitTemp
                            metric2NameTemp = metric1Name
                            metric1Name = metric2Name
                            metric2Name = metric2NameTemp

                            metric2ValSplit = np.pi/2-metric2ValSplit/1*np.pi/2 # stretching it from 0-1 to be pi/2-0 (so the angle)
                        elif metric2Name.find('factor load'):
                            metric2ValSplit = np.pi/2-metric2ValSplit/1*np.pi/2 # stretching it from 0-1 to be pi/2-0 (so the angle)
                else:        
                    ax=plt.subplot(1,1,1)

                xCoordsDiffSplits = metric1ValSplit
                yCoordsDiffSplits = metric2ValSplit

                xStarts = xCoordsDiffSplits[:-1,:]
                yStarts = yCoordsDiffSplits[:-1,:]

                xEnds = xCoordsDiffSplits[1:,:]
                yEnds = yCoordsDiffSplits[1:,:]

                # try:
                #     xArrowLengths = np.diff(xCoordsDiffSplits, axis=0)
                #     yArrowLengths = np.diff(yCoordsDiffSplits, axis=0)
                # except ValueError:
                #     continue



                # a counter to tell us which transition the arrow being drawn is for
                transitionList = np.arange(xStarts.shape[0])
                transitionNumber = (transitionList[:,None] + np.zeros_like(xStarts[0])).astype('int')

#                [plt.arrow(x, y, dx, dy) for x,y,dx,dy in zip(xStarts.flat, yStarts.flat, xArrowLengths.flat, yArrowLengths.flat)]
                transitionColors = plt.cm.Pastel2(transitionList)
                if np.hstack(xStarts[0].flatten()).size == xStarts[0].size:
                    [ax.annotate("", xy = (xE,yE), xytext = (xS, yS), arrowprops=dict(arrowstyle="->", ec=tuple(transitionColors[trNum]))) for xS,yS,xE,yE,trNum in zip(xStarts.flat, yStarts.flat, xEnds.flat, yEnds.flat, transitionNumber.flat) if not np.any(np.isnan([xS, yS]))]
                else:
                    for xS,yS,xE,yE,trNum in zip(xStarts.flat, yStarts.flat, xEnds.flat, yEnds.flat, transitionNumber.flat): 
                        if not np.any(np.isnan([xS, yS])):
                            minSize = np.min([xS.size, xE.size])
                            xS = xS[:minSize]
                            yS = yS[:minSize]
                            xE = xE[:minSize]
                            yE = yE[:minSize]
                            [ax.annotate("", xy = (xESub,yESub), xytext = (xSSub, ySSub), arrowprops=dict(arrowstyle="->", ec=tuple(transitionColors[trNum]))) for xSSub, ySSub, xESub, yESub in zip(xS, yS, xE, yE)]
            
                # mmk a trick little way to make sure colors and markers stay
                # the same depending on the label, while also allowing colors
                # and markers from each split to be split-specific
                # 1) we first concatenate all the labels together
                labelForMarkerConc = np.concatenate(labelForMarkersSp)
                labelForColorConc = np.concatenate(labelForColsSp)
                # 2) we then find the unique values that exist for each
                labelForMarkerUnique = np.unique(labelForMarkerConc)
                labelForColorUnique = np.unique(labelForColorConc)
                # 3) below, we'll now concatenate these unique values to any
                # unique search, to make sure each unique value retains its
                # position/order, even if its not present in a particular
                # split, and then make sure to boot the first [length unique]
                # from the inverse search so the sizes are correct

                scPts = []
                for m1Val, m2Val, labColors, labMrkrs in zip(metric1ValSplit, metric2ValSplit, labelForColsSp, labelForMarkersSp):

                    # look at above comment 3) to understand this concatenation
                    # before unique and why we ignore the beginning of colNum
                    # and mcNum
                    unLabForCol, colNum = np.unique(np.concatenate([labelForColorUnique, labColors]), return_inverse=True, axis=0)
                    unLabForTup, mcNum = np.unique(np.concatenate([labelForMarkerUnique, labMrkrs]), return_inverse=True, axis=0)
                    colNum = colNum[len(labelForColorUnique):]
                    mcNum = mcNum[len(labelForMarkerUnique):]

#                    if mVl1Conc.size != mVl2Conc.size:
#                        # some small cases have values broken down even more than just
#                        # by-condition, so they don't line up well with the
#                        # by-condition ones; I'm skipping these for now
#                        continue
                    scPts.append([ax.scatter(m1, m2, label=desc, color=colorsUse[colN,:], marker=markerCombos[mcN]) for m1, m2, desc, colN, mcN in zip(m1Val, m2Val, descriptionsSp, colNum, mcNum)])

                # this is a little opaque--basically, scPts is a list of lists,
                # where each sublist is a handle to the plots of a certain
                # grouping; we want the handles to the plots where *all* the
                # datasets are present--i.e. to the largest grouping; that's
                # what's in grpSort. But grpSort has been shuffled around
                # based on the orderForLabels, so this serves to find where it
                # has ended up
                if np.array(grpSort).size==1:
                    scPts = list(np.array(scPts)[(orderForLabels==grpSort).squeeze()].squeeze())
                else:
                    unGrpItm,inds=np.unique(groupItemSplits,return_inverse=True)
                    whereUnGrp = [np.nonzero([np.any(gpSp==unG) for gpSp in groupItemSplits])[0][0] for unG in unGrpItm]
                    scPts = [np.array(scPts)[:,colSplit==wUG][:,groupItemSplits[wUG]==uG] for wUG, uG in zip(whereUnGrp, unGrpItm)]
                    scPts = list(np.vstack(scPts).squeeze())
                # apparently running np.array() on a list that has a mix of
                # strings and np.array(nan) types results in the nan being cast
                # to the string 'nan', so this is what I search for here to
                # remove the nans...
                unLabForCol = unLabForCol[unLabForCol != 'nan']
                if metric1ValSplit.shape[1] > 2*unLabForCol.shape[0]:
                    colLegElem = [Patch(facecolor=colorsUse[colN, :], label=unLC) for colN, unLC in enumerate(unLabForCol)]
                else:
                    colLegElem = []

                # see above comment on unLabForCol
                unLabForTup = unLabForTup[unLabForTup != 'nan']
                if metric1ValSplit.shape[1] > 2*unLabForTup.shape[0]:
                    colGray = [0.5,0.5,0.5]
#                    mrkLegElem = [Line2D([0], [0], marker=markerCombos[mcN], label=unLM, color='w', markerfacecolor=colGray, markersize=7) for mcN, unLM in enumerate(unLabForTup)]
                    mrkLegElem = [Line2D([0], [0], marker=markerCombos[lblOrd], label=unLabForTup[lblOrd], markeredgewidth=0.0, markerfacecolor=colGray, markersize=7) for lblOrd in orderForLabels.squeeze()]
                else:
                    mrkLegElem = []

                if metric1Name.find('r_{sc}') != -1 or metric1Name.find('%sv') != -1 or metric1Name.find('sh pop') != -1:
                    minMet1Val = np.nanmin(np.hstack(xCoordsDiffSplits.flatten()))
                    maxMet1Val = np.nanmax(np.hstack(xCoordsDiffSplits.flatten()))

                    if metric2Name.find('r_{sc}') != -1 or metric2Name.find('%sv') != -1 or metric2Name.find('sh pop') != -1:
                        minMet2Val = np.nanmin(np.hstack(yCoordsDiffSplits.flatten()))
                        maxMet2Val = np.nanmax(np.hstack(yCoordsDiffSplits.flatten()))
                        min2Val = np.array([minMet2Val, 0]).min()

                        minVal = np.array([minMet1Val, minMet2Val, 0]).min()
                        maxVal = np.array([maxMet1Val, maxMet2Val]).max()
                        breathingRoom = 0.1*(maxVal-minVal)
                        ax.set_xlim(left=minVal - breathingRoom, right=maxVal + breathingRoom)
                        ax.set_ylim(bottom=minVal - breathingRoom, top=maxVal + breathingRoom)
                        ax.axhline(0, color='black', linestyle='--')
                        ax.axvline(0, color='black', linestyle='--')
                

                ax.set_xlabel(metric1Name)
                ax.set_ylabel(metric2Name)
                axBx = ax.get_position()
                ax.set_position([axBx.x0, axBx.y0, axBx.width * 0.8, axBx.height])
#                ax.legend(colLegElem + scPts, [elm.get_label() for elm in colLegElem+scPts],prop={'size':5},loc='center left', bbox_to_anchor=(1, 0.5))
                if len(colLegElem + mrkLegElem):
                    lgGen = ax.legend(handles=colLegElem + mrkLegElem,prop={'size':5},loc='upper left', bbox_to_anchor=(1, 1),
                        handler_map = 
                            {type(colLegElem[0]) : HandlerPatch(),
                            type(mrkLegElem[0]) : TransitionHandler(stNamesInOrder = unLabForTup[orderForLabels.squeeze()], transitionColors=transitionColors)})
                    ax.add_artist(lgGen)
                ax.legend(handles=scPts,prop={'size':5},loc='lower left', bbox_to_anchor=(1, 0))
