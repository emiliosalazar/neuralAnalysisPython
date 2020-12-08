"""
For plot methods I don't know where to put... may this list be short and in
flux as I move thing to their final destinations...
"""
import numpy as np
from classes.BinnedSpikeSet import BinnedSpikeSet
from methods.plotUtils.ScatterBar import scatterBar

from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from itertools import product,chain

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
                ax=plt.subplot(1,1,1)
                unLabForCol, colNum = np.unique(labelForCol, return_inverse=True, axis=0)
                unLabForTup, mcNum = np.unique(labelForMarker, return_inverse=True, axis=0)
                count = np.unique(labelForMarker,return_counts=1)[1]
                angle = grp_range(count)[np.argsort(labelForMarker).argsort()]
#                breakpoint()
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
                    minMet1Val = mVlConc.min()
                    maxMet1Val = mVlConc.max()

                    if metric2Name.find('r_{sc}') != -1 or metric2Name.find('%sv') != -1 or metric2Name.find('sh pop') != -1:
                        minMet2Val = mVl2Conc.min()
                        maxMet2Val = mVl2Conc.max()
                        min2Val = np.array([minMet2Val, 0]).min()

                        minVal = np.array([minMet1Val, minMet2Val, 0]).min()
                        maxVal = np.array([maxMet1Val, maxMet2Val]).max()
                        breathingRoom = 0.1*(maxVal-minVal)
                        ax.set_xlim(left=minVal - breathingRoom, right=maxVal + breathingRoom)
                        ax.set_ylim(bottom=minVal - breathingRoom, top=maxVal + breathingRoom)
                        ax.axhline(0, color='black', linestyle='--')
                        ax.axvline(0, color='black', linestyle='--')

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
        plt.title('%s: %s vs %s' % ( metricName, splitNames[0], splitNames[1] ))
        plt.suptitle(supTitle)
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
    metricVal = [np.array(m).squeeze() for m in metricVal]
    metricValSplitInit = [list(np.array(metricVal)[colSplit==uLfS]) for uLfS in unLabForSplit]

    groupSplitLens = np.array([len(mVSI) for mVSI in metricValSplitInit])
    groupLargest = groupSplitLens.argmax()
    metricValSplit = [[mVSplit[(gISplit == gI).nonzero()[0][0]] if np.any(gISplit == gI) else np.full_like(metricValSplitInit[groupLargest][(groupItemSplit[groupLargest]==gI).nonzero()[0][0]], np.nan, dtype=np.double) for gI in groupItemSplit[groupLargest]] for mVSplit, gISplit in zip(metricValSplitInit, groupItemSplit)]
        
    sortingOfGroups = groupLargest
#    if len(metricValSplit1Init) > len(metricValSplit2Init):
#        metricValSplit1 = metricValSplit1Init
#        metricValSplit2 = [metricValSplit2Init[(pairItemSplit2 == pI1).nonzero()[0][0]] if np.any(pairItemSplit2 == pI1) else np.full_like(metricValSplit1[(pairItemSplit1==pI1).nonzero()[0][0]], np.nan, dtype=np.double) for pI1 in pairItemSplit1]
#    else:
#        metricValSplit2 = metricValSplit2Init
#        metricValSplit1 = [metricValSplit1Init[(pairItemSplit1 == pI2).nonzero()[0][0]] if np.any(pairItemSplit1 == pI2) else np.full_like(metricValSplit2[(pairItemSplit2==pI2).nonzero()[0][0]], np.nan, dtype=np.double) for pI2 in pairItemSplit2]
#
    return metricValSplit, sortingOfGroups



def plotTimeEvolution(descriptions, timeShiftMetricDict, labelForMarkers, labelForColors, supTitle = ''):
    colorsUse = BinnedSpikeSet.colorset

    times = []

    timeShiftMetricMats = {}
    for tm, tmMetricDict in timeShiftMetricDict.items():
        times.append(tmMetricDict.pop('timeCenterPoint'))
        for metricName, metricVal in tmMetricDict.items():
            # I'm actually not sure what squeeze() does here... it must be
            # useful for some situations and I ported it from above
            metricVal = np.array([np.array(m).squeeze() for m in metricVal])
            mxLen = np.max([m.shape[0] for m in metricVal])
            metricValNanPad = np.array([np.pad(m.astype('float'), ((0,mxLen-m.shape[0])), constant_values=(np.nan,)) for m in metricVal])
            if metricName in timeShiftMetricMats:
                currVals = timeShiftMetricMats[metricName]
                timeShiftMetricMats[metricName] = np.insert(currVals, -1, metricValNanPad, axis=2)
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
        
        for dset, labDesc, colN, mcN in zip(metricTmMat, descriptions, colNum, mcNum):
            # plot one for the label...
            ax.plot(times[0], np.nan, label=labDesc, color=colorsUse[colN, :], marker=markerCombos[mcN], linestyle='-')
            ax.plot(times, dset.T, color=colorsUse[colN, :], marker=markerCombos[mcN], linestyle='-')
        
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

def plotMetricsBySeparation(metricDict, descriptions, separationName, labelForSeparation, labelForColors, labelForMarkers, supTitle = ''):
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

    unLabForCol, colNumAll = np.unique(labelForColors, return_inverse=True, axis=0)
    unLabForTup, mcNumAll = np.unique(labelForMarkers, return_inverse=True, axis=0)

    markerCombos = computeMarkerCombos()
    for metricName, metricVal in metricDict.items():
        metricVal = [np.array(m).squeeze() for m in metricVal]
        
        plt.figure()
        plt.title('%s split by %s' % ( metricName, separationName))
        plt.suptitle(supTitle)
        ax=plt.subplot(1,1,1)
        
        
        for xloc, lbl in enumerate(unLabForSep):
            # here we're saying to grab the metric vals whose col (index?) for
            # separation is equal to the index we're at, which is xloc... 
            metricValSep = list(np.array(metricVal)[colSep==xloc])
            colNum = colNumAll[colSep==xloc]
            mcNum = mcNumAll[colSep==xloc]
            descHere = list(np.array(descriptions)[colSep==xloc])


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
            meanVal = mVlConc.mean()
            
            numSepValsPerGroup = [mVal.shape[0] if len(mVal.shape)>0 else 1 for mVal in metricValSep]
            groupPartitions = np.cumsum(numSepValsPerGroup)
            

            # not sure why I was doing this instead of using what I calculated above...
#            try:
#                scatterXY, scatterOrigInds = scatterBar(np.concatenate(metricValSep))
#            except ValueError:
#                scatterXY, scatterOrigInds = scatterBar(np.stack(metricValSep))
            scatterXY, scatterOrigInds = scatterBar(mVlConc)

            # scatterOrigInds comes out as a numPts x numBars array, but here
            # we're only doing one bar for each group and the 0 index serves to
            # squeeze it for appropriate indexing; honestly, the same goes for
            # why the third dimension of scatterXY has a zero index as well
            scatterXY = scatterXY[scatterOrigInds[:,0].astype('int'),:,0]
            
            metricValSepWithXPos = np.split(scatterXY, groupPartitions)
             
            # mtrcWX has two columns with the x and y positions; we transpose
            # to make them two rows, which numpy can represent as the first two
            # elements of a list, which the asterisk then unpacks
            [ax.scatter(*((mtrcWX + [xloc,0]).T), label=desc, color=colorsUse[colN,:], marker=markerCombos[mcN]) for mtrcWX, desc, colN, mcN in zip(metricValSepWithXPos, descHere, colNum, mcNum)]

            # realized I can hard code the width to be 0.25 around the
            # center... trying to make it based on the x-coordinate of the
            # points sometimes caused problems when all the points ended
            # up in the same x location heh
            minX = -0.125
            maxX = 0.125
            ax.plot(np.array([minX,maxX]) + xloc, [meanVal,meanVal], 'k')


        
        ax.set_xticks(np.arange(len(unLabForSep)))
        ax.set_xticklabels(unLabForSep)
        ax.set_ylabel(metricName)
        ax.set_xlabel(separationName)
        
        # make space for a legend to the right of the plot
        axBx = ax.get_position()
        ax.set_position([axBx.x0, axBx.y0, axBx.width * 0.8, axBx.height])
        ax.legend(prop={'size':5},loc='center left', bbox_to_anchor=(1, 0.5))
#        ax.axis('equal')



    pass
    

