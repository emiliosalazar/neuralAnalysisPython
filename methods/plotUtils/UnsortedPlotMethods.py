"""
For plot methods I don't know where to put... may this list be short and in
flux as I move thing to their final destinations...
"""
import numpy as np
from classes.BinnedSpikeSet import BinnedSpikeSet
from methods.plotUtils.ScatterBar import scatterBar
from matplotlib import pyplot as plt
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
            if metric2Num > metricNum:
                plt.figure()
                plt.title('%s vs %s' % ( metricName, metric2Name ))
                plt.suptitle(supTitle)
                ax=plt.subplot(1,1,1)
                unLabForCol, colNum = np.unique(labelForCol, return_inverse=True, axis=0)
                unLabForTup, mcNum = np.unique(labelForMarker, return_inverse=True, axis=0)
                count = np.unique(labelForMarker,return_counts=1)[1]
                angle = grp_range(count)[np.argsort(labelForMarker).argsort()]
#                breakpoint()
                [ax.scatter(m1, m2, label=desc, color=colorsUse[colN,:], marker=markerCombos[mcN]) for m1, m2, desc, colN, mcN in zip(metricVal, metric2Val, descriptions, colNum, mcNum)]
                ax.set_xlabel(metricName)
                ax.set_ylabel(metric2Name)
                axBx = ax.get_position()
                ax.set_position([axBx.x0, axBx.y0, axBx.width * 0.8, axBx.height])
                ax.legend(prop={'size':5},loc='center left', bbox_to_anchor=(1, 0.5))

def plotMetricVsExtractionParams(descriptions, metricDict, splitNames, labelForSplit, labelForCol, labelForMarker, supTitle=""):
    colorsUse = BinnedSpikeSet.colorset
    try:
        unLabForSplit, colSplit = np.unique(labelForSplit, return_inverse=True, axis=0)
    except TypeError:
        # allows us to use string objects for the label here... (note that the
        # above allows us to use string *lists*... don't ask me why np.unique
        # doesn't work well on object arrays of strings, it's dumb
        splitNames, colSplitTemp = np.unique(labelForSplit, return_inverse=True)
        unLabForSplit, colSplit = np.unique(colSplitTemp, return_inverse=True) # converts the strings in unLabForSplit to integers!

    if unLabForSplit.size!=2:
        raise Exception("Currently can only plot a 2-way split!")
    if np.sum(colSplit==unLabForSplit[0]) != np.sum(colSplit==unLabForSplit[1]):
        raise Exception("Splits need to be PAIRED--which means the label split should evenly split things in two!")

    # here we're kind of assuming everything is organized in order... which...
    # may not be the most correct assumption, hmm...
    descriptions = list(np.array(descriptions)[colSplit==unLabForSplit[0]])
    labelForCol = labelForCol[colSplit==unLabForSplit[0]]
    labelForMarker = labelForMarker[colSplit==unLabForSplit[0]]
    
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
        metricValSplit1 = list(np.array(metricVal)[colSplit==unLabForSplit[0]])
        splitName1 = splitNames[unLabForSplit[0]]
        metricValSplit2 = list(np.array(metricVal)[colSplit==unLabForSplit[1]])
        splitName2 = splitNames[unLabForSplit[1]]
        
        
        plt.figure()
        plt.title('%s: %s vs %s' % ( metricName, splitNames[0], splitNames[1] ))
        plt.suptitle(supTitle)
        ax=plt.subplot(1,1,1)
        unLabForCol, colNum = np.unique(labelForCol, return_inverse=True, axis=0)
        unLabForTup, mcNum = np.unique(labelForMarker, return_inverse=True, axis=0)
        count = np.unique(labelForMarker,return_counts=1)[1]
        angle = grp_range(count)[np.argsort(labelForMarker).argsort()]
        [ax.scatter(m1, m2, label=desc, color=colorsUse[colN,:], marker=markerCombos[mcN]) for m1, m2, desc, colN, mcN in zip(metricValSplit1, metricValSplit2, descriptions, colNum, mcNum)]
        ax.set_xlabel(splitName1)
        ax.set_ylabel(splitName2)

        # make space for a legend to the right of the plot
        axBx = ax.get_position()
        ax.set_position([axBx.x0, axBx.y0, axBx.width * 0.8, axBx.height])
        ax.legend(prop={'size':5},loc='center left', bbox_to_anchor=(1, 0.5))
        ax.axis('equal')
        
        # draw a dashed line representing y=x line
        axXlim = ax.get_xlim()
        axYlim = ax.get_ylim()
        newMin = np.min([axXlim[0],axYlim[0]])
        newMax = np.max([axXlim[1],axYlim[1]])
        ax.set_xlim(newMin,newMax)
        ax.set_ylim(newMin,newMax)
        ax.plot(np.array([newMin,newMax]), np.array([newMin,newMax]), 'k--')


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
                       
            
            numSepValsPerGroup = [mVal.shape[0] for mVal in metricValSep]
            groupPartitions = np.cumsum(numSepValsPerGroup)
            
            scatterXY, scatterOrigInds = scatterBar(np.concatenate(metricValSep))
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
    

