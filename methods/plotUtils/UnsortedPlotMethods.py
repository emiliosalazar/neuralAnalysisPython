"""
For plot methods I don't know where to put... may this list be short and in
flux as I move thing to their final destinations...
"""
import numpy as np
from classes.BinnedSpikeSet import BinnedSpikeSet
from matplotlib import pyplot as plt

def plotAllVsAll(descriptions, metricDict, labelForCol, labelForMarker, supTitle=""):
    colorsUse = BinnedSpikeSet.colorset
    from itertools import product,chain
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
#    breakpoint()
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
        # allows us to use strings for hte label here...
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
        axBx = ax.get_position()
        ax.set_position([axBx.x0, axBx.y0, axBx.width * 0.8, axBx.height])
        ax.legend(prop={'size':5},loc='center left', bbox_to_anchor=(1, 0.5))
        ax.axis('equal')
        
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
    from itertools import product,chain
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

