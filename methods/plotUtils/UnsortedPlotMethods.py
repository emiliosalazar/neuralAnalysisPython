"""
For plot methods I don't know where to put... may this list be short and in
flux as I move thing to their final destinations...
"""
import numpy as np
from classes.BinnedSpikeSet import BinnedSpikeSet
from matplotlib import pyplot as plt

def pltAllVsAll(descriptions, metricDict, labelForCol, labelForMarker, supTitle=""):
    def grp_range(a):
        idx = a.cumsum()
        id_arr = np.ones(idx[-1],dtype=int)
        id_arr[0] = 0
        id_arr[idx[:-1]] = -a[:-1]+1
        return id_arr.cumsum()
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

