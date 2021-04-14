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
from matplotlib.legend_handler import HandlerLine2D, HandlerPatch

from itertools import product,chain

def plotPointProjections(projOnSignalZscByExParams, meanOnSignalByExParams, latentsOnSignalZscByExParams, descTitles):

    for projSig, meanSig, latSig, ttls in zip(projOnSignalZscByExParams, meanOnSignalByExParams, latentsOnSignalZscByExParams, descTitles):
        numSubsamp = len(projSig)
        subRows = np.sqrt(numSubsamp)
        subCols = subRows

        fgDataset, axs = plt.subplots(int(subRows), int(subCols))
        if subRows == 1:
            axs = np.array(axs)
        fgDataset.tight_layout()
        fgDataset.suptitle(ttls)

        # prep twenty colors for conditions... not sure why there'd be more?
        condCols = plt.cm.Set2(np.arange(20))

        for ax, pS, mS, lS in zip(axs.flat, projSig, meanSig, latSig):
            [ax.scatter(p[:, 0], p[:, 1], color=col, marker='.') for p, col in zip(pS, condCols)]
            [ax.plot(m[0], m[1], color=col, marker='d') for m, col in zip(mS, condCols)]
            [ax.plot(m[0]+0.5*np.array([-l[0], l[0]]), m[1]+0.5*np.array([-l[1], l[1]])) for l in lS for m in mS]