#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 17:05:23 2020

@author: emilio
"""
from matplotlib import pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.axes._subplots import Axes

def BlueWhiteRedColormap(figs=None, lowPt = None, midPt = 0, highPt = None):
    if figs is None:
        if plt.get_fignums() != []:
            figs = [plt.gcf()]
        else:
            return
        
    for fig in figs:
        childs = fig.get_children()
        for child in childs:
            if isinstance(child, Axes):
                imgs = child.get_images()
                for img in imgs:
                    dat = img.get_array()
                    if lowPt is None:
                        lowPt = np.min(dat)
                    if highPt is None:
                        highPt = np.max(dat)
                        
                    # we're centering around the low/high point of clim
                    minDatOrig = lowPt 
                    maxDatOrig = highPt                   
                    
                    # get 255 values, meaning 127 is the halfway white
                    cmapInit = plt.get_cmap('RdBu_r', 255)
                    
                    # make 0 the midpoint
                    minDatMidCenter = minDatOrig - midPt
                    maxDatMidCenter = maxDatOrig - midPt
                    midPtIsZero = 0
                    
                    if minDatMidCenter <= midPtIsZero and maxDatMidCenter >= midPtIsZero:
                        if np.abs(minDatMidCenter) > np.abs(maxDatMidCenter):
                            valSmallCmap = 0
                            valLargeCmap = 127 + np.abs(maxDatMidCenter)/np.abs(minDatMidCenter) * 127
                        elif np.abs(maxDatMidCenter) > np.abs(minDatMidCenter):
                            valSmallCmap = 127 - np.abs(minDatMidCenter)/np.abs(maxDatMidCenter) * 127
                            valLargeCmap = 255
                            
                        
                        print(valLargeCmap)
                    elif minDatMidCenter >= midPtIsZero:
                        valSmallCmap = 127 + minDatMidCenter/maxDatMidCenter * 127
                        valLargeCmap = 255
                    elif maxDatMidCenter <= midPtIsZero:
                        valSmallCmap = 0
                        valLargeCmap = 127 - maxDatMidCenter/minDatMidCenter
                        
                    
                    print(valLargeCmap)
                    valSmallCmap = valSmallCmap/255
                    valLargeCmap = valLargeCmap/255
                    print(valLargeCmap)
                    
                    cmapVals = ListedColormap(cmapInit(np.linspace(valSmallCmap, valLargeCmap, 100)))
                    
                    img.set_cmap(cmapVals)
                    img.set_clim(lowPt, highPt)
    
# this function more speicifcally moves the *axes* of the original figure to
# the subplot--I'm actually not sure whether titles and legends and labels move
# alongside, but I hope they do?       
# 
# note that newSubplotToCopyInto should be provided as a list or tuple of
# [newLeft, newBot, newWidth, newHeight] as gets returned from get_position()
# on the subplot             
def MoveFigureToSubplot(origFig, newFig, newSubplotToCopyInto):
    
    oldFigAx = origFig.axes
    newFigPos = newSubplotToCopyInto.get_position().bounds
    newSubplotToCopyInto.remove()
    
    for ax in oldFigAx:
        ax.remove() # remove axis from the old figure
        ax.figure = newFig # add new figure to the axis
        #newFig.axes.append(ax) # add new axis to the figure also has to be done
        newFig.add_axes(ax) # either or both? Not sure
        
        origPos = ax.get_position().bounds # this is still saved even though the axis was deleted from the old figure...
        newAxPos = [newFigPos[0]+origPos[0]*newFigPos[2], newFigPos[1]+origPos[1]*newFigPos[3], origPos[2]*newFigPos[2], origPos[3]*newFigPos[3]]
        ax.set_position(newAxPos, which='both')
#        breakpoint()
        
    plt.close(origFig)


# same as above, but only one axis now...  shout out to
# https://gist.github.com/salotz/8b4542d7fe9ea3e2eacc1a2eef2532c5 for...
# well, basically the code
#
# note that not removing the old fig might cause problems is my
# understanding... byuuut.... maybe you don't wanna so
# let's keep it False as default
def MoveAxisToSubplot(axToMove, newFig, newSubplotToCopyInto, removeOldFig = False):
    # original figure
    oldFig = axToMove.figure

    # remove axis from said figure
    axToMove.remove()

    # assign axis to new figure and vice-versa
    axToMove.figure = newFig
    newFig.axes.append(axToMove) # add new axis to the figure also has to be done
    newFig.add_axes(axToMove)

    # set its position to the correct subplot
    axToMove.set_position(newSubplotToCopyInto.get_position())

    # remove the temporary subplot
    newSubplotToCopyInto.remove()

    # there definitely might be problems if you don't do this...
    if removeOldFig:
        plt.close(oldFig)

