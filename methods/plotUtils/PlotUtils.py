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
                        
                        
                    
                    
