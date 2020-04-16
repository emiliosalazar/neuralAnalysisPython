#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 11:12:18 2020

@author: emilio
"""

def parallelLoad(data):
    dataUse = data[ind]
    dataMatPath = dataUse['path'] / processedDataMat
    
    print('processing data set ' + dataUse['description'])
    datasetHere = Dataset(dataMatPath, dataUse['processor'], notChan = [31,0])
    data[ind]['dataset'] = datasetHere
    datasets.append(datasetHere)
    
    cosTuningCurves.append(datasetHere.computeCosTuningCurves())