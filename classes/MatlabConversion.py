#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 15:12:14 2020

@author: emilio
"""

from matlab import mlarray, double
import numpy as np

class MatlabConversion:

    def __init__(self):
        # nothin' to see here folks
        return self
    
    def convertMatlabArrayToNumpy(matlabArray):
        npArr = np.array(matlabArray._data).reshape(matlabArray.size, order='F')
        return npArr
    
    def convertNumpyArrayToMatlab(numpyArray):
        mlArr = mlarray.double(numpyArray.tolist())
        return mlArr
    
    def convertMatlabDictToNumpy(matlabDict):
        numpyDict = {}
        for key, val in matlabDict.items():
            if type(val) is double:
                numpyDict[key] = MatlabConversion.convertMatlabArrayToNumpy(val)
            elif type(val) is dict:
                numpyDict[key] = MatlabConversion.convertMatlabDictToNumpy(val)
            
        return numpyDict
            