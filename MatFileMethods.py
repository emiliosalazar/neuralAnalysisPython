#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 13:54:27 2020

@author: emilio
"""

# from scipy.io import loadmat as tradLoadmat
# import h5py
#from numpy import ndarray, stack
# import numpy as np
import hdf5storage as hdf5
from pathlib import Path

import pickle

def LoadMatFile(matfilePath, pickleOnLoad):
    

    out = hdf5.loadmat(str(matfilePath))
    # if pickleOnLoad:
    #     picklePath = matfilePath.with_suffix(".pickle")
    #     outPick = pickle.Pickler(picklePath.open(mode = 'wb'))
    #     outPick.dump(out)
        
    return out

    
                
# forceLoad is meant to be used if you want to use the specific file pointed to
# not matter what, as opposed to finding the pickled version
def LoadDataset(filepathObj, forceLoad=False, pickleOnLoad = True):
    
    # if not forceLoad:
    #     fileSuffix = filepathObj.suffix
    #     if not fileSuffix:
    #         fileDir = filepathObj
    #     else:
    #         fileDir = filepathObj.parent
            
    #     filename = filepathObj.stem
    #     pickFls = fileDir.glob('*.pickle')
        
        
    #     fileFound = False
    #     for pickFl in pickFls:
    #         fileUse = Path(pickFl)
    #         if fileUse.stem == filename:
    #             fileFound = True
    #             break
    #         else:
    #             fileFound = False
        
    #     if not fileFound:
    #         out = LoadMatFile(filepathObj, pickleOnLoad)
    #     else:
    #         outUnpick = pickle.Unpickler(fileUse.open('rb'))
    #         out = outUnpick.load()
            
    # else:
        # if filepathObj.suffix == '*.mat':
    out = LoadMatFile(filepathObj, pickleOnLoad)
        # elif filepathObj.suffix == '*.pickle':
        #     outUnpick = pickle.Unpickler(filepathObj.open('rb'))
        #     out = outUnpick.load()
            
    return out