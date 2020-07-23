#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 13:54:27 2020

@author: emilio
"""

from scipy.io import loadmat as tradLoadmat
import h5py
# from numpy import ndarray, stack
import numpy as np
import hdf5storage as hdf5
from pathlib import Path

import pickle

# def checkoutStuff(name, object)

def LoadHdf5Mat(matfilePath):
    # def LookAheadDeref(hdf5File, reference):
        
    def UnpackHdf5(hdf5Matfile, hdf5Group):
        out = {}
        if type(hdf5Group) is h5py._hl.group.Group:
            for key in hdf5Group:
                out[key] = hdf5Group[key]
                if type(out[key]) is h5py._hl.group.Group:
                    out[key] = UnpackHdf5(hdf5Matfile,out[key])
                elif type(out[key]) is h5py._hl.dataset.Dataset:
                    out[key] = UnpackHdf5(hdf5Matfile,out[key])
                elif type(out[key]) is h5py.h5r.Reference:
                    out[key] = UnpackHdf5(hdf5Matfile,out[key])
                        
                
                    
                    
#                        valsLambda = (lambda hMF=hdf5Matfile, hG=hdf5Group, k=key: [hMF[hG[k][row, col]] for row in range(hG[k].shape[0]) for col in range(hG[k].shape[1])])
#                        valsList = list(valsLambda())
#                        for idx, val in enumerate(valsList):
#                            if type(val) is h5py._hl.group.Group:
#                                valsList[idx] = UnpackHdf5(hdf5Matfile,val)
#                            elif 'MATLAB_empty' in val.attrs.keys(): # deal with empty arrays
#                                valsList[idx] = np.ndarray(0)
#                            elif np.any([type(nestedVal) is h5py.h5r.Reference for nestedVal in val[()][0]]):
#                                valsList[idx] = np.ndarray(val.shape)
#                                for row in range(val.shape[0]):
#                                    for col in range(val.shape[1]):
#                                        #if valsList[idx] = val[()]
#                                        if type(val[()][row, col]) is h5py.h5r.Reference:
#                                            valsList[idx][row,col] = UnpackHdf5(hdf5Matfile,hdf5Group[val[()][row,col]])
#                                        else:
#                                            valsList[idx][row,col] = val[()][row,col]
#                            else:
#                                valsList[idx] = val[()]
#                        out[key] = 5
        elif type(hdf5Group) is h5py._hl.dataset.Dataset:
            out = np.ndarray(hdf5Group.shape, dtype=object)
            
            if hdf5Group.dtype == np.dtype('object'):
                
#                out = np.frompyfunc(list, 0, 1)(np.empty(hdf5Group.shape, dtype=object))
                with np.nditer([out, hdf5Group], ['refs_ok'], [['writeonly'], ['readonly']]) as iterRef:
                    #while not iterRef.finished:
                    for valOut, valIn in iterRef:
                        if type(valIn[()]) is h5py._hl.group.Group:
                            valOut[()] = UnpackHdf5(hdf5Matfile, valIn[()])
    #                            valOut[()].append(UnpackHdf5(hdf5Matfile, valIn[()]))
                        elif type(valIn[()]) is h5py._hl.dataset.Dataset:
                            valOut[()] = UnpackHdf5(hdf5Matfile, valIn[()])
    #                            valOut[()].append(UnpackHdf5(hdf5Matfile, valIn[()]))
                        elif type(valIn[()]) is h5py.h5r.Reference:
                            valOut[()] = UnpackHdf5(hdf5Matfile, valIn[()])
    #                            valOut[()].append(UnpackHdf5(hdf5Matfile, valIn[()]))
                        else:
                            valOut[()] = valIn[()]
    #                            valOut[()].append(valIn[()])
                            
                        #iterRef.iternext()
                      
                   
                    out = iterRef.operands[0]
                    out = out.T # undo Matlab's weird transpose when saving...
#                for row in range(out.shape[0]):
#                    for col in range(out.shape[1]):
#                       #if valsList[idx] = val[()]
#                       if type(out[row, col]) is h5py.h5r.Reference:
#                           out[row,col] = UnpackHdf5(hdf5Matfile, out[row,col])
#                       elif type(out[row, col]) is h5py._hl.group.Group:
#                           out[row,col] = UnpackHdf5(hdf5Matfile, out[row,col])
#                       else:
#                           out[row,col] = out[row,col]
            else:
                # apparently type dataset can also store arrays like type
                # reference and I just give up
                #
                # but I'm also renaming this variable to parallel what was done
                # for the reference and perhaps someday I'll make it its own
                # function
                deref = hdf5Group
                if 'MATLAB_empty' in deref.attrs.keys(): # deal with empty arrays
                    print('empty array')
                    print(deref[()])
                    if 'MATLAB_class' in deref.attrs.keys():
                        print(deref.attrs['MATLAB_class'])
                    out = np.ndarray(0)
                    return out.T
                
                if 'MATLAB_int_decode' in deref.attrs.keys():
                    if 'MATLAB_class' in deref.attrs.keys():
                        if deref.attrs['MATLAB_class'] == b'char':
                            out = "".join([chr(ch) for ch in deref[()]])
                            return out
                        elif deref.attrs['MATLAB_class'] == b'logical':
                            pass # uint8, the default, is a fine type for logicals
                        else:
                            print(deref.attrs['MATLAB_class'])
                            print('int decode but class not char...')
                    else:
                        print('int decode but no class?')
                
                out = deref[()]
                out = out.T # for some reason Matlab transposes when saving...
        elif type(hdf5Group) is h5py.h5r.Reference:
            deref = hdf5Matfile[hdf5Group]
            
            if deref.dtype == np.dtype('object'):
                try:
                    out = np.ndarray(deref.shape, dtype=object)
                except (AttributeError) as err:
                    if type(deref) is h5py._hl.group.Group:
                        out = UnpackHdf5(hdf5Matfile, deref)
                    else:
                        raise RuntimeError('problem with forming iterator of a non-group a')
                        
                else:
    #                    with np.nditer(out, ['refs_ok'], ['readwrite']) as iterRef:
                    with np.nditer([out, deref], ['refs_ok'], [['writeonly'], ['readonly']]) as iterRef:
                        for valOut, valIn in iterRef:
                            if type(valIn[()]) is h5py._hl.group.Group:
                                valOut[()] = UnpackHdf5(hdf5Matfile, valIn[()])
                            elif type(valIn[()]) is h5py._hl.dataset.Dataset:
                                valOut[()] = UnpackHdf5(hdf5Matfile, valIn[()])
                            elif type(valIn[()]) is h5py.h5r.Reference:
                                valOut[()] = UnpackHdf5(hdf5Matfile, valIn[()])
                            else:
                                print('non-hdf5 object')
                                if 'MATLAB_empty' in deref.attrs.keys(): # deal with empty arrays
                                    valOut[()] = np.ndarray(0)
                                else:
                                    valOut[()] = valIn[()]
                        out = iterRef.operands[0]
                        out = out.T # undo Matlab's weird transpose when saving...
            else:
                if 'MATLAB_empty' in deref.attrs.keys(): # deal with empty arrays
                    print('empty array')
                    print(deref[()])
                    if 'MATLAB_class' in deref.attrs.keys():
                        print(deref.attrs['MATLAB_class'])
                    out = np.ndarray(0)
                    return out.T
                
                if 'MATLAB_int_decode' in deref.attrs.keys():
                    if 'MATLAB_class' in deref.attrs.keys():
                        if deref.attrs['MATLAB_class'] == b'char':
                            out = "".join([chr(ch) for ch in deref[()]])
                            return out
                        elif deref.attrs['MATLAB_class'] == b'logical':
                            pass # uint8, the default, is a fine type for logicals
                        else:
                            print(deref.attrs['MATLAB_class'])
                            print('int decode but class not char...')
                    else:
                        print('int decode but no class?')
                
                out = deref[()]
                out = out.T # for some reason Matlab transposes when saving...
#                except (AttributeError, TypeError) as err:
#                    if type(out) is h5py._hl.group.Group:
#                        out = np.asarray(UnpackHdf5(hdf5Matfile, deref))
#                    else:
#                        raise RuntimeError('problem with forming iterator of a non-group b')
                    
#            elif type(hdf5Group) is h5py.h5r.Reference:
#                out = hdf5Matfile[hdf5Group]
#                iterRef = np.nditer(out, ['refs_ok'])
#                for val in iterRef:
#                    if type(val[()]) is h5py._hl.group.Group:
#                        UnpackHdf5(hdf5Matfile, val[()])
#                    elif type(val[()]) is h5py._hl.dataset.Dataset:
#                        UnpackHdf5(hdf5Matfile, val[()])
#                    elif type(val[()]) is h5py.h5r.Reference:
#                        UnpackHdf5(hdf5Matfile, val[()])
#                    else:
#                        if 'MATLAB_empty' in out.attrs.keys(): # deal with empty arrays
#                            np.ndarray(0)
#                        else:
#                            val[()]
#                out = iterRef.operands[0]
        
        return out
    
    hdf5Matfile = h5py.File(matfilePath, 'r')
    
    out = {}

    # this loop looks very similar to that in unpacking the group, but it
    # specifically ignores the #refs# key... I'm also not sure its terminal
    # condition is quite right, as I don't know if a non-structure variable
    # is saved as a Dataset at the top of the hierarchy--I assume so?
    for key in hdf5Matfile:
        if key == '#refs#':
            pass
        
        elif type(hdf5Matfile[key]) is h5py._hl.group.Group:
            out[key] = UnpackHdf5(hdf5Matfile, hdf5Matfile[key])
        elif type(hdf5Matfile[key]) is h5py._hl.dataset.Dataset:
            out[key] = UnpackHdf5(hdf5Matfile, hdf5Matfile[key])
            
    return out
 
    

def LoadMatFile(matfilePath):
    
    try:
        annots = tradLoadmat(matfilePath)
    except NotImplementedError:
        annots = LoadHdf5Mat(matfilePath)
    # out = hdf5.loadmat(str(matfilePath))
    # if pickleOnLoad:
    #     picklePath = matfilePath.with_suffix(".pickle")
    #     outPick = pickle.Pickler(picklePath.open(mode = 'wb'))
    #     outPick.dump(out)
        
    return annots

    
                
# forceLoad is meant to be used if you want to use the specific file pointed to
# not matter what, as opposed to finding the pickled version
def LoadDataset(filepathObj, forceLoad=False):
    out = LoadMatFile(filepathObj)
            
    return out
