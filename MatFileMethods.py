#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 13:54:27 2020

@author: emilio
"""

from scipy.io import loadmat as tradLoadmat
import h5py
#from numpy import ndarray, stack
import numpy as np
import hdf5storage as hdf5

def LoadMatFile(matfilePath):
    
#    def LoadHdf5Mat(matfilePath):
#        def UnpackHdf5(hdf5Matfile, hdf5Group):
#            out = {}
#            if type(hdf5Group) is h5py._hl.group.Group:
#                for key in hdf5Group:
#                    out[key] = hdf5Group[key]
#                    if type(out[key]) is h5py._hl.group.Group:
#                        out[key] = UnpackHdf5(hdf5Matfile,out[key])
#                    elif type(out[key]) is h5py._hl.dataset.Dataset:
#                        out[key] = UnpackHdf5(hdf5Matfile,out[key])
#                    elif type(out[key]) is h5py.h5r.Reference:
#                        out[key] = UnpackHdf5(hdf5Matfile,out[key])
#                            
#                    
#                        
#                        
##                        valsLambda = (lambda hMF=hdf5Matfile, hG=hdf5Group, k=key: [hMF[hG[k][row, col]] for row in range(hG[k].shape[0]) for col in range(hG[k].shape[1])])
##                        valsList = list(valsLambda())
##                        for idx, val in enumerate(valsList):
##                            if type(val) is h5py._hl.group.Group:
##                                valsList[idx] = UnpackHdf5(hdf5Matfile,val)
##                            elif 'MATLAB_empty' in val.attrs.keys(): # deal with empty arrays
##                                valsList[idx] = np.ndarray(0)
##                            elif np.any([type(nestedVal) is h5py.h5r.Reference for nestedVal in val[()][0]]):
##                                valsList[idx] = np.ndarray(val.shape)
##                                for row in range(val.shape[0]):
##                                    for col in range(val.shape[1]):
##                                        #if valsList[idx] = val[()]
##                                        if type(val[()][row, col]) is h5py.h5r.Reference:
##                                            valsList[idx][row,col] = UnpackHdf5(hdf5Matfile,hdf5Group[val[()][row,col]])
##                                        else:
##                                            valsList[idx][row,col] = val[()][row,col]
##                            else:
##                                valsList[idx] = val[()]
##                        out[key] = 5
#            elif type(hdf5Group) is h5py._hl.dataset.Dataset:
#                out = np.ndarray(hdf5Group.shape, dtype=object)
##                out = np.frompyfunc(list, 0, 1)(np.empty(hdf5Group.shape, dtype=object))
#                with np.nditer([out, hdf5Group], ['refs_ok'], [['writeonly'], ['readonly']]) as iterRef:
#                    #while not iterRef.finished:
#                    for valOut, valIn in iterRef:
#                        if type(valIn[()]) is h5py._hl.group.Group:
#                            valOut[()] = UnpackHdf5(hdf5Matfile, valIn[()])
##                            valOut[()].append(UnpackHdf5(hdf5Matfile, valIn[()]))
#                        elif type(valIn[()]) is h5py._hl.dataset.Dataset:
#                            valOut[()] = UnpackHdf5(hdf5Matfile, valIn[()])
##                            valOut[()].append(UnpackHdf5(hdf5Matfile, valIn[()]))
#                        elif type(valIn[()]) is h5py.h5r.Reference:
#                            valOut[()] = UnpackHdf5(hdf5Matfile, valIn[()])
##                            valOut[()].append(UnpackHdf5(hdf5Matfile, valIn[()]))
#                        else:
#                            valOut[()] = valIn[()]
##                            valOut[()].append(valIn[()])
#                            
#                        #iterRef.iternext()
#                      
#                   
#                    out = iterRef.operands[0]
##                for row in range(out.shape[0]):
##                    for col in range(out.shape[1]):
##                       #if valsList[idx] = val[()]
##                       if type(out[row, col]) is h5py.h5r.Reference:
##                           out[row,col] = UnpackHdf5(hdf5Matfile, out[row,col])
##                       elif type(out[row, col]) is h5py._hl.group.Group:
##                           out[row,col] = UnpackHdf5(hdf5Matfile, out[row,col])
##                       else:
##                           out[row,col] = out[row,col]
#            elif type(hdf5Group) is h5py.h5r.Reference:
#                deref = hdf5Matfile[hdf5Group]
#                try:
#                    out = np.ndarray(deref.shape, dtype=object)
#                except (AttributeError) as err:
#                    if type(deref) is h5py._hl.group.Group:
#                        out = UnpackHdf5(hdf5Matfile, deref)
#                    else:
#                        raise RuntimeError('problem with forming iterator of a non-group a')
#                        
#                else:
##                    with np.nditer(out, ['refs_ok'], ['readwrite']) as iterRef:
#                    with np.nditer([out, deref], ['refs_ok'], [['writeonly'], ['readonly']]) as iterRef:
#                        for valOut, valIn in iterRef:
#                            if type(valIn[()]) is h5py._hl.group.Group:
#                                valOut[()] = UnpackHdf5(hdf5Matfile, valIn[()])
#                            elif type(valIn[()]) is h5py._hl.dataset.Dataset:
#                                valOut[()] = UnpackHdf5(hdf5Matfile, valIn[()])
#                            elif type(valIn[()]) is h5py.h5r.Reference:
#                                valOut[()] = UnpackHdf5(hdf5Matfile, valIn[()])
#                            else:
#                                if 'MATLAB_empty' in deref.attrs.keys(): # deal with empty arrays
#                                    valOut[()] = np.ndarray(0)
#                                else:
#                                    valOut[()] = valIn[()]
#                        out = iterRef.operands[0]
##                except (AttributeError, TypeError) as err:
##                    if type(out) is h5py._hl.group.Group:
##                        out = np.asarray(UnpackHdf5(hdf5Matfile, deref))
##                    else:
##                        raise RuntimeError('problem with forming iterator of a non-group b')
#                        
##            elif type(hdf5Group) is h5py.h5r.Reference:
##                out = hdf5Matfile[hdf5Group]
##                iterRef = np.nditer(out, ['refs_ok'])
##                for val in iterRef:
##                    if type(val[()]) is h5py._hl.group.Group:
##                        UnpackHdf5(hdf5Matfile, val[()])
##                    elif type(val[()]) is h5py._hl.dataset.Dataset:
##                        UnpackHdf5(hdf5Matfile, val[()])
##                    elif type(val[()]) is h5py.h5r.Reference:
##                        UnpackHdf5(hdf5Matfile, val[()])
##                    else:
##                        if 'MATLAB_empty' in out.attrs.keys(): # deal with empty arrays
##                            np.ndarray(0)
##                        else:
##                            val[()]
##                out = iterRef.operands[0]
#            
#            return out
#        
#        hdf5Matfile = h5py.File(matfilePath)
#        
#        out = {}
#    
#        # this loop looks very similar to that in unpacking the group, but it
#        # specifically ignores the #refs# key... I'm also not sure its terminal
#        # condition is quite right, as I don't know if a non-structure variable
#        # is saved as a Dataset at the top of the hierarchy--I assume so?
#        for key in hdf5Matfile:
#            if key == '#refs#':
#                pass
#            
#            elif type(hdf5Matfile[key]) is h5py._hl.group.Group:
#                out[key] = UnpackHdf5(hdf5Matfile, hdf5Matfile[key])
#            elif type(hdf5Matfile[key]) is h5py._hl.dataset.Dataset:
#                out[key] = UnpackHdf5(hdf5Matfile, hdf5Matfile[key])
#                
#        return out
                    
                    
        
        
#    try:
#        annots = tradLoadmat(matfilePath)
#    except NotImplementedError:
#        annots = LoadHdf5Mat(matfilePath)
    out = hdf5.loadmat(matfilePath)
    return out

    
                
            
    #trlDat = annots['Data']['TrialData']
    #res = {trlDat[0][0][0][0][i]: trlDat[0][0][0][0].dtype.names[i] for i in range(len(trlDat[0][0][0][0]))}
    
#def QuashStructure(matStruct):
#    if isinstance(matStruct, dict): # this happens with the first pass
#        matStructOut = {}
#        for key in list(matStruct):
#            if key.find('__') == -1:
#                matStructOut[key] = QuashStructure(matStruct[key])
#        return matStructOut
#    elif type(matStruct) is ndarray:
#        if matStruct.dtype.names is None:
#            matStructOut = {}
#            ind = 0
#            for matStructSlice in matStruct:
#                try:
#                    # NOTE: doesn't quite work if an array's the last thing in here...
#                    #checker = matStructSlice[0][0]
#                    matStructOut[ind] = QuashStructure(matStructSlice[0]) 
#                except IndexError:
#                    matStructOut[ind] = matStructSlice
#                ind+=1
#            
##            matStructOut = [QuashStructure(matStructSlice) for matStructSlice in matStruct]
##            try:
##                matStructOut = QuashStructure(matStruct[0])
##            except IndexError:
##                matStructOut = matStruct
#            
#            return matStructOut
#        else:
#            matStructOut = {}
#            for name in matStruct.dtype.names:
#                matStructOut[name] = QuashStructure(matStruct[name])
#            return matStructOut
#    else:
#        return matStruct