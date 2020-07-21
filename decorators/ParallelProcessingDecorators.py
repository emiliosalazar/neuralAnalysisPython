#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 11:12:18 2020

@author: emilio
"""
import functools

def multiprocessNumpy(func):
    @functools.wraps(func)
    def multiprocessSetup(*args, **kwargs):

        # In order to ensure numpy processes work well in a multiprocess
        # environment, we set these environment variables (they need to exist
        # both when numpy gets loaded for the first time *AND* whenever the
        # processes are running
        os.environ['OPENBLAS_NUM_THREADS'] = '1'
        os.environ['GOTO_NUM_THREADS'] = '1'
        os.environ['OMP_NUM_THREADS'] = '1'
            
        funcRet = func(*args, **kwargs)
        # To prevent these variables from causing other methods/classes to load
        # numpy in a non-'optimized' (parallel) state, we delete (effectively
        # unset) the variables after our multiprocessing has finished
        del os.environ['OPENBLAS_NUM_THREADS']
        del os.environ['GOTO_NUM_THREADS']
        del os.environ['OMP_NUM_THREADS']
        
        return funcRet
    
    return multiprocessSetup
