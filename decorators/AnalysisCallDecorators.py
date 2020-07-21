#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 11:12:18 2020

@author: emilio
"""
import functools
from setup.DataJointSetup import AnalysisRunInfo
import inspect
import sh

def saveCallsToDatabase(func):
    @functools.wraps(func)
    def saveCallToDb(*args, **kwargs):
        ari = AnalysisRunInfo()

        analysisMethod = func.__name__
        callSignature = inspect.signature(func)
        methodInputs = inspect.getcallargs(func, *args, **kwargs)
        funcRet = func(*args, **kwargs)

        gitRepo = sh.git('--no-pager', _cwd = '../') # a level up from decorators is now baked in ... >.>
        diffRepo = gitRepo.diff('--no-color')
        gitHash = gitRepo.log("--pretty=format:%H", "-n 1")

        outputFiles = funcRet['outputFiles']
        ari.insert1(dict(
            analysis_method = analysisMethod,
            method_signature = str(callSignature),
            git_commit = gitHash,
            patch = diffRepo,
            method_inputs = funcRet,
            date = ,
            output_files = outputFiles,
            metadata = ,
        ))
        return funcRet
    
    return saveCallToDb
