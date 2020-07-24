#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 11:12:18 2020

@author: emilio
"""
import functools
import inspect
import sh
from datetime import datetime

from setup.DataJointSetup import AnalysisRunInfo
from methods.GeneralMethods import loadDefaultParams



def saveCallsToDatabase(func):

    defaultParams = loadDefaultParams()
    rootPath = defaultParams['rootCodePath']

    @functools.wraps(func)
    def saveCallToDb(*args, **kwargs):
        ari = AnalysisRunInfo()

        analysisMethod = func.__name__
        callSignature = inspect.signature(func)
        methodInputs = inspect.getcallargs(func, *args, **kwargs)

        runStartTime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        funcRet = func(*args, **kwargs)
        runEndTime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        gitRepo = sh.git.bake('--no-pager', _cwd = rootPath) # a level up from decorators is now baked in ... >.>
        diffRepo = gitRepo.diff('--no-color')
        gitHash = gitRepo.log("--pretty=format:%H", "-n 1")

        if funcRet is not None:
            outputFiles = funcRet['outputFiles'] if 'outputFiles' in funcRet else None
        else:
            outputFiles = None
#        ari.insert1(dict(
#            analysis_method = analysisMethod,
#            method_call_signature = str(analysisMethod) + str(callSignature),
#            git_commit = gitHash,
#            patch = diffRepo,
#            method_inputs = methodInputs,
#            date_start = runStartTime,
#            date_end = runEndTime,
#            output_files = outputFiles,
#            metadata = ,
#        ))
        return funcRet
    
    return saveCallToDb
