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

        gitRepo = sh.git.bake('--no-pager', _cwd = rootPath) # a level up from decorators is now baked in ... >.>
        diffRepo = str(gitRepo.diff('--no-color'))
        gitHash = str(gitRepo.log("--pretty=format:%H", "-n 1"))

        # this is smaller than the varchar size in the db!
        if len(diffRepo) > 100000:
            raise Exception("Commit yo' shit!")

        runStartTime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        funcRet = func(*args, **kwargs)
        runEndTime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')


        asiInsertInfo = dict(
            analysis_method = analysisMethod,
            method_call_signature = str(analysisMethod) + str(callSignature),
            git_commit = gitHash,
            patch = diffRepo,
            method_inputs = methodInputs,
            date_start = runStartTime,
            date_end = runEndTime,
        )

        if funcRet is not None:
            asiInsertInfo.update(dict(output_figures_relative_path = funcRet['outputFiguresRelativePath'])) if 'outputFiguresRelativePath' in funcRet else None
            asiInsertInfo.update(dict(output_files = funcRet['outputFiles'])) if 'outputFiles' in funcRet else None
            asiInsertInfo.update(dict(output_files = funcRet['metadata'])) if 'metadata' in funcRet else None

        try:
            ari.insert1(dict(
                **asiInsertInfo
            ))
        except Exception as err:
            hah = err
            breakpoint()
            # handle unique exceptions here...

        return funcRet
    
    return saveCallToDb
