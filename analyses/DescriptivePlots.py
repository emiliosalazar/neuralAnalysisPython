"""
This function is responsible for outputting descriptive plots of datasets--useful for checking new datasets to see that they conform to what's expected;
Some of what's on here is like the olde HeadlessAnalysisRun4.py for now...
"""
from matplotlib import pyplot as plt
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['GOTO_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np
import re
import dill as pickle # because this is what people seem to do?
import hashlib
import json
from pathlib import Path
from scipy.stats import binned_statistic

from methods.GeneralMethods import saveFiguresToPdf
# database stuff
from setup.DataJointSetup import DatasetInfo, DatasetGeneralLoadParams, BinnedSpikeSetInfo, FilterSpikeSetParams
import datajoint as dj
# the decorator to save these to a database!
from decorators.AnalysisCallDecorators import saveCallsToDatabase

from classes.Dataset import Dataset
from classes.BinnedSpikeSet import BinnedSpikeSet


# for generating the binned spike sets
from methods.BinnedSpikeSetListMethods import generateBinnedSpikeListsAroundState as genBSLAroundState

# subsampling
from methods.BinnedSpikeSetListMethods import subsampleBinnedSpikeSetsToMatchNeuronsAndTrialsPerCondition as subsmpMatchCond

# for the gpfa...
from methods.BinnedSpikeSetListMethods import gpfaComputation

# for descriptions of the data
from methods.BinnedSpikeSetListMethods import generateLabelGroupStatistics as genBSLLabGrp
from methods.BinnedSpikeSetListMethods import plotExampleChannelResponses
from methods.BinnedSpikeSetListMethods import plotStimDistributionHistograms
from methods.BinnedSpikeSetListMethods import plotFiringRates

# for computing metrics
from methods.BinnedSpikeSetListMethods import rscComputations
from methods.GpfaMethods import computePopulationMetrics

# for plotting the metrics
from methods.plotUtils.UnsortedPlotMethods import plotAllVsAll, plotMetricVsExtractionParams, plotMetricsBySeparation


#@saveCallsToDatabase
def datasetDescriptiveOverview(datasetSqlFilter, binnedSpikeSetGenerationParamsDict, plotParams):

    dsgl = DatasetGeneralLoadParams()
    dsi = DatasetInfo()
    bsi = BinnedSpikeSetInfo()
    fsp = FilterSpikeSetParams()

    outputFiguresRelativePath = []

    if len(datasetSqlFilter)>0:
        dsiUse = dsi[datasetSqlFilter]
    else:
        dsiUse = dsi

    print("Computing/loading binned spike sets")
    _, bssKeys = genBSLAroundState(dsiUse,
                                        **binnedSpikeSetGenerationParamsDict
                                        )

    breakpoint()

    if plotParams['plotChannelRespParams']:
        plotChannelRespParams = plotParams['plotChannelRespParams']
        for bssKeyOne in bssKeys:
            bssExp = bsi[bssKeyOne]
            dsID = dsi[bssKeyOne]['dataset_name'][0].replace(' ', '_')
            pdfName = dsID + '_' + bssExp['bss_hash'][0][:5]
            bssExp.generateDescriptivePlots(plotTypes = plotChannelRespParams)
            saveFiguresToPdf(pdfname=pdfName, analysisDescription="descriptivePlots")
            plt.close('all')
