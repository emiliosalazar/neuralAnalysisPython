#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 15:07:23 2020

@author: emilio
"""

from methods.GeneralMethods import loadDefaultParams
from methods.GeneralMethods import saveFiguresToPdf
from matplotlib import pyplot as plt

from classes.Dataset import Dataset
from setup.DataJointSetup import DatasetInfo, DatasetGeneralLoadParams
import datajoint as dj


from pathlib import Path

import numpy as np
import re
import dill as pickle # because this is what people seem to do?
import hashlib
import json

defaultParams = loadDefaultParams(defParamBase = ".")
dataPath = defaultParams['dataPath']

data = []
data.append({'description': 'Earl 2019-03-18 M1 - MGR',
              'area' : 'M1',
              'path': Path('memoryGuidedReach/Earl/2019/03/18/'),
              'keyStates' : {
                  'delay': 'Delay Period',
                  'stimulus': 'Target Flash',
                  'action' : 'Target Reach'
              },
              'alignmentStates': [],
              'processor': 'Erinn'});
data.append({'description': 'Earl 2019-03-22 M1 - MGR',
              'area' : 'M1',
              'path': Path('memoryGuidedReach/Earl/2019/03/22/'),
              'keyStates' : {
                  'delay': 'Delay Period',
                  'stimulus': 'Target Flash',
                  'action' : 'Target Reach'
              },
              'alignmentStates': [],
              'processor': 'Erinn'});
data.append({'description': 'Earl 2019-03-23 M1 - MGR',
              'area' : 'M1',
              'path': Path('memoryGuidedReach/Earl/2019/03/23/'),
              'keyStates' : {
                  'delay': 'Delay Period',
                  'stimulus': 'Target Flash',
                  'action' : 'Target Reach'
              },
              'alignmentStates': [],
              'processor': 'Erinn'});
data.append({'description': 'Pepe A1 2018-07-14 PFC - MGS',
              'area' : 'PFC',
              'path': Path('memoryGuidedSaccade/Pepe/2018/07/14/Array1_PFC/'),
              'keyStates' : {
                  'delay': 'TARG_OFF',
                  'stimulus': 'TARG_ON',
                  'action' : 'SACCADE'
              },
              'alignmentStates': ['SOUND_CHANGE','ALIGN'],
              'processor': 'Emilio'});
data.append({'description': 'Pepe A2 2018-07-14 PFC - MGS',
              'area' : 'PFC',
              'path': Path('memoryGuidedSaccade/Pepe/2018/07/14/Array2_PFC/'),
              'keyStates' : {
                  'delay': 'TARG_OFF',
                  'stimulus': 'TARG_ON',
                  'action' : 'SACCADE'
              },
              'alignmentStates': ['SOUND_CHANGE','ALIGN'],
              'processor': 'Emilio'});
data.append({'description': 'Wakko A1 2018-02-11 PFC - MGS',
              'area' : 'PFC',
              'path': Path('memoryGuidedSaccade/Wakko/2018/02/11/Array1_PFC/'),
              'keyStates' : {
                  'delay': 'TARG_OFF',
                  'stimulus': 'TARG_ON',
                  'action' : 'SACCADE'
              },
              'alignmentStates': ['SOUND_CHANGE','ALIGN'],
              'processor': 'Emilio'});
data.append({'description': 'Wakko A2 2018-02-11 PFC - MGS',
              'area' : 'PFC',
              'path': Path('memoryGuidedSaccade/Wakko/2018/02/11/Array2_PFC/'),
              'keyStates' : {
                  'delay': 'TARG_OFF',
                  'stimulus': 'TARG_ON',
                  'action' : 'SACCADE'
              },
              'alignmentStates': ['SOUND_CHANGE','ALIGN'],
              'processor': 'Emilio'});
data.append({'description': 'Pepe 2016-02-02 V4 - cuedAttn',
              'area' : 'V4',
              'path': Path('cuedAttention/Pepe/2016/02/02/Array1_V4/'),
              'keyStates' : {
                  'delay': 'Blank Before',
                  'stimulus': 'Target',
                  'action' : 'Success'
              },
              'alignmentStates': ['SOUND_CHANGE','ALIGN'],
              'processor': 'Emilio'});
data.append({'description': 'Pepe 2016-02-02 PFC - cuedAttn',
              'area' : 'PFC',
              'path': Path('cuedAttention/Pepe/2016/02/02/Array2_PFC/'),
              'keyStates' : {
                  'delay': 'Blank Before',
                  'stimulus': 'Target',
                  'action' : 'Success'
              },
              'alignmentStates': ['SOUND_CHANGE','ALIGN'],
              'processor': 'Emilio'});
data.append({'description': 'Pepe 2016-02-03 V4 - cuedAttn',
              'area' : 'V4',
              'path': Path('cuedAttention/Pepe/2016/02/03/Array1_V4/'),
              'keyStates' : {
                  'delay': 'Blank Before',
                  'stimulus': 'Target',
                  'action' : 'Success'
              },
              'alignmentStates': ['SOUND_CHANGE','ALIGN'],
              'processor': 'Emilio'});
data.append({'description': 'Pepe 2016-02-03 PFC - cuedAttn',
              'area' : 'PFC',
              'path': Path('cuedAttention/Pepe/2016/02/03/Array2_PFC/'),
              'keyStates' : {
                  'delay': 'Blank Before',
                  'stimulus': 'Target',
                  'action' : 'Success'
              },
              'alignmentStates': ['SOUND_CHANGE','ALIGN'],
              'processor': 'Emilio'});

data = np.asarray(data) # to allow fancy indexing
#%% process data
processedDataMat = 'processedData.mat'

# dataset = [ [] for _ in range(len(data))]
dataUseLogical = np.zeros(len(data), dtype='bool')
dataIndsProcess = np.arange(len(data))#np.array([1,6])#np.array([1,4,6])
dataUseLogical[dataIndsProcess] = True

removeCoincidentChans = True
coincidenceTime = 1 #ms
coincidenceThresh = 0.2 # 20% of spikes
checkNumTrls = 0.1 # use 10% of trials
datasetGeneralLoadParams = {
    'remove_coincident_chans' : removeCoincidentChans,
    'coincidence_time' : coincidenceTime, 
    'coincidence_thresh' : coincidenceThresh, 
    'coincidence_fraction_trial_test' : checkNumTrls
}
dsgl = DatasetGeneralLoadParams()
if len(dsgl & datasetGeneralLoadParams)>1:
    raise Exception('multiple copies of the same parameters have been saved... I thought I avoided that')
elif len(dsgl & datasetGeneralLoadParams)>0:
    genParamId = (dsgl & datasetGeneralLoadParams).fetch1('ds_gen_params_id')
else:
    # tacky, but it doesn't return the new id, syoo...
    currIds = dsgl.fetch('ds_gen_params_id')
    DatasetGeneralLoadParams().insert1(datasetGeneralLoadParams)
    newIds = dsgl.fetch('ds_gen_params_id')
    genParamId = list(set(newIds) - set(currIds))[0]

datasetDill = 'dataset.dill'
dsiLoadParamsJson = json.dumps(datasetGeneralLoadParams, sort_keys=True) # needed for consistency as dicts aren't ordered
dsiLoadParamsHash = hashlib.md5(dsiLoadParamsJson.encode('ascii')).hexdigest()

for dataUse in data[dataUseLogical]:
# for ind, dataUse in enumerate(data):
#    dataUse = data[1]
    # dataUse = data[ind]
    dataMatPath = dataPath / dataUse['path'] / processedDataMat

    dataDillPath = dataPath / dataUse['path'] / ('dataset_%s' % dsiLoadParamsHash[:5]) / datasetDill
    
    if dataUse['processor'] == 'Erinn':
        notChan = np.array([31, 0])
    else:
        notChan = np.array([])

    if dataDillPath.exists():
        print('loading dataset ' + dataUse['description'])
        with dataDillPath.open(mode='rb') as datasetDillFh:
            datasetHere = pickle.load(datasetDillFh)
#        setattr(datasetHere, 'keyStates', dataUse['keyStates'])
#        with dataDillPath.open(mode='wb') as datasetDillFh:
#            pickle.dump(datasetHere, datasetDillFh)
            
    else:
        print('processing data set ' + dataUse['description'])
        datasetHere = Dataset(dataMatPath, dataUse['processor'], notChan = notChan, checkNumTrls=checkNumTrls, metastates = dataUse['alignmentStates'], keyStates = dataUse['keyStates'])

        
        # first, removed explicitly ignored channels
        if notChan is not None:
            datasetHere.removeChannels(notChan)

        # now, initialize a logical with all the channels
        chansKeepLog = datasetHere.maskAgainstChannels([])

        # remove non-coincident spikes
        if removeCoincidentChans:
            chansKeepLog &= datasetHere.findCoincidentSpikeChannels(coincidenceTime=coincidenceTime, coincidenceThresh=coincidenceThresh, checkNumTrls=checkNumTrls)[0]

        chansKeepNums = np.where(chansKeepLog)[0]
        datasetHere.keepChannels(chansKeepNums)

        dataDillPath.parent.mkdir(parents=True, exist_ok = True)
        with dataDillPath.open(mode='wb') as datasetDillFh:
            pickle.dump(datasetHere, datasetDillFh)

    datasetHash = datasetHere.hash().hexdigest()
    # do some database insertions here
    datasetHereInfo = {
        'dataset_relative_path' : str(dataDillPath.relative_to(dataPath)),
        'dataset_hash' : datasetHash,
        'dataset_name' : dataUse['description'],
        'ds_gen_params_id' : genParamId,
        'processor_name' : dataUse['processor'],
        'brain_area' : dataUse['area'],
        'task' : Path(dataUse['path']).parts[0],
        'date_acquired' : re.search('.*?(\d+-\d+-\d+).*', dataUse['description']).group(1),
    }

    dsi = DatasetInfo()
    if len(dsi[datasetHereInfo]) > 1:
        raise Exception('multiple copies of same dataset in the table...')
    elif len(dsi & datasetHereInfo) > 0:
        dsId = (dsi & datasetHereInfo).fetch1('dataset_id')
    else:
        dsId = len(dsi) # 0 indexed
        datasetHereInfo.update({
            'dataset_id' : dsId,
            'explicit_ignore_channels' : notChan,
            'channels_keep' : chansKeepNums
        })
        dsi.insert1(datasetHereInfo)


    datasetHere.id = dsId
    dataUse['dataset'] = datasetHere
    

##%% get desired time segment
#from classes.BinnedSpikeSet import BinnedSpikeSet
#from methods.BinnedSpikeSetListMethods import generateBinnedSpikeListsAroundState as genBSLAroundState
#binnedSpikes = []
#binnedSpikesAll = []
#binnedSpikesOnlyDelay = []
#binnedSpikesEnd = []
#groupedSpikesTrialAvg = []
#groupedSpikesEndTrialAvg = []
#grpLabels = []
#
## this'll bleed a little into the start of the new stimulus with the offset,
## but before any neural response can happen
#lenSmallestTrl = 301 #ms; 
#furthestBack = 300 #ms
#furthestForward = 300
#binSizeMs = 25 # good for PFC LDA #50 # 
#
#trialType = 'successful'
#
#stateNamesDelayStart = [data[ind]['keyStates']['delay'] for ind in dataIndsProcess]
#
#binnedSpikes, _ = genBSLAroundState(data[dataIndsProcess],
#                                    stateNamesDelayStart,
#                                    trialType = trialType,
#                                    lenSmallestTrl=lenSmallestTrl, 
#                                    binSizeMs = binSizeMs, 
#                                    furthestTimeBeforeState=furthestBack, 
#                                    furthestTimeAfterState=furthestForward,
#                                    setStartToDelayEnd = False,
#                                    setEndToDelayStart = True)
#
#
#binnedSpikesAll, _ = genBSLAroundState(data[dataIndsProcess],
#                                    stateNamesDelayStart,
#                                    trialType = trialType,
#                                    lenSmallestTrl=lenSmallestTrl, 
#                                    binSizeMs = binSizeMs, 
#                                    furthestTimeBeforeState=furthestBack, 
#                                    furthestTimeAfterState=furthestForward,
#                                    setStartToDelayEnd = False,
#                                    setEndToDelayStart = False)
#
#binnedSpikesOnlyDelay, _ = genBSLAroundState(data[dataIndsProcess],
#                                    stateNamesDelayStart,
#                                    trialType = trialType,
#                                    lenSmallestTrl=lenSmallestTrl, 
#                                    binSizeMs = binSizeMs, 
#                                    furthestTimeBeforeState=0, 
#                                    furthestTimeAfterState=0,
#                                    setStartToDelayEnd = False,
#                                    setEndToDelayStart = False)
#
#binnedSpikeEnd, _ = genBSLAroundState(data[dataIndsProcess],
#                                    stateNamesDelayStart,
#                                    trialType = trialType,
#                                    lenSmallestTrl=lenSmallestTrl, 
#                                    binSizeMs = binSizeMs, 
#                                    furthestTimeBeforeState=furthestBack, 
#                                    furthestTimeAfterState=furthestForward,
#                                    setStartToDelayEnd = True,
#                                    setEndToDelayStart = False)
#
#binnedSpikesShortStart, _ = genBSLAroundState(data[dataIndsProcess],
#                                    stateNamesDelayStart,
#                                    trialType = trialType,
#                                    lenSmallestTrl=lenSmallestTrl, 
#                                    binSizeMs = binSizeMs, 
#                                    furthestTimeBeforeState=0, 
#                                    furthestTimeAfterState=lenSmallestTrl,
#                                    setStartToDelayEnd = False,
#                                    setEndToDelayStart = True)
#
#binnedSpikesShortEnd, _ = genBSLAroundState(data[dataIndsProcess],
#                                    stateNamesDelayStart,
#                                    trialType = trialType,
#                                    lenSmallestTrl=lenSmallestTrl, 
#                                    binSizeMs = binSizeMs, 
#                                    furthestTimeBeforeState=lenSmallestTrl, 
#                                    furthestTimeAfterState=0,
#                                    setStartToDelayEnd = True,
#                                    setEndToDelayStart = False)
#
## NOTE: this one is special because it returns *residuals*
#offshift = 75 #ms
#firingRateThresh = 1
#fanoFactorThresh = 4
#baselineSubtract = True
#binnedResidualsShortStartOffshift, chFanosResidualsShortStartOffshift = genBSLAroundState(
#                                    data[dataIndsProcess],
#                                    stateNamesDelayStart,
#                                    trialType = trialType,
#                                    lenSmallestTrl=lenSmallestTrl, 
#                                    binSizeMs = binSizeMs, 
#                                    furthestTimeBeforeState=-offshift, # note that this starts it *forwards* from the delay
#                                    furthestTimeAfterState=lenSmallestTrl+offshift,
#                                    setStartToDelayEnd = False,
#                                    setEndToDelayStart = True,
#                                    returnResiduals = baselineSubtract,
#                                    removeBadChannels = True,
#                                    unitsOut = 'count', # this shouldn't affect GPFA... but will affect fano factor cutoffs...
#                                    firingRateThresh = firingRateThresh,
#                                    fanoFactorThresh = fanoFactorThresh # suggestion of an okay value (not too conservative as with 8, not too lenient as with 1)
#                                    )
#baselineSubtract = False
#furthestTimeBeforeDelay=-offshift # note that this starts it *forwards* from the delay
#furthestTimeAfterDelay=lenSmallestTrl+offshift
#binnedSpikesShortStartOffshift, _ = genBSLAroundState(data[dataIndsProcess],
#                                    stateNamesDelayStart,
#                                    trialType = trialType,
#                                    lenSmallestTrl=lenSmallestTrl, 
#                                    binSizeMs = binSizeMs, 
#                                    furthestTimeBeforeState=-offshift, # note that this starts it *forwards* from the delay
#                                    furthestTimeAfterState=lenSmallestTrl+offshift,
#                                    setStartToDelayEnd = False,
#                                    setEndToDelayStart = True,
#                                    returnResiduals = baselineSubtract,
#                                    removeBadChannels = True,
#                                    unitsOut = 'count', # this shouldn't affect GPFA... but will affect fano factor cutoffs...
#                                    firingRateThresh = firingRateThresh,
#                                    fanoFactorThresh = fanoFactorThresh # suggestion of an okay value (not too conservative as with 8, not too lenient as with 1)
#                                    )
#
#breakpoint()
#baselineSubtract = False
#furthestTimeBeforeState=300 # note that this starts it *forwards* from the delay
#furthestTimeAfterState=300
#dtExtract = dsi['brain_area="V4"'].grabDatasets()
#stateNamesTarget = ['Target']*len(dtExtract)
#binnedSpikesShortStartOffshift, _ = genBSLAroundState(dtExtract,
#                                    stateNamesTarget,
#                                    trialType = trialType,
#                                    lenSmallestTrl=0, 
#                                    binSizeMs = binSizeMs, 
#                                    furthestTimeBeforeState=-300, # note that this starts it *forwards* from the delay
#                                    furthestTimeAfterState=lenSmallestTrl+offshift,
#                                    setStartToDelayEnd = False,
#                                    setEndToDelayStart = False,
#                                    returnResiduals = baselineSubtract,
#                                    removeBadChannels = True,
#                                    unitsOut = 'count', # this shouldn't affect GPFA... but will affect fano factor cutoffs...
#                                    firingRateThresh = firingRateThresh,
#                                    fanoFactorThresh = fanoFactorThresh # suggestion of an okay value (not too conservative as with 8, not too lenient as with 1)
#                                    )

breakpoint()
