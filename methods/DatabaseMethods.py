import numpy as np
from pathlib import Path
from setup.DataJointSetup import DatasetInfo, DatasetGeneralLoadParams
from classes.Dataset import Dataset
from methods.GeneralMethods import prepareMatlab, loadDefaultParams
import dill as pickle # because this is what people seem to do?
import re

def insertDataset(dataMatPath, datasetInfo, removeCoincidentChans=True, coincidenceDetectionParams={'coincidenceTime':1, 'coincidenceThresh':0.2, 'checkNumTrls':1}):
    dsi = DatasetInfo()
    dsgl = DatasetGeneralLoadParams()
    defaultParams = loadDefaultParams()
    dataPath = defaultParams['dataPath']

    desc = datasetInfo['description']
    print(f'processing and inserting dataset "{desc}" into database')
    datasetHere = Dataset(dataMatPath, datasetInfo['processor'], metastates = datasetInfo['alignmentStates'], keyStates = datasetInfo['keyStates'], keyStateEnds = datasetInfo['keyStateEnds'], stateWithAngleName = datasetInfo['stateWithAngleName'] if 'stateWithAngleName' in datasetInfo else None)

    if datasetInfo['processor'] == 'Erinn':
        removeSort = np.array([31, 0])
    elif datasetInfo['processor']  == 'EmilioJoystick' or datasetInfo['processor'] == 'EmilioJoystickHE' or datasetInfo['processor'] == 'EmilioKalmanBci':
        removeSort = np.array([255])
    else:
        removeSort = np.array([])

    
    # first, remove explicitly ignored channels
    if removeSort is not None:
        datasetHere.removeChannelsWithSort(removeSort)

    # now, initialize a logical with all the channels/sorts (same thing when each channel is considered a neuron)
    chansKeepLog = datasetHere.maskAgainstSort([])

    # remove non-coincident spikes
    if removeCoincidentChans:
        chansKeepLog &= datasetHere.findCoincidentSpikeChannels(**coincidenceDetectionParams)[0]

    chansKeepNums = np.where(chansKeepLog)[0]
    datasetHere.keepChannels(chansKeepNums)

    if datasetInfo['description'].find('Quincy') >= 0:
        trialsRemoveLog = datasetHere.findInitialHighSpikeCountTrials()
        breakpoint()
        datasetHere, trialsKeepLog = datasetHere.filterTrials(~trialsRemoveLog)
        trialsKeepNums = np.where(trialsKeepLog)[0]
    else:
        trialsKeepNums = np.arange(len(datasetHere.spikeDatTimestamps))



    genParamId = retrieveDatasetGeneralLoadParamsId(removeCoincidentChans, **coincidenceDetectionParams)

    datasetDill = 'dataset.dill'
    dsiLoadParamsHash = dsgl[f'ds_gen_params_id={genParamId}'].shorthash()
    dataDillPath = dataPath / datasetInfo['path'] / ('dataset_%s' % dsiLoadParamsHash) / datasetDill
    dataDillPath.parent.mkdir(parents=True, exist_ok = True)
    with dataDillPath.open(mode='wb') as datasetDillFh:
        pickle.dump(datasetHere, datasetDillFh)



    datasetHash = datasetHere.hash().hexdigest()
    # do some database insertions here
    datasetHereInfo = {
        'dataset_relative_path' : str(dataDillPath.relative_to(dataPath)),
        'dataset_hash' : datasetHash,
        'dataset_name' : datasetInfo['description'],
        'ds_gen_params_id' : genParamId,
        'processor_name' : datasetInfo['processor'],
        'brain_area' : datasetInfo['area'],
        'task' : Path(datasetInfo['path']).parts[0],
        'date_acquired' : re.search('.*?(\d+-\d+-\d+).*', datasetInfo['description']).group(1),
        'spike_identification_method' : datasetInfo['spikeIdMethod'],
    }
    if 'nasUsed' in datasetInfo:
        datasetHereInfo.update({
            'nas_used' : datasetInfo['nasUsed'], 
            'nas_gamma' : datasetInfo['nasGamma']
        }) 

    if len(dsi[datasetHereInfo]) > 1:
        raise Exception('multiple copies of same dataset in the table...')
    elif len(dsi & datasetHereInfo) > 0:
        dsId = (dsi & datasetHereInfo).fetch1('dataset_id')
    else:
        dsId = len(dsi) # 0 indexed
        datasetHereInfo.update({
            'dataset_id' : dsId,
            'explicit_ignore_sorts' : removeSort,
            'channels_keep' : chansKeepNums,
            'trials_keep' : trialsKeepNums
        })
        dsi.insert1(datasetHereInfo)
    
    return datasetHereInfo

def retrieveDatasetGeneralLoadParamsId(removeCoincidentChans, coincidenceTime=1, coincidenceThresh=0.2, checkNumTrls=1):
    datasetGeneralLoadParams = {
        'remove_coincident_chans' : removeCoincidentChans,
        'coincidence_time' : coincidenceTime, 
        'coincidence_thresh' : coincidenceThresh, 
        'coincidence_fraction_trial_test' : checkNumTrls
    }
    dsgl = DatasetGeneralLoadParams()
    if len(dsgl[datasetGeneralLoadParams])>1:
        raise Exception('multiple copies of the same parameters have been saved... I thought I avoided that')
    elif len(dsgl[datasetGeneralLoadParams])>0:
        genParamId = dsgl[datasetGeneralLoadParams].fetch1('ds_gen_params_id')
    else:
        # tacky, but it doesn't return the new id, syoo...
        currIds = dsgl.fetch('ds_gen_params_id')
        print('Inserting new entry to DatasetGeneralLoadParams')
        DatasetGeneralLoadParams().insert1(datasetGeneralLoadParams)
        newIds = dsgl.fetch('ds_gen_params_id')
        genParamId = list(set(newIds) - set(currIds))[0]

    return genParamId