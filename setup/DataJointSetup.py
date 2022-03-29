"""
This file will contain all the setup of tables and such that sets up Data
Joint so I can access my data! Woo!
"""

import sys
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from methods.GeneralMethods import loadDefaultParams, userChoice

defaultParams = loadDefaultParams()#defParamBase = ".")
sys.path.append(defaultParams['datajointLibraryPath'])
import datajoint as dj
import dill as pickle
from dill import source as dillSource
import hashlib
import json

from sqlite3 import IntegrityError

dbLocation = defaultParams['databaseHost']
dbPort = defaultParams['databasePort']
dataPath = defaultParams['dataPath']
if dbPort == 'sqlite':
#    dbLocation = 'backup20200624.db'
    dbLocation = str(Path(dataPath) / dbLocation)

dj.config['database.user'] = 'emilio'
dj.config['database.port'] = dbPort
dj.config['database.host'] = dbLocation
dj.config['database.password']=''
dj.config["enable_python_native_blobs"] = True

# NOTE: This was not quite what I needed after all... because I couldn't easily
# search on the link between dataset and this
## we need to allow filepaths to be used as a type... tbd whether this was a bad
## idea as they haven't fully validated it yet heh...
#dj.errors._switch_filepath_types(True)
#dj.config['stores'] = {
#    'processedDatasetFiles' : dict(
#        protocol = 'file',
#        location = dataPath, # this is the 'external' file path, but for me, and probably for sqlite, it should be the same
#        stage = dataPath, # note this seems to be the *local* file path
#    ),
#    'processedBinnedSpikeSetFiles' : dict(
#        protocol = 'file',
#        location = dataPath, # this is the 'external' file path, but for me, and probably for sqlite, it should be the same
#        stage = dataPath, # note this seems to be the *local* file path
#    )
#}

schema = dj.schema('main',locals())


@schema
class DatasetGeneralLoadParams(dj.Manual):
    definition = """
    # dataset info extraction params
    ds_gen_params_id : int auto_increment # params id
    ---
    remove_coincident_chans : enum(0, 1) # flag on whether these were removed
    coincidence_time : int # ms apart spikes need to be to be counted as coincident
    coincidence_thresh : float # fraction of spikes needing to be coincident to remove channel
    coincidence_fraction_trial_test : float # fraction of trials to check (0-1)
    """

    def _update(self, attrname, value=None):
        assert len(self)==1, "Can currently only update one DatasetGeneralLoadParams at a time"
        oldValue = self.fetch(attrname)[0]
        try:
            super(DatasetGeneralLoadParams, self)._update(attrname, value)
        except Exception as err:
            raise
        else:
            dsi = DatasetInfo()
            dsiForDsglKey = dsi[self].fetch('KEY')
            dsetNewHash = self.shorthash()

            for dsiHereKey in dsiForDsglKey:
                breakpoint()
                dsiHere = dsi[dsiHereKey]
                dsetPath = dsiHere['dataset_relative_path'][0]
                # this is the start of the hash in the pathname
                dsetHashLocationStart = dsetPath.find('dataset') + len('dataset') + 1 
                dsetHashLocationEnd = dsetHashLocationStart + len(dsetNewHash) # hash is five characters long
                dsetHashOrig = dsetPath[dsetHashLocationStart:dsetHashLocationEnd] 

                dsetPathNew = dsetPath[:dsetHashLocationStart] + dsetNewHash + dsetPath[dsetHashLocationEnd:]

                dsiHere._update('dataset_relative_path', value=dsetPathNew, allowPkUpdate=True)



    def hash(self):
        assert len(self)==1, "Can currently only hash one DatasetGeneralLoadParams at a time"

        dsglParams = self.fetch(as_dict=True)[0]
        dsglParams.pop('ds_gen_params_id') # umm... the ID doesn't have to do with the parameters...
        dsglParamsHashable =  {prm : vl.item() if hasattr(vl, 'item') else vl for prm,vl in dsglParams.items()}

        dsglJson = json.dumps(dsglParamsHashable, sort_keys=True) # needed for consistency as dicts aren't ordered
        dsglHash = hashlib.md5(dsglJson.encode('ascii')).hexdigest()

        return dsglHash

    def shorthash(self):
        hsh = self.hash()

        return hsh[:5]

#    (dataset_relative_path, dataset_id, ds_gen_params_id, dataset_hash, dataset_name, processor_name, brain_area, task, date_acquired, explicit_ignore_sorts, channels_keep, trials_keep, spike_identification_method, nas_used, nas_gamma) 
@schema
class DatasetInfo(dj.Manual):
    definition = """
    # dataset
    dataset_relative_path : varchar(500) # relative path to dataset
    dataset_id : int # dataset ID I can easily refer to
    -> DatasetGeneralLoadParams
    ---
    dataset_hash : char(32) # hash of the dataset for data integrity
    dataset_name : varchar(100) # name for dataset
    processor_name : varchar(100) # person responsible for preprocessing data
    brain_area : varchar(100) # brain area
    task : varchar(100) # task the monkey was doing in this dataset
    date_acquired : date # date data was acquired
    explicit_ignore_sorts : blob # channels explicitly removed (for spike sorting goodness)
    channels_keep : blob # channels kept from overall dataset (*after* explicit_ignore_sorts are removed--useful for finding channels not removed from coincidence detection)
    trials_keep : blob # trials kept from overall dataset (useful to keep track if some trials have to be explicitly removed for whatever reason)
    spike_identification_method : enum('threshold','nas','handSort','other') # to store how waveforms are identified as spikes
    nas_used = null : varchar(100) # if nas is selected for spike id, this will hold the NAS used
    nas_gamma = null : float # if nas is selected for spike id, this is the gamma threshold on the nas_used
    """
    
    # I don't want to use the external storage paradigm they have, as I'm not
    # sure how well it would work for sql.
    #
    # As importantly, it's not as useful if everything is locally stored
    # anyway, as I don't think it actually *loads* the data into memory. Though
    # even if it does, I don't think their serialization can handle the saving
    # of a class which is what Dataset (and BinnedSpikeSet below) are
    def grabDatasets(self):
        defaultParams = loadDefaultParams()
        dataPath = Path(defaultParams['dataPath'])

        datasets = []

        # give some ordering here
        dsPaths, dsIds = self.fetch('dataset_relative_path', 'dataset_id', order_by='dataset_id')
        for path, dsId in zip(dsPaths, dsIds):
            fullPath = dataPath / path
            with fullPath.open(mode='rb') as datasetDillFh:
                dataset = pickle.load(datasetDillFh)
                dataset.id = dsId
                datasets.append(dataset)

        return datasets
    
    # like above, but requires that self only be one record, and it doesn't
    # return a list but the actual dataset
    def grabDataset(self):
        defaultParams = loadDefaultParams()
        dataPath = Path(defaultParams['dataPath'])

        if len(self) > 1:
            raise("Can only grabDataset if one record in restriction expression")
        
        path, dsId = self.fetch('dataset_relative_path', 'dataset_id', order_by='dataset_id')
        fullPath = dataPath / path[0]
        with fullPath.open(mode='rb') as datasetDillFh:
            dataset = pickle.load(datasetDillFh)
            dataset.id = dsId[0]

        return dataset

    # alright... this guy is to create the raw dataset with *nothing*
    # removed--generally not saved, but nice to be able to go back to the
    # earliest format
    #
    # NOTE: this is not a direct replacement for how DatabaseInsertion.py loads
    # up the dataset at the start... because... I didn't actually save some of
    # that metadata >.> (but also because for the path it relies on existing
    # paths in the database, syoo...)
    def createRawDataset(self, removeExplicitIgnore = True):
        if len(self)>1:
            raise Exception("Currently only taking one dataset at a time thank you")

        defaultParams = loadDefaultParams()
        dataPath = Path(defaultParams['dataPath'])

        dataMatPath = dataPath / Path(self.fetch('dataset_relative_path')[0]).parent.parent / 'processedData.mat'
        processor = self.fetch('processor_name')[0]
        selfDs = self.grabDatasets()[0]
        metastates = selfDs.metastates
        keyStates = selfDs.keyStates
        stateWithAngleName = selfDs.stateWithAngleName

        from classes.Dataset import Dataset
        datasetHere = Dataset(dataMatPath, processor, metastates = metastates, keyStates = keyStates, stateWithAngleName=stateWithAngleName)

        if removeExplicitIgnore:
            removeSort = self.fetch('explicit_ignore_sorts')
            datasetHere.removeChannelsWithSort(removeSort[0])

        return datasetHere

    def computeBinnedSpikesAroundState(self, bssProcParams, keyStateName, trialFilterLambdaDict=None, units=None):

        bssi = BinnedSpikeSetInfo()
        dsi = DatasetInfo()
        bsspp = BinnedSpikeSetProcessParams()

        bssppUse = bsspp[bssProcParams]

        # Compute the lambda info, but only add once we grab the bssppUse
        # (turns out SQLite (and maybe DataJoint)) can't filter on blobs
        if trialFilterLambdaDict is not None:
            trialFilterLambdaStrList = []
            trialFilterLambdaDescStrList = []
            for tFLDesc, tFL in trialFilterLambdaDict.items():
                trialFilterLambdaStrList.append(tFL)
                trialFilterLambdaDescStrList.append(tFLDesc)

            trialFilterLambdaStr = " ; ".join(trialFilterLambdaStrList)
            trialFilterLambdaDescStr = " ; ".join(trialFilterLambdaDescStrList)

        # NOTE this might need a potential check to compare the trialFilters,
        # since they are not directly searchable in the db... I'll cross that
        # line when I get to it
        assert len(bssppUse) <= 1, 'provided binned spike set process params are not unique'

        if len(bssppUse)>0:
            checkFilts = bssppUse.fetch('dataset_trial_filters', 'dataset_trial_filter_description', as_dict=True)[0] # can index by 0 because we know this is length one...
            if trialFilterLambdaDict is not None:
                if checkFilts['dataset_trial_filter_description'] == trialFilterLambdaDescStr and checkFilts['dataset_trial_filters'] == trialFilterLambdaStr:
                    procParamId = bssppUse.fetch1('bss_params_id')
                    # only add once you have fetched, as adding db-blobs to the
                    # dictionary ruins the fetch
                    bssProcParams.update(dict(
                        dataset_trial_filters = trialFilterLambdaStr,
                        dataset_trial_filter_description = trialFilterLambdaDescStr
                    ))

                else:
                    raise Exception("Add code to correctly insert here...")
            else:
                raise Exception("How do we check that db returns are null?")
        else:
            # Add in the lambda info if it's there...
            if trialFilterLambdaDict is not None:
                bssProcParams.update(dict(
                    dataset_trial_filters = trialFilterLambdaStr,
                    dataset_trial_filter_description = trialFilterLambdaDescStr
                ))

            # tacky, but it doesn't return the new id, syoo...
            currIds = bsspp.fetch('bss_params_id')
            bsspp.insert1(bssProcParams)
            newIds = bsspp.fetch('bss_params_id')
            procParamId = list(set(newIds) - set(currIds))
            assert len(procParamId) == 1, 'somehow more than one unique id exists >.>'
            procParamId = procParamId[0]



        trialType = bssProcParams['trial_type']
        binSizeMs = bssProcParams['bin_size']
        lenSmallestTrial = bssProcParams['len_smallest_trial']
        setStartToStateEnd = bssProcParams['start_offset_from_location'] == 'stateEnd'
        setEndToStateStart = bssProcParams['end_offset_from_location'] == 'stateStart'
        firingRateThresh = bssProcParams['firing_rate_thresh']
        returnResiduals = bssProcParams['residuals']
        startTimeOffset = bssProcParams['start_offset']
        endTimeOffset = bssProcParams['end_offset']
        fanoFactorThresh = bssProcParams['fano_factor_thresh']

        binnedSpikes = []
        bssiKeys = []

        for datasetInfo in self.fetch("KEY"):
            # First we want to check if we're going to compute new data or if it already exists...

            binnedSpikeSetDill = 'binnedSpikeSet.dill'
            # a nice way to distinguish the path for each BSS based on extraction parameters...
            bSSProcParamsJson = json.dumps(bssProcParams, sort_keys=True) # needed for consistency as dicts aren't ordered
            # encode('ascii') needed for json to be hashable...
            bSSProcParamsHash = hashlib.md5(bSSProcParamsJson.encode('ascii')).hexdigest()
            saveBSSRelativePath = Path(datasetInfo['dataset_relative_path']).parent / ('binnedSpikeSet_%s' % bSSProcParamsHash[:5]) / binnedSpikeSetDill


            saveBSSPath = dataPath / saveBSSRelativePath

            # Load data if it has been processed
            if saveBSSPath.exists():
                bssiKey = bssi[('bss_relative_path = "%s"' % str(saveBSSRelativePath))].fetch("KEY", as_dict=True)

                assert len(bssiKey) == 1, "More than one key came out for some reason even though only one should have been added"
                bssiKey = bssiKey[0]

                with saveBSSPath.open(mode='rb') as saveBSSFh:
                    binnedSpikesHere = pickle.load(saveBSSFh)

            else:
                # Here is the meat of the function, where the BinnedSpikeSets are actually computed...
                dataset = self[datasetInfo].grabDatasets()
                assert len(dataset) == 1, "More than one dataset being grabbed per key?"
                dataset = dataset[0]

                dsId = dataset.id
                stateNameStateStart = dataset.keyStates[keyStateName]

                trialsKeep = np.full(dataset.trialStatuses.shape, True)

                # Filter by trial type
                if trialType == 'successful':
                    _, trialFiltType = dataset.successfulTrials()
                elif trialType == 'failure':
                    _, trialFiltType = dataset.failTrials()
                else:
                    trialFiltType = trialsKeep

                trialsKeep &= trialFiltType

                # These extra filters are in a { description : lambda } style dictionary
                if trialFilterLambdaDict is not None:
                    for tFLDesc, tFL in trialFilterLambdaDict.items():
                        lambdaFilt = eval(tFL)
                        _, trialFiltLambda = lambdaFilt(dataset)
                        trialsKeep &= trialFiltLambda

                alignmentStates = dataset.metastates
                    
                # Filter by trial length
                startState, endState, stateNameAfter = dataset.computeStateStartAndEnd(stateName = stateNameStateStart, ignoreStates=alignmentStates)

                startStateArr = np.asarray(startState)
                endStateArr = np.asarray(endState)
                timeInStateArr = endStateArr - startStateArr

                _, trialFiltLen = dataset.filterTrials(timeInStateArr>=lenSmallestTrial)
                trialsKeep &= trialFiltLen

                dataset = dataset.filterTrials(trialsKeep)[0]

                # Find appropriate start/end times for remaining trials
                startState, endState, stateNameAfter = dataset.computeStateStartAndEnd(stateName = stateNameStateStart, ignoreStates=alignmentStates)
                startStateArr = np.asarray(startState)
                endStateArr = np.asarray(endState)

                # Assign appropriate start/end time given parameters
                if setStartToStateEnd:
                    startTimeArr = endStateArr
                else:
                    startTimeArr = startStateArr
                
                if setEndToStateStart:
                    endTimeArr = startStateArr
                else:
                    endTimeArr = endStateArr
                    
                # Bin spikes in the dataset based on state times computed above!
                # Add binSizeMs/20 to endMs to allow for that timepoint to be included when using arange
                binnedSpikesHere = dataset.binSpikeData(startMs = list(startTimeArr+startTimeOffset), endMs = list(endTimeArr+endTimeOffset+binSizeMs/20), binSizeMs=binSizeMs, alignmentPoints = list(zip(startTimeArr, endTimeArr)))

                chansKeep = np.full((binnedSpikesHere.shape[1]), True)
                # Filter to only include high firing rate channels
                _, chansHighFR = binnedSpikesHere.channelsAboveThresholdFiringRate(firingRateThresh=firingRateThresh, asLogical = True)
                chansKeep &= chansHighFR

                # Filter for fano factor threshold, but use counts *over the trial*
                if binnedSpikesHere.dtype == 'object':
                    trialLengthMs = binSizeMs*np.array([bnSpTrl[0].shape[0] for bnSpTrl in binnedSpikesHere])
                else:
                    trialLengthMs = np.array([binnedSpikesHere.shape[2]*binSizeMs])
                binnedCountsPerTrial = binnedSpikesHere.convertUnitsTo('count').increaseBinSize(trialLengthMs)

                # Under certain processing parameters, not all trials may be
                # the same length--since the fano factor computation depends on
                # counts *per trial*, having different length trials really
                # messes with things because variance in the spike count could
                # reflect variance in the trial length. So, here, I weight
                # spike counts by the trial length...
                #
                # NOTE: I believe this is okay because it's as if we were
                # taking just a similarly sized window for each trial (under
                # the assumption of uniform firing), but the std and mean is
                # taken *afterwards*
                if trialLengthMs.size > 1:
                    binnedCountsPerTrial = binnedCountsPerTrial * trialLengthMs.max()/trialLengthMs[:,None,None]
                _, chansGood = binnedCountsPerTrial.channelsBelowThresholdFanoFactor(fanoFactorThresh=fanoFactorThresh, asLogical=True)
                chansKeep &= chansGood

                binnedSpikesHere = binnedSpikesHere[:,chansKeep]

                # Previous iteration of this code accounted for binSpikeData()
                # returning a list... but *that* code has been updated to
                # return object arrays if the trial lengths are different
                binnedSpikesHere.labels['stimulusMainLabel'] = dataset.markerTargAngles

                # This might fail if trial lengths are different sizes and
                # alignment points are unequal... but we'll let it fail when it
                # fails
                if returnResiduals:
                    labels = binnedSpikesHere.labels['stimulusMainLabel']
                    binnedSpikesHere, labelBaseline = binnedSpikesHere.baselineSubtract(labels=labels)

                    
                # Save the output
                saveBSSPath.parent.mkdir(parents=True, exist_ok = True)
                with saveBSSPath.open(mode='wb') as saveBSSFh:
                    pickle.dump(binnedSpikesHere, saveBSSFh)

                binnedSpikeSetHereInfo = dict(
                    bss_params_id = procParamId,
                    bss_relative_path = str(saveBSSRelativePath),
                    start_alignment_state = stateNameStateStart,
                    end_alignment_state = stateNameStateStart, # in this function it's always from the start, either the end or the beginning of the start, but from the start
                )

                if len(bssi & binnedSpikeSetHereInfo) > 1:
                    raise Exception("we've saved this processing more than once...")
                elif len(bssi & binnedSpikeSetHereInfo) == 1:
                    breakpoint() # shouldn't really be here... means that the database entry was saved but the file had been deleted...?
                    bssiKey = bssi[('bss_relative_path = "%s"' % str(saveBSSRelativePath))].fetch("KEY", as_dict=True)
                elif len(bssi & binnedSpikeSetHereInfo) == 0:
                    bssHash = hashlib.md5(str(binnedSpikesHere).encode('ascii')).hexdigest()


                    addlBssInfo = dict(
                        bss_hash = bssHash,
                        start_time_alignment = np.array(startTimeArr),
                        start_time = np.array(startTimeArr + startTimeOffset),
                        end_time_alignment = np.array(endTimeArr),
                        end_time = np.array(endTimeArr + endTimeOffset)
                    )
                    binnedSpikeSetHereInfo.update(addlBssInfo)
                        
                    dsiPks = datasetInfo #self[datasetInfo].fetch1("KEY")
                    binnedSpikeSetHereInfo.update(dsiPks)


                    bssi.insert1(binnedSpikeSetHereInfo)

                    bssAdded = bssi[{k : v for k, v in binnedSpikeSetHereInfo.items() if k in bssi.primary_key}]
                    # add this binned spike set as a filtered spike set part as
                    # well... (obviously with no filtering, hence the 'original'
                    # reason
                    _ = bssAdded.filterBSS(filterReason = 'original', filterDescription = 'unfiltered original', condLabel = 'stimulusMainLabel', trialFilter = trialsKeep, channelFilter = chansKeep )

                    bssiKey = bssAdded.fetch("KEY", as_dict=True)

                    assert len(bssiKey) == 1, "More than one key came out for some reason even though only one should have been added"
                    bssiKey = bssiKey[0]

            if units is not None:
                binnedSpikesHere.convertUnitsTo(units=units)
            
            binnedSpikes.append(binnedSpikesHere)
            bssiKeys.append(bssiKey)

        return binnedSpikes, bssiKeys

    def _update(self, attrname, value=None, allowPkUpdate=False):
        selfKey = self.fetch('KEY')[0]
        valOld = self.fetch(attrname)[0]
        try:
            super(DatasetInfo, self)._update(attrname, value, allowPkUpdate=allowPkUpdate)
            if allowPkUpdate:
                selfKey.update({attrname : value})
        except Exception as err:
            raise
        else:
            bsi = BinnedSpikeSetInfo()
            if attrname == 'dataset_relative_path':
                dataPath = Path(defaultParams['dataPath'])
                dsetPath = Path(valOld).parent
                dsetPathNew = Path(value).parent


                # we do this outer exists() check because sometimes a parent
                # folder moved everything below it so no attempt at a move is
                # even necessary
                if not (dataPath / dsetPathNew).exists():
                    info = "Moving\n\n" + str(dataPath / dsetPath) + "\n\nto\n\n" + str(dataPath / dsetPathNew)
                    print(info)
                    from shutil import move
                    try:
                        move(dataPath / dsetPath, dataPath / dsetPathNew)
                    except FileNotFoundError as e:
                        print("Original path not found -- perhaps it has already been moved (though not to the new location, as that one didn't exist...)?")

                # now move all those downstream BinnedSpikeSetInfos...
                bsiForDsiKey = bsi[selfKey].fetch('KEY')

                for bsiKey in bsiForDsiKey:
                    bsiHere = bsi[bsiKey]
                    if len(bsiHere)<1:
                        # this might occur if an inner call already changed
                        # this bsiKey before it was reached in this outer call
                        continue
                    bsiOrigPath = bsiHere['bss_relative_path'][0]
                    if bsiOrigPath.find(str(dsetPath)) != 0:
                        breakpoint() # somethig weird here
                    bsiNewPath = dsetPathNew / bsiOrigPath[len(str(dsetPath))+1:]
                    bsiHere._update('bss_relative_path', value=str(bsiNewPath), allowPkUpdate=True)


    @staticmethod
    def rmFilesByKey(key, quickDelete=False):
        defaultParams = loadDefaultParams()
        dataPath = Path(defaultParams['dataPath'])

        relPath = Path(key['dataset_relative_path']).parent 
        fullPath = dataPath / relPath

        allFolderPaths = [str(pth) for pth in fullPath.glob('**/*')]
        if len(allFolderPaths) == 0:
            print('No dataset files to delete. Removing entry from db.')
            return True
        info = "About to delete\n\n" + "\n".join(allFolderPaths) + "\n\nand their contents."
        prompt = "Please confirm:"
        if True:#not quickDelete:
            response = userChoice(prompt, info, choices=("yes", "no"), default="no")
        if quickDelete or response=="yes":

            from shutil import rmtree
            rmtree(fullPath)

        return quickDelete or response == 'yes'
        
    @classmethod
    def rmFilesByKeyList(cls, keyList, keysDeletedList, quickDelete=False):
        for key in keyList:
            cls.rmFilesByKey(key, quickDelete=quickDelete)
            # this is meant to track the deleted keys for error handling--since
            # it's a reference, that calling function can see it, and
            # appropriately delete database entries whose keys have already
            # been used to delete files...
            keysDeletedList.append(key)

    def delete(self, quickDelete = False):
        print('zoqmh')
        dsiKeys = self.fetch('KEY') # always a dict


        for key in dsiKeys:
            if len(self[key]) == 0:
                # this condition might be reached if a parent BSS was deleted
                # before the filtered child...
                breakpoint()
                continue

            deleteDbEntry = self.rmFilesByKey(key, quickDelete=quickDelete)

            if deleteDbEntry:
                with dj.config(safemode = not quickDelete) as cfg: 
                    # this syntax creates a super instance of *just* the subset of
                    # self--so that we don't delete all of self in one go
                    # accidentally!
                    super(DatasetInfo, self[key]).delete()

    def delete_quick(self, get_count = False):
        print('qlnxo')
        dsiKeys = self.fetch('KEY') # always a dict
        try:
            delRet = super(DatasetInfo, self[dsiKeys]).delete_quick(get_count = get_count)
        except Exception as e:
            raise(e)
        else:
            try:
                keysDelList = []
                self.rmFilesByKeyList(dsiKeys, keysDelList, quickDelete=True)
            except Exception as e:
                print(e)
                pthsDeleted = [ky['dataset_relative_path'] for ky in keysDelList]
                pthsNotDelete = [ky['dataset_relative_path'] for ky in bsiKeys if ky not in keysDelList]
                print('Only managed to delete:\n' + '\n'.join(pthsDeleted))
                print('\n')
                print('NOTE: did not delete these paths:\n' + '\n'.join(pthsNotDelete) + '\nEVEN THOUGH DATABASE ENTRIES DELETED')
            finally:
                # So why is this a finally statement? Because even if we crap
                # out on deletion of files, I want to actually continue the
                # caller's calls and return a delRet to keep the database
                # consistent...
                return delRet



@schema
class BinnedSpikeSetProcessParams(dj.Manual):
    definition = """
    # dataset info extraction params
    bss_params_id : int auto_increment # params id
    ---
    key_state_start : varchar(20) # key state abstracts actual start state names each task type uses
    key_state_end : varchar(20) # key state abstracts actual end state names each task type uses
    start_offset : int # offset in ms (from start_time_alignment in the specific params)
    start_offset_from_location : enum('stateStart', 'stateEnd') # location to offset from in start_alignment_state
    end_offset : int # offset in ms (from end_time_alignment in the specific params)
    end_offset_from_location : enum('stateStart', 'stateEnd') # location to offset from in end_alignment_state
    bin_size : int # bin size in ms
    firing_rate_thresh : float # lowest firing rate threshold per channel
    fano_factor_thresh : float # highest fano factor threshold per channel
    trial_type : enum('all', 'successful', 'failure') # type of trials analyzed
    len_smallest_trial : int # length of the shortest alignment_state for trials retained
    residuals : enum(0, 1) # whether this bss is residuals (PSTH subtracted) or not
    dataset_trial_filters = null : blob # filter that removes trials from dataset before creating BinnedSpikeSet
    dataset_trial_filter_description = null : varchar(200) # reason for the filter(s)
    """

@schema
class BinnedSpikeSetInfo(dj.Manual):
    definition = """
    # binned spike set
    -> DatasetInfo
    -> BinnedSpikeSetProcessParams
    bss_relative_path: varchar(500) # path to processed binned spike set
    # these two below are pks because they tell us which chunk of time is being grabbed
    # BUT they're not in the BinnedSpikeSetProcessParams table because each dataset might
    # have a different name for these states though they represent the same thing (say, the
    # delay start and end) (poo)
    start_alignment_state : varchar(100) # name of alignment state from whose beginning start_time is offset
    end_alignment_state : varchar(100) # name of alignment state from whose beginning end_time is offset
    ---
    bss_hash : char(32) # mostly important to check data consistency if we want...
    start_time_alignment : blob # start time in ms of the alignment_state
    start_time : blob # true start_time in ms of each trial -- start_time_alignment + start_offset
    end_time_alignment : blob # end time in ms from where each bss trial end was offset
    end_time : blob # true end_time in ms of each trial -- end_time_alignment + end_offset
    """

    # for filtering spike sets...
    def filterBSS(self, filterReason, filterDescription, condLabel, trialNumPerCondMin = None, binnedSpikeSet = None, trialFilter = None, channelFilter = None, returnKey = False):
        keys = self.fetch("KEY")
        if len(keys)>1:
            raise Exception("Not sure that you want to filter multiple spike sets with the same parameters...")


        if binnedSpikeSet is None:
            # if you forgot to include the binnedSpikeSet...
            #
            # or you have voodoo magic knowing about number of
            # trials/channels... possible? Might allow you to group things if I
            # decide to allow this function to be called for groups of
            # lists of binnedSpikeSets
            bss = self.grabBinnedSpikes()
        else:
            bss = binnedSpikeSet

        if len(bss)>1:
            raise Exception("filterBSS should only be called on one binned spike set...")
        elif len(bss) == 0:
            raise Exception("filterBSS didn't find any binned spike sets")
        else:
            bssFiltered = bss[0]
            key = keys[0]

            bssFilterParams = dict(filter_reason = filterReason, filter_description = filterDescription)

            if trialFilter is not None:
                # for the original filter, trialFilter actually refers to how
                # the raw dataset was filtered before being saved as a
                # BinnedSpikeSet
                if filterReason != 'original':
                    bssFiltered = bssFiltered[trialFilter]
                bssFilterParams.update(dict(trial_filter = trialFilter))

            if channelFilter is not None:
                # for the original filter, channelFilter actually refers to how
                # the raw dataset was filtered before being saved as a
                # BinnedSpikeSet
                if filterReason != 'original':
                    bssFiltered = bssFiltered[:, channelFilter]
                bssFilterParams.update(dict(ch_filter = channelFilter))

            # we run this *after* the filtering has happened...
            _, condCnts = np.unique(bssFiltered.labels[condLabel], axis=0, return_counts=True)
            if trialNumPerCondMin is not None and np.min(condCnts) != trialNumPerCondMin:
                # this would be a weird occurrence... for some reason a minimum
                # is given but it's the incorrect minimum...
                breakpoint()
            trialNumPerCondMin = np.min(condCnts)

            bssFilterParams.update(dict(
                    trial_num_per_cond_min = trialNumPerCondMin,
                    condition_label = condLabel,
                    trial_num = bssFiltered.shape[0],
                    ch_num = bssFiltered.shape[1]
                ))


            defaultParams = loadDefaultParams()
            dataPath = Path(defaultParams['dataPath'])

            filteredSpikeSetDill = 'filteredSpikeSet.dill'

            # a nice way to distinguish the path for each filter based on extraction parameters...
            bSSFilteredProcParamsJson = json.dumps(str(bssFilterParams), sort_keys=True) # needed for consistency as dicts aren't ordered
            bSSFilteredProcParamsHash = hashlib.md5(bSSFilteredProcParamsJson.encode('ascii')).hexdigest()

            # NOTE this is for the past's sake that I'm not changing
            # bssFilterParams before it's hashed--I don't want to go through
            # and change all the paths because the hash has changed ;_;
            bssFilterParams.update(dict(
                condition_num = condCnts.size # the size tells us how *many* unique conditions there were
            ))

            fsp = FilterSpikeSetParams()
            bsi = BinnedSpikeSetInfo()
            parentPath = fsp[key]['bss_relative_path']
            assert len(parentPath)<=1, 'Too many filter params for one bin spike set'
            if len(parentPath) == 1:
                parentPath = parentPath[0]
                parFsp = fsp[bsi[('bss_relative_path="%s"' % parentPath)]]
                parFspHash = parFsp['fss_param_hash'][0]
                bssPathHash = ('%s_%s' % (parFspHash[:5], bSSFilteredProcParamsHash[:5]))
                origFilter = False
            else:
                if filterReason != "original":
                    breakpoint() # mostly because I'm not sure why we'd be here... if it wasn't the original one
                parentPath = key['bss_relative_path']
                origFilter = True
                bssPathHash = bSSFilteredProcParamsHash[:5]

            if filterReason != "original":
                pathRelativeToParent = Path('filteredSpikes_%s' % bssPathHash) / filteredSpikeSetDill
            else:
                pathRelativeToParent = Path(key['bss_relative_path']).name # now it's just the parent's dill file

            bssFilterParams.update(dict(
                    fss_param_hash = bSSFilteredProcParamsHash,
                ))

            if not origFilter:
                # note that this leaves it as null otherwise
                bssFilterParams.update(dict(
                        parent_bss_relative_path = parentPath,
                    ))


            pathRelativeToBase = Path(parentPath).parent / pathRelativeToParent
            fullPath = dataPath / pathRelativeToBase
            
            # I could have checked this with the fss_param_hash if statement
            # above, but I wanted to be more explicit here
            existingSS = fsp[{k : v for k,v in bssFilterParams.items() if k in fsp.primary_key + ['parent_bss_relative_path']}] 
            existingSS = existingSS[self.fetch('bss_params_id', as_dict=True)[0]]

            # this is important for not grabbing repeats when looking for the
            # originally filtered binned spike set
            if 'parent_bss_relative_path' not in bssFilterParams:
                existingSS = existingSS['parent_bss_relative_path is NULL']

            if len(existingSS) > 1:
                raise Exception('Multiple filtered spike sessions have been saved with these parameters')
            elif len(existingSS) > 0:
                prmPk = bsi[('bss_relative_path = "%s"' % str(pathRelativeToBase))].fetch("KEY",as_dict=True)[0]
                if not fullPath.exists():
                    print('Db record existed for FilterSpikeSetParams but not file... saving file now')
                    fullPath.parent.mkdir(parents=True, exist_ok = True)
                    with fullPath.open(mode='wb') as filteredSpikeSetDillFh:
                        pickle.dump(bssFiltered, filteredSpikeSetDillFh)
            else:
                # this check is really here in case we're dropping in the
                # 'original' FSS, whose path is the parent, so there's not need
                # to resave
                if not fullPath.exists():
                    fullPath.parent.mkdir(parents=True, exist_ok = True)
                    with fullPath.open(mode='wb') as filteredSpikeSetDillFh:
                        pickle.dump(bssFiltered, filteredSpikeSetDillFh)

                bssFilteredHash = hashlib.md5(str(bssFiltered).encode('ascii')).hexdigest()


                if not origFilter:
                    paramsNew = self.fetch(as_dict=True)[0]
                    paramsNew['bss_relative_path'] = str(pathRelativeToBase)
                    paramsNew['bss_hash'] = bssFilteredHash

                    # take care of filtering the start and end times saved with
                    # this BSS here
                    if trialFilter is not None:
                        paramsNew['start_time'] = paramsNew['start_time'][trialFilter]
                        paramsNew['start_time_alignment'] = paramsNew['start_time_alignment'][trialFilter]
                        paramsNew['end_time'] = paramsNew['end_time'][trialFilter]
                        paramsNew['end_time_alignment'] = paramsNew['end_time_alignment'][trialFilter]
                    print('inserting bsi')
                    bsi.insert1(dict(
                            **paramsNew
                    ))
                    prmPk = {k : v for k,v in paramsNew.items() if k in self.primary_key}
                    print('inserting fsp')
                    fsp.insert1(dict(
                        **prmPk,
                        **bssFilterParams
                    ))
                    print('inserted')
                else:
                    fsp.insert1(dict(
                        **key,
                        **bssFilterParams
                    ))

            if returnKey:
                bssKey = dict(**prmPk)
                return bssFiltered, bssKey
            return bssFiltered

    def computeRandomSubsets(self, filterDescription, numTrialsPerCond, numChannels, labelName, binnedSpikeSet = None, numSubsets = 1, returnInfo = False):
        filterReason = 'randomSubset' # that's the purpose of this function...
        if binnedSpikeSet is None:
            # if you forgot to include the binnedSpikeSet...
            #
            # or you have voodoo magic knowing about number of
            # trials/channels... possible? Might allow you to group things if I
            # decide to allow this function to be called for groups of
            # lists of binnedSpikeSets
            if returnInfo:
                bss, bssInfo = self.grabBinnedSpikes(returnInfo=returnInfo)
            else:
                bss = self.grabBinnedSpikes(returnInfo=returnInfo)
        else:
            bss = binnedSpikeSet

        if len(self)>1:
            raise Exception("computeRandomSubset should only be called on one binned spike set...")
        elif len(self) == 0:
            raise Exception("computeRandomSubset didn't find any binned spike sets")
        else:

            bssFilterParams = dict(
                    filter_reason = filterReason,
                    )

            if numTrialsPerCond is not None:
                useAllTrials = False
                bssFilterParams['trial_num_per_cond_min'] = numTrialsPerCond
            elif numTrialsPerCond == "all":
                useAllTrials = True
                _, condCounts = np.unique(bss.labels[labelName], return_counts=True, axis=0)
                numTrialsPerCond = np.min(condCounts)
                bssFilterParams['trial_num_per_cond_min'] = numTrialsPerCond

            if numChannels is not None:
                bssFilterParams['ch_num'] = numChannels
            else:
                numChannels = bss.shape[1]
                bssFilterParams['ch_num'] = numChannels


            fsp = FilterSpikeSetParams()
            bsi = BinnedSpikeSetInfo()
            existingSS = fsp[bssFilterParams][{'parent_bss_relative_path' : self.fetch1('bss_relative_path')}]


            if list(existingSS['filter_description']).count(filterDescription) != len(existingSS['filter_description']):
                # NOTE: maybe we can just filter on the filterDescription. I'm
                # not sure that's what I want, though, so let's hold off for
                # now...
                raise Exception("Filter descriptions are different... come back and fix this so it doesn't break!")

            trChFiltsPth = existingSS.fetch('trial_filter','ch_filter', 'bss_relative_path', order_by='fss_param_hash', as_dict=True)
            # from a list of trial and a list of ch filters, tuples of (tr,ch) filters
            trChFilts = trChFiltsPth[:2] # first two values
            trlChanFilters = [(tr['trial_filter'], tr['ch_filter']) for tr in trChFiltsPth]
            trlChanFilters = trlChanFilters[:numSubsets]
            fssKeys =  [{'bss_relative_path' : tr['bss_relative_path']} for tr in trChFiltsPth]
            # NOTE might be worth being able to specify how many subsets to
            # grab in grabBinnedSpikes so we don't have too much
            # overhead... for later
#            randomSubsets = bsi[fssKeys].grabBinnedSpikes(orderBy='bss_hash')[:numSubsets]
            # NOTE: I think usign fssKeys was from a prior iteration fo
            # FilteredSpikeSetParams that was a child of BSI--now I believe I
            # can directly filter with the existingSS value
            randomSubsets = bsi[existingSS].grabBinnedSpikes(orderBy='bss_hash')[:numSubsets]

            for bs in existingSS.fetch('KEY'):
                bssFiltered = bsi[bs].grabBinnedSpikes()[0]
                if bsi[bs]['start_time'][0].shape[0] != bssFiltered.shape[0]:
                    # NOTE: this'll be painful, but we need the filtered times
                    # to be the correct length so we're doing it here
                    trialFilter = fsp[bsi[bs]]['trial_filter'][0]
                    startTimeOld = bsi[bs]['start_time'][0]
                    startTimeAlignOld = bsi[bs]['start_time_alignment'][0]
                    endTimeOld = bsi[bs]['end_time'][0]
                    endTimeAlignOld = bsi[bs]['end_time_alignment'][0]
                    breakpoint()
                    bsi[bs]._update('start_time', value=startTimeOld[trialFilter])
                    bsi[bs]._update('start_time_alignment', value=startTimeAlignOld[trialFilter])
                    bsi[bs]._update('end_time', value=endTimeOld[trialFilter])
                    bsi[bs]._update('end_time_alignment', value=endTimeAlignOld[trialFilter])

            if len(randomSubsets) < numSubsets:

                bnSpOrig = bss[0]

                chanInds = range(bnSpOrig.shape[1])

                while len(randomSubsets) < numSubsets:
                    # note that the trials are randomly chosen, but returned in sorted
                    # order (balancedTriaInds does this) from first to last... I think
                    # this'll make things easier to compare in downstream analyses that
                    # care about changes over time... neurons being sorted on the other
                    # hand... still for ease of comparison, maybe not for any specific
                    # analysis I can think of
                    if useAllTrials:
                        trlsUse = np.arange(bnSpOrig.shape[0])
                    else:
                        try:
                            trlsUse = bnSpOrig.balancedTrialInds(bnSpOrig.labels[labelName], minCnt = numTrialsPerCond)
                        except Exception as e:
                            if str(e) == "One of your labels doesn't have the minimum number of samples":
                                keepSubsamp = False
                            else:
                                raise e
                        else:
                            keepSubsamp = True

                    chansUse = np.sort(np.random.permutation(chanInds)[:numChannels])


                    if keepSubsamp:
                        randomSubset, fssKey = self.filterBSS(filterReason, filterDescription,trialNumPerCondMin = numTrialsPerCond, condLabel = labelName, binnedSpikeSet = [bnSpOrig], trialFilter = trlsUse, channelFilter = chansUse, returnKey=returnInfo)
                    else:
                        randomSubset = None
                        fssKey = 'bss_relative_path IS NULL'
                        trlsUse = None
                        chansUse = None

                    if randomSubset not in randomSubsets:
                        randomSubsets += [randomSubset]
                        trlChanFilters.append((trlsUse, chansUse))
                        fssKeys.append(fssKey)


            if returnInfo:
                return randomSubsets, trlChanFilters, bssInfo['dataset_names'][0], fssKeys
            return randomSubsets

    def grabBinnedSpikes(self, returnInfo = False, orderBy = None):
        defaultParams = loadDefaultParams()
        dataPath = Path(defaultParams['dataPath'])

        binnedSpikeSets = []

        # give some ordering here
        # putting this in a list for some fancy shit below with zip to work if there is no fss_rel_path_from_parent
        dsi = DatasetInfo()
        fsp = FilterSpikeSetParams()
        bsi = BinnedSpikeSetInfo()
        relPaths = []
        dsNames = []
        bssPaths = (self * dsi).fetch('bss_relative_path', 'dataset_name', order_by=orderBy, as_dict=True)
        try:
            fssPaths = self.fetch('fss_rel_path_from_parent')
        except:
            fssPaths = None
        else:
            bssPaths = (self * dsi).fetch('bss_relative_path', 'fss_rel_path_from_parent', 'dataset_name', order_by=orderBy, as_dict=True)
        for path in bssPaths:
            if 'fss_rel_path_from_parent' in path:
                relPath = Path(path['bss_relative_path']).parent / path['fss_rel_path_from_parent']
#                print(relPath)
            else:
                relPath = path['bss_relative_path']
            relPaths.append(relPath)
            dsNames.append(path['dataset_name'])
            fullPath = dataPath / relPath
            try:
                with fullPath.open(mode='rb') as binnedSpikeSetDillFh:
                    bss = pickle.load(binnedSpikeSetDillFh)
                    binnedSpikeSets.append(bss)
            except FileNotFoundError:
                if fsp[self]['reason'] == 'original':
                    raise Exception('consider coming back here to force a recomputation of the binned spike sets...')
                else:
                    filterParams = fsp[path]
                    assert len(filterParams)==1, "More than one filter parameter here..."
                    # probably a sql or sqlite restriction, but we can't
                    # project to a key without reassigning that key, which is
                    # what temp is doing here
                    bsiParent = bsi[filterParams.proj(bss_relative_path='parent_bss_relative_path', temp='bss_relative_path')]
                    bssParent = bsiParent.grabBinnedSpikes()
                    assert len(bssParent)==1, "Not sure why, but more than one parent found...?"
                    bssParent = bssParent[0]
                    trialFilt = filterParams['trial_filter'][0]
                    chFilt = filterParams['ch_filter'][0]
                    bssChild = bssParent[trialFilt][:,chFilt]

                    fullPath.parent.mkdir(parents=True) # I want to let it error if the directory exists for now... not sure why that would happen...
                    with fullPath.open(mode='wb') as binnedSpikeSetDillFh:
                        pickle.dump(bssChild, binnedSpikeSetDillFh)

                    binnedSpikeSets.append(bssChild)


        if returnInfo:
            return binnedSpikeSets, dict(paths = relPaths, dataset_names=dsNames)
        return binnedSpikeSets

    def grabFilteredDataset(self):
        if len(self)>1:
            raise Exception('Can only grab the filtered dataset of one bss at a time')

        bsp = BinnedSpikeSetProcessParams()
        dsi = DatasetInfo()

        ds = dsi[self].grabDatasets()[0]
        bsParamsDsFilts = bsp[self]['dataset_trial_filters'][0]

        # filter trials by special filters
        bsPDsFiltsInd = bsParamsDsFilts.split(';')
        for filtStr in bsPDsFiltsInd:
            lambdaFilt = eval(filtStr)
            ds, _ = lambdaFilt(ds)

        # filter trials by type
        trialType = bsp[self]['trial_type'][0]
        if trialType == 'successful':
            ds, _ = ds.successfulTrials()
        elif trialType == 'failure':
            ds, _ = ds.failTrials()

        # filter trials by length in state
        # NOTE this is special to expecting trials to be around a state...
        stateNameStateStart = self['start_alignment_state'][0]
        alignmentStates = ds.metastates
        startState, endState, stateNameAfter = ds.computeStateStartAndEnd(stateName = stateNameStateStart, ignoreStates=alignmentStates)
        startStateArr = np.asarray(startState)
        endStateArr = np.asarray(endState)
        timeInStateArr = endStateArr - startStateArr

        lenSmallestTrial = bsp[self].fetch('len_smallest_trial')
        ds, _ = ds.filterTrials(timeInStateArr>=lenSmallestTrial)

        return ds


    def rebinSpikes(self, newBinSizeMs):
        dsi = DatasetInfo()
        bsp = BinnedSpikeSetProcessParams()
        ds = self.grabFilteredDataset()
#        ds = dsi[self].grabDatasets()[0]
#        bsParamsDsFilts = bsp[self]['dataset_trial_filters'][0]
#
#        # filter trials by special filters
#        bsPDsFiltsInd = bsParamsDsFilts.split(';')
#        for filtStr in bsPDsFiltsInd:
#            lambdaFilt = eval(filtStr)
#            ds, _ = lambdaFilt(ds)
#
#        # filter trials by type
#        trialType = bsp[self]['trial_type'][0]
#        if trialType == 'successful':
#            ds, _ = ds.successfulTrials()
#        elif trialType == 'failure':
#            ds, _ = ds.failTrials()
#
#        # filter trials by length in state
#        # NOTE this is special to expecting trials to be around a state...
#        stateNameStateStart = self['start_alignment_state'][0]
#        alignmentStates = ds.metastates
#        startState, endState, stateNameAfter = ds.computeStateStartAndEnd(stateName = stateNameStateStart, ignoreStates=alignmentStates)
#        startStateArr = np.asarray(startState)
#        endStateArr = np.asarray(endState)
#        timeInStateArr = endStateArr - startStateArr
#
#        lenSmallestTrial = bsp[self].fetch('len_smallest_trial')
#        ds, _ = ds.filterTrials(timeInStateArr>=lenSmallestTrial)
#
        bnSpTmBounds = self.fetch('start_time','end_time',as_dict=True)[0]
        bnSpTmAlign = self.fetch('start_time_alignment', 'end_time_alignment')
        alBins = list(zip(bnSpTmAlign[0][0], bnSpTmAlign[1][0]))

        reBinnedSpikeSet = ds.binSpikeData(startMs = list(bnSpTmBounds['start_time']), endMs = list(bnSpTmBounds['end_time']+newBinSizeMs/20), binSizeMs=newBinSizeMs, alignmentPoints = alBins)
        reBinnedSpikeSet.labels['stimulusMainLabel'] = ds.markerTargAngles

        return reBinnedSpikeSet

    def grabAlignedKinematics(self, kinBinning = None):
#    start_alignment_state : varchar(100) # name of alignment state from whose beginning start_time is offset
#    end_alignment_state : varchar(100) # name of alignment state from whose beginning end_time is offset
#    ---
#    bss_hash : char(32) # mostly important to check data consistency if we want...
#    start_time_alignment : blob # start time in ms of the alignment_state
#    start_time : blob # true start_time in ms of each trial -- start_time_alignment + start_offset
#    end_time_alignment : blob # end time in ms from where each bss trial end was offset
#    end_time : blob # true end_time in ms of each trial -- end_time_alignment + end_offset
        def forwardFillNans(arr, arrPad=None):
            if arrPad is None:
                arrPad = arr
            valCurr = arrPad[0]
            for ind in range(len(arr)):
                if np.isnan(arr[ind]):
                    arr[ind] = valCurr
                else:
                    valCurr = arrPad[ind]
            return arr

        dsi = DatasetInfo()
        bsp = BinnedSpikeSetProcessParams()
        binnedSpikes, bSInfo = self.grabBinnedSpikes(returnInfo=True)

        for bnSp, bSPath in zip(binnedSpikes, bSInfo['paths']):
            bsPk = dict(bss_relative_path = bSPath)

            bss = self[bsPk].grabBinnedSpikes()[0]
                
            bssStartTimesInTrial = self[bsPk]['start_time'][0]
            bssEndTimesInTrial = self[bsPk]['end_time'][0]

            ds = self[bsPk].grabFilteredDataset()
            trialKinematics = ds.kinematics
            kinsForBss = [kin[(kin[:, 2]>stTr) & (kin[:,2]<endTr), :] for kin, stTr, endTr in zip(trialKinematics, bssStartTimesInTrial, bssEndTimesInTrial)]

            if kinBinning == 'binToMatch':
                binSize = bss.binSize
                from scipy.stats import binned_statistic

                # adding binSize/20 reflects how spikes are binned with the end
                # point, to allow its inclusion...
                lastValPerBin = np.array([binned_statistic(kin[:,2], kin[:,:2].T, statistic=lambda x:x[-1], bins=np.arange(st, et+binSize/20, binSize))[0].T if kin.size else np.array([]) for kin, st, et in zip(kinsForBss, bssStartTimesInTrial, bssEndTimesInTrial)])
                kinsForBss = np.array([binned_statistic(kin[:,2], kin[:,:2].T, statistic='mean', bins=np.arange(st, et+binSize/20, binSize))[0].T if kin.size else np.array([]) for kin, st, et in zip(kinsForBss, bssStartTimesInTrial, bssEndTimesInTrial)])
                # note that forwardFillNans works in place syoo... it's
                # changing the values in place... heh; no need to assign an
                # output
                #
                # anyway... this part replaces points where the cursor didn't
                # move (and thus there were no codes and its nan here) into the
                # actual location...which was its previous location!
                [[forwardFillNans(kin[:,0],lstValKin[:,0]), forwardFillNans(kin[:,1],lstValKin[:,1])] for kin,lstValKin in zip(kinsForBss, lastValPerBin)]



        return kinsForBss

    def generateDescriptivePlots(self, plotTypes = ['all'], chPlot = None):
        from methods.plotUtils.BinnedSpikeSetPlotMethods import plotResponseOverTime
        binnedSpikes, bSInfo = self.grabBinnedSpikes(returnInfo=True)


        if 'raster' in plotTypes or 'all' in plotTypes:
            rasterPlotInfo = plotTypes['raster'] if 'raster' in plotTypes else {}
            rasterPlotInfo['plotMethod'] = 'eventplot'
            reBinnedSpikes = []
            for bnSp, bSPath in zip(binnedSpikes, bSInfo['paths']):
                bsPk = dict(bss_relative_path = bSPath)
                    
                binSizeMs = 1 # we are now binning at 1ms for this...
                reBinnedSpikeSet = self[bsPk].rebinSpikes(binSizeMs)
                reBinnedSpikes.append(reBinnedSpikeSet.convertUnitsTo('count'))

            plotResponseOverTime(reBinnedSpikes, bSInfo['dataset_names'], rasterPlotInfo, chPlot = chPlot)


        if 'psth' in plotTypes or 'all' in plotTypes:
            psthPlotInfo = plotTypes['psth'] if 'psth' in plotTypes else {}
            psthPlotInfo['plotMethod'] = 'plot'
            plotResponseOverTime(binnedSpikes, bSInfo['dataset_names'], psthPlotInfo, chPlot = chPlot)

        #%% PCA projections
        if 'pca' in plotTypes or 'all' in plotTypes:
            for idx, (bnSp, dsName) in enumerate(zip(binnedSpikes, bSInfo['dataset_names'])):
                bnSp.pca(labels = bnSp.labels['stimulusMainLabel'], plot = True)
                plt.suptitle('PCA - %s' % (dsName))
            
        #%% LDA projections
        if 'lda' in plotTypes or 'all' in plotTypes:
            for idx, (bnSp, dsName) in enumerate(zip(binnedSpikes, bSInfo['dataset_names'])):
                bnSp.lda(bnSp.labels['stimulusMainLabel'], plot=True)
                plt.suptitle('LDA - %s' % (dsName))

        #%% decodability
        if ('decode' in plotTypes and plotTypes['decode']) or 'all' in plotTypes:
            for idx, (bnSp, dsName) in enumerate(zip(binnedSpikes, bSInfo['dataset_names'])):
                bnSp.decode(bnSp.labels['stimulusMainLabel'], plot=True)
                plt.suptitle('Decode - %s' % (dsName))


    def _update(self, attrname, value=None, allowPkUpdate=False):
        valOld = self.fetch(attrname)[0]
        try:
            super(BinnedSpikeSetInfo, self)._update(attrname, value, allowPkUpdate=allowPkUpdate)
        except Exception as err:
            raise
        else:
            bsi = BinnedSpikeSetInfo()
            if attrname == 'bss_relative_path':
                dataPath = Path(defaultParams['dataPath'])
                bssPathOrig = Path(valOld).parent
                bssPathNew = Path(value).parent

                # we do this outer exists() check because sometimes a parent
                # folder moved everything below it so no attempt at a move is
                # even necessary
                if not (dataPath / bssPathNew).exists():
                    info = "Moving\n\n" + str(dataPath / bssPathOrig) + "\n\nto\n\n" + str(dataPath / bssPathNew)
                    print(info)
                    from shutil import move
                    try:
                        move(dataPath / bssPathOrig, dataPath / bssPathNew)
                    except FileNotFoundError as e:
                        print("Original path not found -- perhaps it has already been moved (though not to the new location, as that one didn't exist...)?")

                # now move all those downstream BinnedSpikeSetInfos...
                bsiChildBsiKey = bsi['bss_relative_path LIKE "%{}%"'.format(bssPathOrig)].fetch('KEY')

                for bsiKey in bsiChildBsiKey:
                    bsiHere = bsi[bsiKey]
                    if len(bsiHere)<1:
                        # this might occur if an inner call already changed
                        # this bsiKey before it was reached in this outer call
                        continue
                    bsiOrigPath = bsiHere['bss_relative_path'][0]
                    if bsiOrigPath.find(str(bssPathOrig)) != 0:
                        breakpoint() # somethig weird here
                    bsiNewPath = bssPathNew / bsiOrigPath[len(str(bssPathOrig))+1:]
                    bsiHere._update('bss_relative_path', value=str(bsiNewPath), allowPkUpdate=True)

    def trialAndChannelFilterFromParent(self, parent):
        fsp = FilterSpikeSetParams()
        trlFilter, chFilter = fsp[self].trialAndChannelFilterFromParent(parent)

        return trlFilter, chFilter


    @staticmethod
    def rmFilesByKey(key, quickDelete=False):
        defaultParams = loadDefaultParams()
        dataPath = Path(defaultParams['dataPath'])

        relPath = Path(key['bss_relative_path']).parent 
        fullPath = dataPath / relPath

        allFolderPaths = [str(pth) for pth in fullPath.glob('**/*')]
        if len(allFolderPaths) == 0:
            print('No binned spike set files to delete. Removing entry from db.')
            return True
        info = "About to delete\n\n" + "\n".join(allFolderPaths) + "\n\nand their contents."
        prompt = "Please confirm:"
        if not quickDelete:
            response = userChoice(prompt, info, choices=("yes", "no"), default="no")
        if quickDelete or response=="yes":

            from shutil import rmtree
            rmtree(fullPath)

        return quickDelete or response == 'yes'
        
    @classmethod
    def rmFilesByKeyList(cls, keyList, keysDeletedList, quickDelete=False):
        for key in keyList:
            cls.rmFilesByKey(key, quickDelete=quickDelete)
            # this is meant to track the deleted keys for error handling--since
            # it's a reference, that calling function can see it, and
            # appropriately delete database entries whose keys have already
            # been used to delete files...
            keysDeletedList.append(key)

    def delete(self, quickDelete = False):
        print('lskdj')
        bsiKeys = self.fetch('KEY') # always a dict


        for key in bsiKeys:
            if len(self[key]) == 0:
                # this condition might be reached if a parent BSS was deleted
                # before the filtered child...
                breakpoint()
                continue

            deleteDbEntry = self.rmFilesByKey(key, quickDelete=quickDelete)

            if deleteDbEntry:
                with dj.config(safemode = not quickDelete) as cfg: 
                    # this syntax creates a super instance of *just* the subset of
                    # self--so that we don't delete all of self in one go
                    # accidentally!
                    super(BinnedSpikeSetInfo, self[key]).delete()

    def delete_quick(self, get_count = False):
        print('llkww')
        bsiKeys = self.fetch('KEY') # always a dict
        try:
            delRet = super(BinnedSpikeSetInfo, self[bsiKeys]).delete_quick(get_count = get_count)
        except Exception as e:
            raise(e)
        else:
            try:
                keysDelList = []
                self.rmFilesByKeyList(bsiKeys, keysDelList, quickDelete=True)
            except Exception as e:
                print(e)
                pthsDeleted = [ky['bss_relative_path'] for ky in keysDelList]
                pthsNotDelete = [ky['bss_relative_path'] for ky in bsiKeys if ky not in keysDelList]
                print('Only managed to delete:\n' + '\n'.join(pthsDeleted))
                print('\n')
                print('NOTE: did not delete these paths:\n' + '\n'.join(pthsNotDelete) + '\nEVEN THOUGH DATABASE ENTRIES DELETED')
            finally:
                # So why is this a finally statement? Because even if we crap
                # out on deletion of files, I want to actually continue the
                # caller's calls and return a delRet to keep the database
                # consistent...
                return delRet


@schema
class FilterSpikeSetParams(dj.Manual):
    definition = """
    # for filtered spike sets
    -> BinnedSpikeSetInfo
    fss_param_hash : char(32) # for looking this entry up
    ---
    (parent_bss_relative_path) -> [nullable] BinnedSpikeSetInfo(bss_relative_path)
    trial_filter = null : blob # filter of the trial on the binned spike set info
    trial_num : int # number of trials; useful for filtering shuffles
    trial_num_per_cond_min = null : int # minimum number of trials per condition; useful for filtering shuffles
    condition_label : varchar(50) # label used to identify conditions for trial_num_per_cond_min
    ch_filter = null : blob # filter of the channels on the binned spike set info
    ch_num : int # number of channels; useful for filtering shuffles
    condition_num : int # number of conditions; useful for when combining conditions and matching trials
    filter_reason : enum('original', 'shuffle', 'randomSubset', 'other') # the 'original' option is for the default FSS that refers to its parent
    filter_description = null : varchar(100) # reason for filter (if not shuffle, usually)
    """
    def trialAndChannelFilterFromParent(self, parent):
        fsp = FilterSpikeSetParams()
        bsi = BinnedSpikeSetInfo()
        if type(parent) == BinnedSpikeSetInfo:
            parent = fsp[parent]

        inverseOrderTrialFilter = [self['trial_filter'][0]]
        inverseOrderChannelFilter = [self['ch_filter'][0]]
        directParent = fsp['bss_relative_path="{}"'.format(self['parent_bss_relative_path'][0])]
        while (len(directParent) > 0) and (directParent['bss_relative_path'][0] != parent['bss_relative_path']):
            parTrlFilt = directParent['trial_filter'][0]
            parChFilt = directParent['ch_filter'][0]

            if parTrlFilt is None:
                parTrlFilt = np.arange(directParent['trial_num'])
            if parChFilt is None:
                parChFilt = np.arange(directParent['ch_num'])

            inverseOrderTrialFilter.append(parTrlFilt)
            inverseOrderChannelFilter.append(parChFilt)

            directParent = fsp['bss_relative_path="{}"'.format(directParent['parent_bss_relative_path'][0])]

        trlFiltOverall = None
        chFiltOverall = None
        for trlFilt, chFilt in zip(inverseOrderTrialFilter[::-1], inverseOrderChannelFilter[::-1]):
            if trlFilt.dtype == 'bool':
                trlFilt = trlFilt.nonzero()[0]
            if chFilt.dtype == 'bool':
                chFilt = chFilt.nonzero()[0]

            if trlFiltOverall is None:
                trlFiltOverall = trlFilt
            else:
                trlFiltOverall = trlFiltOverall[trlFilt]

            if chFiltOverall is None:
                chFiltOverall = chFilt
            else:
                chFiltOverall = chFiltOverall[chFilt]

        return trlFiltOverall, chFiltOverall



    # if a parent BSS induces an entry in this table to be deleted, then all
    # the children should be deleted as well...
    def delete(self, quickDelete = False):
        # NOTE if an FSS gets deleted because its parent got deleted, we need
        # to make sure to delete the child! On the other hand, if its deleted
        # because its reference child is deleted, we can just delete the child.
        # So here I'm basically gonna say that if you're deleting an FSS...
        # delete the child as well (that'll cascade down to deleting the FSS of
        # any grandchildren and so on, so we'd be good there...
        print('aawiq')
        fspKeys = self.fetch("KEY")
        bsi = BinnedSpikeSetInfo()

        # Get rid of any BinnedSpikeSetInfo who are the child of this
        # one--gotta grab the info before this FilteredSpikeSetParams is
        # deleted!
        fsp = FilterSpikeSetParams()
        fspProjBsiPathToParent = self.proj(parent_bss_relative_path = 'bss_relative_path')
        parentBsiPaths = fspProjBsiPathToParent.fetch('parent_bss_relative_path', as_dict=True)
        breakpoint()


        for key in fspKeys:
            bsi[self[key]].delete()
            with dj.config(safemode = not quickDelete) as cfg: 
                super(FilterSpikeSetParams, self[key]).delete()

        # Grab and delete children of this BSI/FSP combo
        fspWithParentlessBsi = fsp[parentBsiPaths]
        childBsiDel = bsi[fspWithParentlessBsi].fetch('KEY', as_dict=True)

        bsi[childBsiDel].delete(quickDelete=quickDelete)

    def delete_quick(self, get_count=False):
        print('blamslk')
        fspKeys = self.fetch("KEY")
        bsi = BinnedSpikeSetInfo()

        fsp = FilterSpikeSetParams()
        fspProjBsiPathToParent = self.proj(parent_bss_relative_path = 'bss_relative_path')
        parentBsiPaths = fspProjBsiPathToParent.fetch('parent_bss_relative_path', as_dict=True)
        fspWithParentlessBsi = fsp[parentBsiPaths]
        childBsiDel = bsi[fspWithParentlessBsi].fetch('KEY', as_dict=True)
        bsiDel = bsi[self].fetch('KEY', as_dict=True)

        dbDelete = super(FilterSpikeSetParams, self[fspKeys]).delete_quick(get_count = get_count)
        # can't be a delete_quick because this has to percolate to all of *its*
        # connections... (and the QueryExpression quick_delete does not delete
        # dependencies)... but at least for the purposes of my code I can have
        # it be a quickDelete
        # Moreover NOTE: we delete these to get rid of paths *after* we delete
        # the fsp, to avoid a recursive loop of deeeath
        bsi[bsiDel].delete(quickDelete=True)
        bsi[childBsiDel].delete(quickDelete=True)

        return dbDelete

@schema
class GpfaAnalysisParams(dj.Lookup):
    definition = """
    # gpfa analyses parameters
    gpfa_params_id : int auto_increment # gpfa params id
    ---
    method_used='gpfa' : enum('gpfa', 'fa') # this allows database storage of FA results
    dimensionality : int # extraction dimensionality
    overall_fr_thresh : float # the firing rate thresh for all the spikes before splitting by condition, where -1 lets all through (this would need to happen if residuals are being input -- though note that residuals could be calculated in function after this threshold, so this would not be -1)
    balance_conds : enum(0, 1) # whether to have equal trials from each condition
    sqrt_spikes : enum(0, 1) # whether to square root spike counts
    num_conds : int # number of conditions used, only matters if combine_conditions is 'subset' - if combine_conditions is 'all', num_conds is 0
    combine_conditions : enum('no', 'subset', 'all') # whether multiple conditions are combined
    num_folds_crossvalidate : int # folds of crossvalidation
    on_residuals : enum(0, 1) # flag for whether input was baseline subtraced
    units : varchar(10) # I'm guessing this'll be Hz or count... but I guess it could be something else?
    """
    def hash(self):
        assert len(self)==1, "Can't hash more than one entry at once"

        params = self.fetch(as_dict=True)[0]
        params.pop('gpfa_params_id')
        paramsJson = json.dumps(params, sort_keys=True)

        return hashlib.md5(paramsJson.encode('ascii')).hexdigest()


@schema
class GpfaAnalysisInfo(dj.Manual):
    definition = """
    # gpfa analyses info
    -> BinnedSpikeSetInfo
    -> GpfaAnalysisParams
    gpfa_rel_path_from_bss : varchar(500) # path to the GPFA output
    ---
    gpfa_hash : char(32) # mostly important to check data consistency if we want...
    condition_nums : blob # will typically be a single condition, but it's a blob in case conditions are combined and it needs to be an array
    condition_grp_fr_thresh : float # the firing rate threshold for each GPFA run (i.e. only trials for this/these condition(s))
    label_use = "stimulusMainLabel" : varchar(50) # label to use for conditions
    cval_train_converged : blob # array with 1 or 0 for each crossvalidation indicating if GPFA converged
    cval_train_full_rank : blob # array with 1 or 0 for each cval indicating whether train set was full rank or not
    max_iterations = 500 : int # max number of iterations, the default of 500 is what was in the code
    tolerance = 1e-8 : float # tolerance for GPFA convergence--the default of 1e-8 was the default in code; what it means is defined in tolerance_type
    tolerance_type = 'ratio' : enum('ratio', 'diff') # type of tolerance used for convergence--see below
    ratio_final = null : blob # this represents the ratio tolerance value (change_curr/change_prev) - 1 which is < (ratio_tolerance) per crossvalidation
    diff_final = null : blob # this represents the diff tolerance value (ll_curr-ll_prev) per crossvalidation
    """
    
    def grabGpfaResults(self, order=True, returnInfo = False, useFa = False):
        defaultParams = loadDefaultParams()
        dataPath = Path(defaultParams['dataPath'])

        gpfaResults = {}
        gpfaOutPaths = []
        # I should be able to get these from a db call, but returning them here
        # makes me more comfortable that they'll be correctly paired with their
        # gpfa results
        bssPaths = []
        datasetNames = []

        gap = GpfaAnalysisParams()
        dsi = DatasetInfo()
        bsi = BinnedSpikeSetInfo()

        if order:
            gpfaInfo = (self * gap * dsi)['method_used="{}"'.format('fa' if useFa else 'gpfa')].fetch('gpfa_rel_path_from_bss', 'bss_relative_path','condition_nums','num_folds_crossvalidate', 'dimensionality', 'dataset_name', 'cval_train_converged', order_by='dataset_id,dimensionality', as_dict=True)
        else:
            gpfaInfo = (self * gap * dsi)['method_used="{}"'.format('fa' if useFa else 'gpfa')].fetch('gpfa_rel_path_from_bss', 'bss_relative_path','condition_nums','num_folds_crossvalidate', 'dimensionality', 'dataset_name', 'cval_train_converged', as_dict=True)

        for info in gpfaInfo:
            bssPath = Path(info['bss_relative_path'])
            relPath = bssPath.parent
            datasetName = info['dataset_name']
#            if 'fss_rel_path_from_parent' in path:
#                relPath /= Path(path['fss_rel_path_from_parent']).parent
#                bssPath = bssPath.parent / path['fss_rel_path_from_parent']
            
            relPathFromBss = info['gpfa_rel_path_from_bss']
            relPathToFile = relPath / relPathFromBss
            fullPath = dataPath / relPathToFile
            gpfaOutPaths.append(fullPath)
            bssPaths.append(bssPath)
            datasetNames.append(datasetName)


            conditionNum = info['condition_nums']
            numCrossvals = info['num_folds_crossvalidate']
            import re
            # gotta string to make it hashable, but string-ing np.array's make
            # lists that have ' ' instead of ',', which makes them un-evalable
            # back to a list >.> So this fixes that.
            condNumStr = re.sub(' ', ',',str(conditionNum))
            relPathAndCond = (str(bssPath), str(numCrossvals), condNumStr)
            if relPathAndCond not in gpfaResults:
                # set the first element to information about this run...
                gpfaResults[relPathAndCond] = {'condition': conditionNum}
            
            dimensionality = info['dimensionality']
            # note we're using numpy to load here

            cValConv = info['cval_train_converged']
            if np.any(cValConv == False):
                # we're gonna rerun GPFA with a higher iteration number and
                # also a lowered threshold so it will converge...
                unconvergedGpfa = self[{k:v for k,v in info.items() if k in ['gpfa_rel_path_from_bss', 'bss_relative_path']}]
                condNums = unconvergedGpfa['condition_nums'][0]
                gpfaRunArgsMap = dict(
                    labelUse = unconvergedGpfa['label_use'][0],
                    conditionNumbersGpfa = condNums,
                    perCondGroupFiringRateThresh = unconvergedGpfa['condition_grp_fr_thresh'][0]
                )
                
                expMaxIterationMaxNum = unconvergedGpfa['max_iterations']
                tolerance = unconvergedGpfa['tolerance']
                tolType = unconvergedGpfa['tolerance_type'][0] # remember to unpack the string
                ratioFinal = unconvergedGpfa['ratio_final']
                diffFinal = unconvergedGpfa['diff_final']
                # I... don't really want to be doing this...
                if False: #expMaxIterationMaxNum < 2000:
                    # we'll up the iteration number to 2000 (double the now-default
                    # of 1000) to give another chance at convergence...
                    expMaxIterationMaxNum = 2000
                    tolerance = 1e-8
#                    expMaxIterationMaxNum = expMaxIterationMaxNum*4 # make sure to make it a float, not an int
#                else:
#                    if tolerance < 1e-6:
#                        tolerance = tolerance*10
#                    else:
#                        # I'm gonna hate myself when overnight code gets stuck here, buuut...
#                        breakpoint() # do not hate thyself
#                if tolerance < 1e-6:
#                    tolerance = tolerance*10
#                else:
#                    if expMaxIterationMaxNum < 2000:
#                        expMaxIterationMaxNum = expMaxIterationMaxNum*4 # make sure to make it a float, not an int
#                    else:
#                        # I'm gonna hate myself when overnight code gets stuck here, buuut...
#                        breakpoint() # do not hate thyself

                    # for the moment we're only running if it hasn't been done at 2000 yet
                    unconvergedGpfa.computeGpfaResults(gap[unconvergedGpfa],bsi[unconvergedGpfa], expMaxIterationMaxNum = expMaxIterationMaxNum, tolerance = tolerance, tolType = tolType, forceNewGpfaRun=True, **gpfaRunArgsMap)

            if True:#not useFa:
                print('Loading %s' % fullPath)
                try:
                    with np.load(fullPath, allow_pickle=True) as gpfaRes:
                        gpfaResLoaded = dict(
                            zip((k for k in gpfaRes), (gpfaRes[k] for k in gpfaRes))
                        )
                except FileNotFoundError:
                    if useFa:
                        breakpoint()
                    print('GPFA saved file not found even though it was in the database, recomputing extraction')
                    unsavedGpfa = self[{k:v for k,v in info.items() if k in ['gpfa_rel_path_from_bss', 'bss_relative_path']}]

                    condNums = unsavedGpfa['condition_nums'][0]
                    combConds = gap[gpfaParams]['combine_conditions']
                    gpfaRunArgsMap = dict(
                        labelUse = unsavedGpfa['label_use'][0],
                        # the way I set it up, setting conditionNumbersGpfa to
                        # None needs to happen if combConds is all
                        conditionNumbersGpfa = condNums if combConds != 'all' else None,
                        perCondGroupFiringRateThresh = unsavedGpfa['condition_grp_fr_thresh'][0]
                    )
                    
                    unsavedGpfa.computeGpfaResults(gap[unsavedGpfa],bsi[unsavedGpfa], **gpfaRunArgsMap)

                    with np.load(fullPath, allow_pickle=True) as gpfaRes:
                        gpfaResLoaded = dict(
                            zip((k for k in gpfaRes), (gpfaRes[k] for k in gpfaRes))
                        )
            else:
                # NOTE this means that this FA test can only be run IF GPFA HAS ALREADY BEEN COMPUTED
                print('Loading FA for %s' % fullPath)
                flName = fullPath.name
                flName = flName[2:] # erm... tacky but it changes gpfa to fa >.>
                fullPath = fullPath.parent / flName
                try:
#                    raise(FileNotFoundError)
                    with np.load(fullPath, allow_pickle=True) as faRes:
                        gpfaResLoaded = dict(
                            zip((k for k in faRes), (faRes[k] for k in faRes))
                        )
                except FileNotFoundError:
                    gpfaParams = self[{k:v for k,v in info.items() if k in ['gpfa_rel_path_from_bss', 'bss_relative_path']}]

                    labelUse = gpfaParams['label_use'][0]
                    condNums = gpfaParams['condition_nums'][0]
                    combConds = gap[gpfaParams]['combine_conditions']
                    gpfaRunArgsMap = dict(
                        labelUse = labelUse,
                        # the way I set it up, setting conditionNumbersGpfa to
                        # None needs to happen if combConds is all
                        conditionNumbersGpfa = condNums if combConds != 'all' else None,
                        perCondGroupFiringRateThresh = gpfaParams['condition_grp_fr_thresh'][0]
                    )
                    
                    # I'm putting this output in the format that saved GPFA files
                    # will have--i.e. a dictionary of variables... of these
                    # specific variables
                    faResLoadedInit = gpfaParams.computeGpfaResults(gap[gpfaParams],bsi[gpfaParams], **gpfaRunArgsMap, useFa=True)

                    if len(faResLoadedInit[0]['faRes'][0]) > 0:
                        with np.load(fullPath, allow_pickle=True) as faRes:
                            gpfaResLoaded = dict(
                                zip((k for k in faRes), (faRes[k] for k in faRes))
                            )
                    else:
                        gpfaResLoaded = []


            gpfaResults[relPathAndCond].update({dimensionality : gpfaResLoaded})

                

        if returnInfo:
            return gpfaResults, dict(gpfaOutPaths = gpfaOutPaths, bssPaths = bssPaths, datasetNames = datasetNames)
        return gpfaResults

    def computeGpfaResults(self, gap, bsiOrFsi, labelUse = "stimulusMainLabel", conditionNumbersGpfa = None, perCondGroupFiringRateThresh = 0.5, expMaxIterationMaxNum = 500, tolerance = 1e-8, tolType = 'ratio', forceNewGpfaRun = False, useFa = False):
        defaultParams = loadDefaultParams()
        dataPath = Path(defaultParams['dataPath'])

        bssKeys = bsiOrFsi.fetch("KEY")
        # gap is GpfaAnalysisParams()
        gapKey = gap.fetch("KEY")

        assert len(gapKey) == 1, "Can only compute one gapKey at a time for now..."
        gapKey = gapKey[0]


        gpfaParamsHash = gap.hash()
        gpfaParams = gap.fetch(as_dict=True)[0]
        # the id isn't one of the params...
        units = gpfaParams.pop('units')
        gpfaParams.pop('gpfa_params_id', None)
        # params for the preparation of bins pre-gpfa
        gpfaPrepParamsMap = dict(
            overall_fr_thresh = 'overallFiringRateThresh',
            balance_conds = 'balanceConds',
            sqrt_spikes = 'sqrtSpikes',
            combine_conditions = 'combineConds',
            on_residuals = 'computeResiduals',
        )
        # changing up the key names to fit into the gpfa function
        gpfaPrepParamsForCall = {gpfaPrepParamsMap[k] : v for k,v in gpfaParams.items() if k in gpfaPrepParamsMap}

        cc = gpfaPrepParamsForCall['combineConds']
        gpfaPrepParamsForCall['combineConds'] = False if cc == 'no' else True
        # remember that gpfaPrepParamsForCall['numConds'] just tells us how many
        # conditions *per extraction*, not *which* conditions--so that's why
        # there's a conditionNumbersGpfa input.
        #
        # Put another way, say combine_conditions='no'; in that case, num_conds
        # (from gap) will *always* be 1, because you're not combining
        # conditions, so each extraction is from only one condition! But
        # numConds in bS.gpfa() asks us *which* conditions, which is what this
        # is now changed to.  Note that numConds=None actually means *all*
        # conditions, but it's unclear how many conditions each bS will have
        # (which is actually why this is not stored in the GpfaAnalysisParams
        # table)
        numberOfConditions = gpfaParams['num_conds']
        # this if statement deals with what happens when you start combining conditions
        if cc != 'no' and ((conditionNumbersGpfa is None and numberOfConditions != 0) or ((conditionNumbersGpfa is not None) and len(conditionNumbersGpfa) != numberOfConditions)):
            breakpoint()
            # the first parenthetical asks that if you want to use all
            # conditions, the numberOfConditions should equal zero you want to
            # use all the conditions (conditionNumbersGpfa is
            # None) then your numberOfConditions should be set to 0
            #
            # the second statement checks that if you *don't* want all the
            # conditions, then the number of conditions provided
            # len(conditionNumbersGpfa) should be equal to the numberOfConditions
            # variable. Note that len() doesn't work on None variables, so I've
            # gotta check that it's not none again
            raise Exception("Incorrect number of conditions provided")
        gpfaPrepParamsForCall['condNums'] = conditionNumbersGpfa # if cc == 'no' else nc
        gpfaPrepParamsForCall['perConditionGroupFiringRateThresh'] = perCondGroupFiringRateThresh
        gpfaPrepParamsForCall['labelUse'] = labelUse

        # params for the actual call to the gpfa computation
        gpfaCallParamsMap = dict(
            num_folds_crossvalidate = 'crossvalidateNum',
            dimensionality = 'xDim',
        )
        gpfaCallParams = {gpfaCallParamsMap[k] : v for k,v in gpfaParams.items() if k in gpfaCallParamsMap}
        gpfaCallParams['labelUse'] = labelUse
        gpfaCallParams['expMaxIterationMaxNum'] = expMaxIterationMaxNum
        gpfaCallParams['tolerance'] = tolerance
        gpfaCallParams['tolType'] = tolType

        retValsAll = []
        for key in bssKeys:
            relPath = Path(key['bss_relative_path']).parent 
            if useFa:
                gpfaRelPathFromBss = Path('fa') / ('params_%s' % gpfaParamsHash[:5] ) 
            else:
                gpfaRelPathFromBss = Path('gpfa') / ('params_%s' % gpfaParamsHash[:5] ) 
            fullPathToConditions = dataPath / relPath / gpfaRelPathFromBss
            binnedSpikeSet = bsiOrFsi[key].grabBinnedSpikes()

            assert len(binnedSpikeSet)==1, "Too many BSS with key"
            bss = binnedSpikeSet[0]
            bss = bss.convertUnitsTo(units)

            groupedBalancedSpikes, condDescriptors, condsUse = bss.prepareGpfaOrFa(**gpfaPrepParamsForCall)
            if useFa:
                retVals = bss.fa(groupedBalancedSpikes, fullPathToConditions, condDescriptors, **gpfaCallParams)
                # retVals = {}
                # retVals['faRes'] = bss.fa(groupedBalancedSpikes, fullPathToConditions, condDescriptors, **gpfaCallParams)
                # retVals['groupedBalancedSpikes'] = groupedBalancedSpikes
                # retVals['pathToCond'] = fullPathToConditions
                # retVals['condDescriptors'] = condDescriptors
                retValsAll.append(retVals)
                # continue
            else:
                retVals = bss.gpfa(groupedBalancedSpikes, fullPathToConditions, condDescriptors, **gpfaCallParams, forceNewGpfaRun = forceNewGpfaRun)
                retValsAll.append(retVals)

            condPaths = retVals[-1]
            gpfaRunFinalDets = retVals[-2]
            condInfo = list(zip(condsUse, condPaths))
            gpfaPrepComp = retVals[0]


            for cI, gPC, rgfD in zip(condInfo, gpfaPrepComp, gpfaRunFinalDets):
                conditionNumsUsed = np.array(cI[0]) # so it can be saved in DataJoint
                conditionSavePaths = str(cI[1]) # so it can be saved in DataJoint
                crossvalFullRankStat = np.array(rgfD[0])
                crossvalConvergeStat = np.array(rgfD[1])
                crossvalFinalRatio = np.array(rgfD[2])
                crossvalFinalDiff = np.array(rgfD[3])

                gpfaHash = hashlib.md5(str(gPC).encode('ascii')).hexdigest()
                try:
                    newGaiRow = dict(
                        **key,
                        **gapKey,
                        gpfa_rel_path_from_bss = str(Path(conditionSavePaths).relative_to(dataPath / relPath)), 
                        gpfa_hash = gpfaHash, 
                        condition_nums = conditionNumsUsed, 
                        condition_grp_fr_thresh = perCondGroupFiringRateThresh,
                        label_use = labelUse,
                        cval_train_converged = crossvalConvergeStat,
                        cval_train_full_rank = crossvalFullRankStat,
                        max_iterations = int(expMaxIterationMaxNum),
                        tolerance = float(tolerance),
                        tolerance_type = tolType,
                        ratio_final = crossvalFinalRatio,
                        diff_final = crossvalFinalDiff,
                    )
                    self.insert1(newGaiRow)
                except IntegrityError as e:
                    if str(e).find("UNIQUE constraint failed") != -1 and forceNewGpfaRun:
                        print("Be careful using this... GPFA was forced to run again without changing recorded parameters--could result in data consistency issues!")
                        # define it by its class to avoid the relation
                        # disappearing as the database changes... this has
                        # happened before >.>
                        gaiToUpdate = self.__class__()[{k:v for k,v in newGaiRow.items() if k in self.fetch('KEY')[0].keys()}]
                        gaiValsOriginal = gaiToUpdate.fetch(as_dict=True)
                        newValChecks = [(ky,val==gaiValsOriginal[0][ky]) for ky,val in newGaiRow.items() if ky in gaiValsOriginal[0].keys() and ky not in self.fetch('KEY')[0].keys()]
                        for vl in newValChecks:
                            if not np.all(vl[1]):
                                gaiToUpdate._update(vl[0], value = newGaiRow[vl[0]])
                    else:
                        # check if this could happen because you're trying to
                        # insert something that's there with no changes to
                        # uniqueness...
#                        breakpoint()
                        pass
#                        raise e
        return retValsAll

    @staticmethod
    def rmFilesByKey(key, quickDelete=False):
        defaultParams = loadDefaultParams()
        dataPath = Path(defaultParams['dataPath'])

        relPath = Path(key['bss_relative_path']).parent 

        relPath /= Path(key['gpfa_rel_path_from_bss'])
        fullPath = dataPath / relPath
        gpfaResultFilepaths = [str(pth) for pth in fullPath.parent.glob('**/*')]
        if len(gpfaResultFilepaths) == 0:
            print('In "%s"\n' % fullPath)
            print('there were no GPFA result files to delete. Removing entry from db.')
            return True

        info = "About to delete\n\n" + "\n".join(gpfaResultFilepaths) + "\n\n"
        prompt = "Please confirm:"
        if not quickDelete:
            response = userChoice(prompt, info, choices=("yes", "no"), default="no")
        if quickDelete or response == "yes":
            [Path(pth).unlink() for pth in gpfaResultFilepaths]
        
        fullPath.parent.rmdir()
        try:
            # if all conditions get erased... just a way to clear up the file system
            fullPath.parent.parent.rmdir()
        except OSError as e:
            # probably still a file in the folder, so just carry on
            print(e)
            pass

        return quickDelete or response == 'yes'
        
    @classmethod
    def rmFilesByKeyList(cls, keyList, keysDeletedList, quickDelete=False):
        for key in keyList:
            cls.rmFilesByKey(key, quickDelete=quickDelete)
            # this is meant to track the deleted keys for error handling--since
            # it's a reference, that calling function can see it, and
            # appropriately delete database entries whose keys have already
            # been used to delete files...
            keysDeletedList.append(key)

    def delete(self,quickDelete=False):
        print('zxczm')
        gaiKeys = self.fetch('KEY') # always a dict

        if not quickDelete:
            info = "About to delete %d entries *and their associated GPFA files*" % len(gaiKeys)
            prompt = "Are you sure?"
            response = userChoice(prompt, info, choices=("yes", "no"), default="no")
            if response == "yes":
                quickDelete = True
            else:
                quickDelete = False
                return


        for key in gaiKeys:
            deleteDbEntry = self.rmFilesByKey(key, quickDelete=quickDelete)

            if deleteDbEntry:
                with dj.config(safemode = not quickDelete) as cfg: 
                    # this syntax creates a super instance of *just* the subset of
                    # self--so that we don't delete all of self in one go
                    # accidentally!
                    super(GpfaAnalysisInfo, self[key]).delete()

    def delete_quick(self, get_count = False):
        print('woiqu')
        gaiKeys = self.fetch('KEY') # always a dict
        try:
            keysDelList = []
            self.rmFilesByKeyList(gaiKeys, keysDelList, quickDelete=True)
        except Exception as e:
            print(e)
            print('Failed to delete one or more files for above reason... deleting database entries of files deleted')
            return super(GpfaAnalysisInfo, self[keysDelList]).delete_quick(get_count = get_count)
        else:
            return super(GpfaAnalysisInfo, self[gaiKeys]).delete_quick(get_count = get_count)


    def drop(self):
        self.delete() # take care of all those paths first...
        super().drop()


@schema
class AnalysisRunInfo(dj.Manual):
    definition = """
    # analysis file/run info
    analysis_method : varchar(100) # name of analysis file run
    method_call_signature : varchar(500) # calling signature for the method
    git_commit : char(32) # don't remember how long a hash a git commit has
    patch : varchar(500000) # this is  super long... but it's to keep the patch information so the analysis can be recreated
    ---
    method_inputs : blob # the inputs to this call--since all calls are different, this is a blob!
    date_start : timestamp # date analysis started running
    date_end : timestamp # date analysis finished running
    output_files = null : blob # this is gonna contain output file paths that aren't figures!
    output_figures_relative_path = null : blob # this is gonna contain output figure paths relative to the output figure directory for this analysis
    metadata = null : blob # this has metadata that would likely be stored by the patch as well, but in case its easier to grab from here...
    """




if __name__ == '__main__':
#    ds = DatasetInfo()
#    dt = [{'dataset_path' : 'yo', 'dataset_meta_data' : 'toodles' }, {'dataset_path' : 'wazam', 'dataset_meta_data' : 'ciao'}]
#    ds.insert(dt)
#
#    dt = [{'dataset_path' : 'hey', 'dataset_meta_data' : 'bye' }, {'dataset_path' : 'boom', 'dataset_meta_data' : 'ciao'}]
#    ds.insert(dt)
#
#    (ds & 'dataset_id=14').delete()
#
#    bss = BinnedSpikeSetInfo()
#    bdt = [{'dataset_id': 1, 'bss_path' : 'bsswizzle', 'bss_params' : 'bssbye' }, {'dataset_id': 2, 'bss_path' : 'bssboom', 'bss_params' : 'bsssssyyyyee'}]
#    bss.insert(bdt)

    breakpoint()
