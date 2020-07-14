"""
This file will contain all the setup of tables and such that sets up Data
Joint so I can access my data! Woo!
"""

import sys
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from methods.GeneralMethods import loadDefaultParams, userChoice

defaultParams = loadDefaultParams(defParamBase = ".")
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
    """


    class DatasetSpecificLoadParams(dj.Part):
        definition = """
        # dataset info extraction params
        -> DatasetInfo
        ds_spec_params_id : int # params id
        ---
        ignore_channels : blob # channels removed during dataset extraction; stored as array
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
#        if expression is None:
#            dsExp = self
#        else:
#            dsExp = (self & expression)

        # give some ordering here
        dsPaths, dsIds = self.fetch('dataset_relative_path', 'dataset_id', order_by='dataset_id')
        for path, dsId in zip(dsPaths, dsIds):
            fullPath = dataPath / path
            with fullPath.open(mode='rb') as datasetDillFh:
                dataset = pickle.load(datasetDillFh)
                dataset.id = dsId
                datasets.append(dataset)

        return datasets

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

        for datasetInfo in self:
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
                with saveBSSPath.open(mode='rb') as saveBSSFh:
                    binnedSpikesHere = pickle.load(saveBSSFh)

            else:
                # Here is the meat of the function, where the BinnedSpikeSets are actually computed...
                dataset = self[datasetInfo].grabDatasets()
                assert len(dataset) == 1, "More than one dataset being grabbed per key?"
                dataset = dataset[0]

                dsId = dataset.id
                stateNameStateStart = dataset.keyStates[keyStateName]


                # Filter by trial type
                if trialType == 'successful':
                    dataset = dataset.successfulTrials()
                elif trialType == 'failure':
                    dataset = dataset.failTrials()


                # These extra filters are in a { description : lambda } style dictionary
                if trialFilterLambdaDict is not None:
                    for tFLDesc, tFL in trialFilterLambdaDict.items():
                        lambdaFilt = eval(tFL)
                        dataset = lambdaFilt(dataset)

                alignmentStates = dataset.metastates
                    
                # Filter by trial length
                startState, endState, stateNameAfter = dataset.computeStateStartAndEnd(stateName = stateNameStateStart, ignoreStates=alignmentStates)

                startStateArr = np.asarray(startState)
                endStateArr = np.asarray(endState)
                timeInStateArr = endStateArr - startStateArr

                dataset = dataset.filterTrials(timeInStateArr>lenSmallestTrial)

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

                # Filter to only include high firing rate channels
                binnedSpikesHere = binnedSpikesHere.channelsAboveThresholdFiringRate(firingRateThresh=firingRateThresh)[0]

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
                _, chansGood = binnedCountsPerTrial.channelsBelowThresholdFanoFactor(fanoFactorThresh=fanoFactorThresh)
                binnedSpikesHere = binnedSpikesHere[:,chansGood]

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
                        
                    dsiPks = self[datasetInfo].fetch1("KEY")
                    binnedSpikeSetHereInfo.update(dsiPks)


                    bssi.insert1(binnedSpikeSetHereInfo)

                    bssAdded = bssi[{k : v for k, v in binnedSpikeSetHereInfo.items() if k in bssi.primary_key}]
                    # add this binned spike set as a filtered spike set part as
                    # well... (obviously with no filtering, hence the 'original'
                    # reason
                    _ = bssAdded.filterBSS(filterReason = 'original', filterDescription = 'unfiltered original', condLabel = 'stimulusMainLabel')

                    bssiKey = bssAdded.fetch("KEY", as_dict=True)

            if units is not None:
                binnedSpikesHere.convertUnitsTo(units=units)
            
            binnedSpikes.append(binnedSpikesHere)
            bssiKeys.append(bssiKey)

        return binnedSpikes, bssiKeys

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
            keysDelList = []
            self.rmFilesByKeyList(dsiKeys, keysDelList, quickDelete=True)
        except Exception as e:
            print(e)
            print('Failed to delete one or more files for above reason... deleting database entries of files deleted')
            return super(DatasetInfo, self[keysDelList]).delete_quick(get_count = get_count)
        else:
            return super(DatasetInfo, self[dsiKeys]).delete_quick(get_count = get_count)


@schema
class BinnedSpikeSetProcessParams(dj.Manual):
    definition = """
    # dataset info extraction params
    bss_params_id : int auto_increment # params id
    ---
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
    end_time : blob # true end_time in ms of each trisl -- end_time_alignment + end_offset
    """

    class FilteredSpikeSetInfo(dj.Part):
        definition = """
        # for filtered spike sets
        -> BinnedSpikeSetInfo
        fss_rel_path_from_parent: varchar(100) # this is the directory within above's relative path where this set lives
        fss_param_hash : char(32) # for looking this entry up
        ---
        fss_hash : char(32) # for the data integrity
        trial_filter = null : blob # filter of the trial on the binned spike set info
        trial_num : int # number of trials; useful for filtering shuffles
        ch_filter = null : blob # filter of the channels on the binned spike set info
        ch_num : int # number of channels; useful for filtering shuffles
        filter_reason : enum('shuffle', 'other')
        filter_description = null : varchar(100) # reason for filter (if not shuffle, usually)
        """

    def filterBSS(self, filterReason, filterDescription, binnedSpikeSet = None, trialFilter = None, channelFilter = None):
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
                bssFiltered = bssFiltered[trialFilter]
                bssFilterParams.update(dict(trial_filter = trialFilter))

            if channelFilter is not None:
                bssFiltered = bssFiltered[:, channelFilter]
                bssFilterParams.update(dict(ch_filter = channelFilter))

            bssFilterParams.update(dict(
                    trial_num = bssFiltered.shape[0],
                    ch_num = bssFiltered.shape[1]
                ))


            defaultParams = loadDefaultParams()
            dataPath = Path(defaultParams['dataPath'])

            bssPath = Path(key['bss_relative_path']).parent # because it includes the .dill file
            filteredSpikeSetDill = 'filteredSpikeSet.dill'

            # a nice way to distinguish the path for each filter based on extraction parameters...
            bSSFilteredProcParamsJson = json.dumps(str(bssFilterParams), sort_keys=True) # needed for consistency as dicts aren't ordered
            bSSFilteredProcParamsHash = hashlib.md5(bSSFilteredProcParamsJson.encode('ascii')).hexdigest()

            pathRelativeToParent = Path('filteredSpikes_%s' % bSSFilteredProcParamsHash[:5]) / filteredSpikeSetDill
            fullPath = dataPath / bssPath / pathRelativeToParent

            fsi = self.FilteredSpikeSetInfo()
            existingSS = fsi[{k : v for k,v in bssFilterParams.items() if k in fsi.primary_key}] 
            if len(existingSS) > 1:
                raise Exception('Multiple filtered spike sessions have been saved with these parameters')
            elif len(existingSS) > 0:
                if not fullPath.exists():
                    print('Db record existed for FilteredSpikeSet but not file... saving file now')
                    fullPath.parent.mkdir(parents=True, exist_ok = True)
                    with fullPath.open(mode='wb') as filteredSpikeSetDillFh:
                        pickle.dumps(bssFiltered, filteredSpikeSetDillFh)
#                with fullPath.open(mode='rb') as filteredSpikeSetDillFh:
#                    bssFiltered = pickle.load(filteredSpikeSetDillFh)
            else:
                fullPath.parent.mkdir(parents=True, exist_ok = True)
                with fullPath.open(mode='wb') as filteredSpikeSetDillFh:
                    pickle.dump(bssFiltered, filteredSpikeSetDillFh)

                bssFilteredHash = hashlib.md5(str(bssFiltered).encode('ascii')).hexdigest()

                bssFilterInfoVals = dict(**bssFilterParams,
                        fss_param_hash = bSSFilteredProcParamsHash,
                        fss_rel_path_from_parent = str(pathRelativeToParent),
                        fss_hash = bssFilteredHash
                    )

                fsi.insert1(dict(
                        **key,
                        **bssFilterInfoVals
                    ))
        return bssFiltered

    # for filtering spike sets... note that only filterParams
    def grabBinnedSpikes(self, returnPath = False):
        defaultParams = loadDefaultParams()
        dataPath = Path(defaultParams['dataPath'])

        binnedSpikeSets = []

        # give some ordering here
        bssPaths = self.fetch('bss_relative_path')
        for path in bssPaths:
            fullPath = dataPath / path
            with fullPath.open(mode='rb') as binnedSpikeSetDillFh:
                bss = pickle.load(binnedSpikeSetDillFh)
                binnedSpikeSets.append(bss)

        if returnPath:
            return binnedSpikeSets, bssPaths
        return binnedSpikeSets

if __name__ == '__main__':
    ds = DatasetInfo()
    dt = [{'dataset_path' : 'yo', 'dataset_meta_data' : 'toodles' }, {'dataset_path' : 'wazam', 'dataset_meta_data' : 'ciao'}]
    ds.insert(dt)

    dt = [{'dataset_path' : 'hey', 'dataset_meta_data' : 'bye' }, {'dataset_path' : 'boom', 'dataset_meta_data' : 'ciao'}]
    ds.insert(dt)

    (ds & 'dataset_id=14').delete()

    bss = BinnedSpikeSetInfo()
    bdt = [{'dataset_id': 1, 'bss_path' : 'bsswizzle', 'bss_params' : 'bssbye' }, {'dataset_id': 2, 'bss_path' : 'bssboom', 'bss_params' : 'bsssssyyyyee'}]
    bss.insert(bdt)

    breakpoint()
