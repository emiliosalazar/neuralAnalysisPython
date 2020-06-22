"""
This file will contain all the setup of tables and such that sets up Data
Joint so I can access my data! Woo!
"""

import sys
from pathlib import Path

from methods.GeneralMethods import loadDefaultParams

defaultParams = loadDefaultParams(defParamBase = ".")
sys.path.append(defaultParams['datajointLibraryPath'])
import datajoint as dj
import dill as pickle
import hashlib
import json

dbLocation = defaultParams['databaseHost']
dbPort = defaultParams['databasePort']
dataPath = defaultParams['dataPath']
if dbPort == 'sqlite':
    dbLocation = str(Path(dataPath) / dbLocation)

dj.config['database.user'] = 'emilio'
dj.config['database.port'] = dbPort
dj.config['database.host'] = dbLocation
dj.config['database.password']=''

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
