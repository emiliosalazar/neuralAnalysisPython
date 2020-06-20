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
    remove_coincident_spikes : enum(0, 1) # flag on whether these were removed
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
    def grabDatasets(self, expression = None):
        import dill as pickle
        defaultParams = loadDefaultParams()
        dataPath = Path(defaultParams['dataPath'])

        datasets = []
        if expression is None:
            dsExp = self
        else:
            dsExp = (self & expression)

        dsPaths, dsIds = dsExp.fetch('dataset_relative_path', 'dataset_id')
        for path, dsId in zip(dsPaths, dsIds):
            fullPath = dataPath / path
            with fullPath.open(mode='rb') as datasetDillFh:
                dataset = pickle.load(datasetDillFh)
                dataset.id = dsId
                datasets.append(pickle.load(datasetDillFh))

        return datasets

@schema
class BinnedSpikeSetProcessParams(dj.Manual):
    definition = """
    # dataset info extraction params
    bss_params_id : int auto_increment # params id
    ---
    start_offset : int # offset in ms (from start_time_alignment in the specific params)
    end_offset : int # offset in ms (from end_time_alignment in the specific params)
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
    binned_spike_set_id: int # binned spike set id
    ---
    bss_relative_path: varchar(500) # path to processed binned spike set
    bss_hash : char(32) # mostly important because of the hash filepath can create to check data consistency
    """

    class BinnedSpikeSetSpecificProcessParams(dj.Part):
        definition = """
        # dataset info extraction params
        -> BinnedSpikeSetInfo
        ds_spec_params : int auto_increment # params id
        ---
        alignment_state : varchar(100) # name of alignment state (note this might be offset from the start_time)
        start_time_alignment : blob # start time in ms of the alignment_state
        start_time : blob # true start_time in ms of each trial -- start_time_alignment + start_offset
        end_time_alignment : blob # end time in ms from where each bss trial end was offset
        end_time : blob # true end_time in ms of each trisl -- end_time_alignment + end_offset
        """


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
