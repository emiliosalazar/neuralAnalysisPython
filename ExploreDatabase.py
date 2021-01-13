"""
This is a script useful for whenever I want to explore the database--it sets up
a bunch of variables and parameters to get crackin'!
"""

from setup.DataJointSetup import DatasetGeneralLoadParams,DatasetInfo,BinnedSpikeSetProcessParams,BinnedSpikeSetInfo,FilterSpikeSetParams,GpfaAnalysisParams,GpfaAnalysisInfo,AnalysisRunInfo
import numpy as np
from pathlib import Path
from methods.GeneralMethods import loadDefaultParams

defaultParams = loadDefaultParams(defParamBase = ".")
dataPath = Path(defaultParams['dataPath'])

dsgl = DatasetGeneralLoadParams()
dsi = DatasetInfo()
bsp = BinnedSpikeSetProcessParams()
bsi = BinnedSpikeSetInfo()
fsp = FilterSpikeSetParams()
gap = GpfaAnalysisParams()
gai = GpfaAnalysisInfo()
ari = AnalysisRunInfo()

if __name__ == "__main__":
#    dsgl._update('remove_coincident_chans', False)
    breakpoint()

