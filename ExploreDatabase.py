"""
This is a script useful for whenever I want to explore the database--it sets up
a bunch of variables and parameters to get crackin'!
"""

from setup.DataJointSetup import DatasetGeneralLoadParams,DatasetInfo,BinnedSpikeSetProcessParams,BinnedSpikeSetInfo,FilterSpikeSetParams,GpfaAnalysisParams,GpfaAnalysisInfo
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

if __name__ == "__main__":
    if False:
        import json
        import hashlib
        import re
        import shutil
        bspVals = bsp.fetch(as_dict=True)
        for bspDict in bspVals:
            bspDictId = bspDict.pop('bss_params_id')
            bspDictJson = json.dumps(bspDict, sort_keys=True)
            bspDictHash = hashlib.md5(bspDictJson.encode('ascii')).hexdigest()
            bspDictHashNew = bspDictHash[:5]

            bspDictPk = {'bss_params_id' : bspDictId}
            binSpks = bsi[bsp[bspDictPk]]

            for bnSpkIn in binSpks:
                bsiHere = bsi[{k:v for k,v in bnSpkIn.items() if k in bsi.primary_key}]
                bssOldPthToDill = dataPath / bnSpkIn['bss_relative_path']

                bssOldPthParent = bssOldPthToDill.parent

                bssNewPthToDill = Path(re.sub(r'/binnedSpikeSet_.*?/', '/binnedSpikeSet_%s/' % bspDictHashNew, str(bssOldPthToDill)))
                bssNewPthParent = bssNewPthToDill.parent
                bssNewRelPthToDillForDb = str(bssNewPthToDill.relative_to(dataPath))

#            breakpoint()
                try:
                    bsiHere._update('bss_relative_path', value=bssNewRelPthToDillForDb)
                except:
                    breakpoint()
                try:
                    print('moving\n\n%s\n\nto\n\n%s' % (bssOldPthParent, bssNewPthParent))
                    shutil.move(str(bssOldPthParent), str(bssNewPthParent))
                except FileNotFoundError:
                    print('%s has probably already been moved' % bssOldPthParent)

            binSpks['bss_relative_path']
    breakpoint()
    #    gai['start_alignment_state="Target"']['dataset_id=6'][gap['dimensionality=5']].delete()
