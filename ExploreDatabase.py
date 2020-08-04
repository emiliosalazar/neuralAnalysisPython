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
    if False:
        import json
        import hashlib
        import re
        import shutil
        bspVals = bsp.fetch(as_dict=True)
        for bspDict in bspVals:
            bspDictId = bspDict.pop('bss_params_id')

            bspDictPk = {'bss_params_id' : bspDictId}
            bspHere = bsp[bspDictPk]
            binSpks = bsi[bspHere]

            # check that all the BSPs came from the same key state
            keyStatesStart = []
            keyStatesEnd = []
            for bnSpIn in binSpks:
                bsiHere = bsi[{k:v for k,v in bnSpIn.items() if k in bsi.primary_key}]
                keyStateNames = dsi[bsiHere].grabDatasets()[0].keyStates
                stState = bnSpIn['start_alignment_state']
                endState = bnSpIn['end_alignment_state']
                keyStateStart = [kyNm for kyNm, stNm in keyStateNames.items() if stNm == stState][0]
                keyStatesStart.append(keyStateStart)
                keyStateEnd = [kyNm for kyNm, stNm in keyStateNames.items() if stNm == stState][0]
                keyStatesEnd.append(keyStateEnd)

            # set said key state in the bsp dict
            if np.all(keyStateStart == np.array(keyStatesStart)) and np.all(keyStateEnd == np.array(keyStatesEnd)):
                if bspDict['key_state_start'] != keyStateStart or bspDict['key_state_end'] != keyStateEnd:
                    breakpoint()
                    bspDict['key_state_start'] = keyStateStart
                    bspDict['key_state_end'] = keyStateEnd
                    bspHere._update('key_state_start', value=bspDict['key_state_start'])
                    bspHere._update('key_state_end', value=bspDict['key_state_end'])
            else:
                breakpoint()
                raise Exception


            # compute new hash
            bspDictJson = json.dumps(bspDict, sort_keys=True)
            bspDictHash = hashlib.md5(bspDictJson.encode('ascii')).hexdigest()
            bspDictHashNew = bspDictHash[:5]

            for bnSpk in binSpks:
                bsiHere = bsi[{k:v for k,v in bnSpk.items() if k in bsi.primary_key}]
                bssOldPthToDill = dataPath / bnSpk['bss_relative_path']

                bssOldPthParent = bssOldPthToDill.parent

                bssNewPthToDill = Path(re.sub(r'/binnedSpikeSet_.*?/', '/binnedSpikeSet_%s/' % bspDictHashNew, str(bssOldPthToDill)))
                bssNewPthParent = bssNewPthToDill.parent
                bssNewRelPthToDillForDb = str(bssNewPthToDill.relative_to(dataPath))

#            breakpoint()
                try:
                    pass
                    bsiHere._update('bss_relative_path', value=bssNewRelPthToDillForDb)
                except:
                    breakpoint()
                try:
                    print('****\nmoving\n\n%s\n\nto\n\n%s\n****' % (bssOldPthParent, bssNewPthParent))
                    shutil.move(str(bssOldPthParent), str(bssNewPthParent))
                except FileNotFoundError:
                    print('%s has probably already been moved' % bssOldPthParent)

            breakpoint()
            binSpks['bss_relative_path']
    breakpoint()
    #    gai['start_alignment_state="Target"']['dataset_id=6'][gap['dimensionality=5']].delete()
