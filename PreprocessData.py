# just some quick methods to preprocess PFC spike data by running it through
# NAS...
# this also could be written in matlab...
# but it wasn't...
# and instead rewrites teh smithLabNevSort script basically...


import glob
from methods.GeneralMethods import loadDefaultParams
from methods.GeneralMethods import prepareMatlab
from pathlib import Path

def main():
    defaultParams = loadDefaultParams()
    newDataPath = defaultParams['newDataPath']
    nevFiles = glob.glob(newDataPath + "/*.nev")
    matFiles = glob.glob(newDataPath + "/*.mat")

    # open a matlab instance
    eng = prepareMatlab()
    for nevFile in nevFiles:
        nvName = Path(nevFile).stem
        matchMatFile = glob.glob(newDataPath + nvName + '*.mat')
        if len(matchMatFile)>0:
            continue
        eng.workspace['nevFile'] = nevFile
        eng.workspace['nasNetBasename'] = defaultParams['nasNetPFCV4']
        eng.workspace['gamma'] = 0.2
        eng.workspace['wvLen'] = 52
        
        # first array...
        eng.workspace['runNas'] = True
        eng.workspace['nmSuffix'] = 'Array1'
        eng.evalc('channelsGrab = 1:200')
        eng.eval("nasSortGeneral(nevFile, nasNetBasename, gamma, wvLen, 'runNas', runNas, 'channelsGrab', channelsGrab, 'nmSuffix', nmSuffix)")

        # second array...
        eng.workspace['runNas'] = False
        eng.workspace['nmSuffix'] = 'Array2'
        eng.evalc('channelsGrab = 201:400')
        eng.eval("nasSortGeneral(nevFile, nasNetBasename, gamma, wvLen, 'runNas', runNas, 'channelsGrab', channelsGrab, 'nmSuffix', nmSuffix)")



if __name__ == '__main__':
    main()
