from pathlib import Path

def loadDefaultParams(defParamName = "paramDefs.txt", defParamBase=""):
    import json

    defParamBasepath = Path(defParamBase).absolute()
    defParamFile = defParamBasepath / Path(defParamName)

    return json.load(defParamFile.open())

def prepareMatlab(eng=None):
    from matlab import engine
     
    if eng is not None:
        try:
            eng.workspace
        except (engine.RejectedExecutionError, NameError) as e:
            eng = engine.start_matlab()
        finally:
            eng.clear('all', nargout=0)
            # add the gpfaEngine path
            defaultParams = loadDefaultParams(defParamBase = ".")
            matlabCodePath = Path(defaultParams['matlabCodePath'])
            eng.evalc("addpath(genpath('"+str(matlabCodePath)+"'))")
    else:
        eng = engine.start_matlab()
        eng.clear('all', nargout=0)
        # add the gpfaEngine path
        defaultParams = loadDefaultParams(defParamBase = ".")
        matlabCodePath = Path(defaultParams['matlabCodePath'])
        eng.evalc("addpath(genpath('"+str(matlabCodePath)+"'))")
        
    return eng