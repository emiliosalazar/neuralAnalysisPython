from pathlib import Path
import json

def loadDefaultParams(defParamName = "defParams.txt", defParamBase=""):
    defParamBasepath = Path(defParamBase)
    defParamFile = defParamBasepath / Path(defParamName)

    return json.load(defParamFile.open())
