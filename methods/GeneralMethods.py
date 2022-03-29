from pathlib import Path

def loadDefaultParams(defParamName = "paramDefs.txt", defParamBase=None):
    import json

    if defParamBase is None:
        # sadly enforces that this file must be one directory down from where
        # the configs are... but I don't know that anything can be done about
        # it
        defParamBasepath = Path(__file__).parent.parent.absolute()
    else:
        defParamBasepath = Path(defParamBase).absolute()

    defParamFile = defParamBasepath / Path(defParamName)

    return json.load(defParamFile.open())

# Shamelessley ripped from DataJoint, but I'd rather have it here because it
# seems more like an internal DataJoint util not meant to be used by other
# libraries directly
def userChoice(prompt, info, choices=("yes", "no"), default=None):
    """
    Prompts the user for confirmation.  The default value, if any, is capitalized.
    :param prompt: Information to display to the user.
    :param choices: an iterable of possible choices.
    :param default: default choice
    :return: the user's choice
    """
    assert default is None or default in choices
    choice_list = ', '.join((choice.title() if choice == default else choice for choice in choices))
    response = None
    while response not in choices:
        print(info)
        response = input(prompt + ' [' + choice_list + ']: ')
        response = response.lower() if response else default
    return response


def prepareMatlab(eng=None, pathsAdd = ['matlabCodePath']):
    # print('HI')
    # from importlib import reload
    # import matlab
    # reload(matlab)
    from matlab import engine
    if type(pathsAdd) is not list:
        pathsAdd = [pathsAdd]
    pathsToAdd = []
    for path in pathsAdd:
        defaultParams = loadDefaultParams()
        if Path(path).exists():
            pathsToAdd.append(Path(path))
        elif path in defaultParams:
            path = defaultParams[path]
            pathsToAdd.append(Path(path))
        else:
            raise Exception('path for addition to Matlab path is neither a key in default params nor a true path') 
    if eng is not None:
        try:
            eng.workspace
        except (engine.RejectedExecutionError, NameError) as e:
            eng = engine.start_matlab()
        finally:
            eng.clear('all', nargout=0)
            # add the gpfaEngine path
            for path in pathsToAdd:
                eng.evalc("addpath(genpath('"+str(path)+"'))")
    else:
        # print('HELLO2')
        # print(engine._engines)
        # print(engine._engine_lock)
        eng = engine.start_matlab("-nodesktop -nodisplay -nojvm")
        
        print('Matlab started')
        eng.clear('all', nargout=0)
        # add the gpfaEngine path
        for path in pathsToAdd:
            eng.evalc("addpath(genpath('"+str(path)+"'))")
        
    return eng

def pMat(mlabEng):
    from matlab.engine import pythonengine
    from matlab.engine import MatlabEngine
    print('yo')
    rah = pythonengine.createMATLABAsync(['-nodesktop'])
    print('sup')
    ja = mlabEng(rah)
    print('yup')
    k = MatlabEngine(ja)
    print('bye')
    
    return k
    

def saveFiguresToPdf(pdfname=None,analysisDescription = None,figNumsToSave=None, subplotsMoved = False):
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib import pyplot as plt
    import datetime
    
    params = loadDefaultParams()
    savePath = Path(params['figurePath'])
    today = datetime.datetime.today() 
    
    if analysisDescription is None:
        analysisDescription = pdfname
    
    if pdfname is None:
        pdfname = savePath / (today.strftime('%Y%m%d') + "_resultFigures") / (today.strftime('%Y%m%d') + "_resultFigure.pdf")
    else:
        pdfname = savePath / (today.strftime('%Y%m%d') + "_" + analysisDescription) / (today.strftime('%Y%m%d') + "_" + pdfname)
        pdfname = pdfname.with_suffix(".pdf")
    
    pdfname.parent.mkdir(parents=True, exist_ok = True)
    
    if figNumsToSave is None:
        figNumsToSave = plt.get_fignums()
        
    with PdfPages(pdfname) as pdf:
        for fig in figNumsToSave: 
            if subplotsMoved:
                pdf.savefig( fig ,bbox_inches="tight",pad_inches=2)
            else: 
                pdf.savefig( fig ,bbox_inches="tight")

    return str(pdfname.relative_to(savePath))
