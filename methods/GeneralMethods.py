from pathlib import Path

def loadDefaultParams(defParamName = "paramDefs.txt", defParamBase="."):
    import json

    defParamBasepath = Path(defParamBase).absolute()
    defParamFile = defParamBasepath / Path(defParamName)

    return json.load(defParamFile.open())

def prepareMatlab(eng=None):
    # print('HI')
    # from importlib import reload
    # import matlab
    # reload(matlab)
    from matlab import engine
    # print('Matlab imported')
    # k = globals()
    # print(k.keys())
    # if "__warningregistry__" in k.keys():
    #     print(k["__warningregistry__"])
    # if "__cached__" in k.keys():
    #     print(k["__cached__"])
    # import multiprocessing as mp
    # print(mp.current_process())
    # print(mp.get_context())
    # import sys
    # print(sys.path)
    # print(mp.parent_process())
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
        # print('HELLO2')
        # print(engine._engines)
        # print(engine._engine_lock)
        eng = engine.start_matlab("-nodesktop -nodisplay -nojvm")
        
        print('Matlab started')
        eng.clear('all', nargout=0)
        # add the gpfaEngine path
        defaultParams = loadDefaultParams(defParamBase = ".")
        matlabCodePath = Path(defaultParams['matlabCodePath'])
        eng.evalc("addpath(genpath('"+str(matlabCodePath)+"'))")
        
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
    

def saveFiguresToPdf(pdfname=None,analysisDescription = None,figNumsToSave=None):
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
        for fig in figNumsToSave: ## will open an empty extra figure :(
            pdf.savefig( fig )
