from pathlib import Path

def loadDefaultParams(defParamName = "paramDefs.txt", defParamBase="."):
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