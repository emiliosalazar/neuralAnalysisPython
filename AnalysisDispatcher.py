"""
This is basically a method that is called to dispatch given analyses--so that
they're called from the top level
"""
import sys
import getopt

# NOTE DEPRECATED: I'm commenting out the imports so it errors in case we ever
# do use it and then I can go back and make sure it's up to scruff
# we're importing all valid analyses here; yeah, it's an extra step, but it's
# unavoidable really, and in some sense lets me be intentional about adding new
# analyses
#from analyses.CrossareaMatchedCovarianceComparison_vAct import crossareaMatchedCovarianceComparison
#from analyses.V4Stim1sDimensionality import v4Stim1sDimensionality
#from analyses.SweepOfExtractionParams import sweepOfExtractionParams

def analysisDispatcher(func, *funcArgs, **funcKeyArgs):
    func(funcArgs, funcKeyArgs)


if __name__ == '__main__':
    try:
        funcRun = eval(sys.argv[1])
    except IndexError:
        raise Error("Call me with a function, dduuuudddeee!!")

    funcRun()
