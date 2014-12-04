"""Module to execute all important things from project.
"""


import sys
import os.path
import shutil

from useful import FEATURES_DIR
import corpus


# extract the MFCCs from base MIT and put in the correct format
if 'extract-features' in sys.argv:
    if os.path.exists(FEATURES_DIR):
        shutil.rmtree(FEATURES_DIR)

    bases = ['mit']
    functions = [corpus.mit_features]
    print('FEATURE EXTRACTION')

    for (base, func) in zip(bases, functions):
        winlen = 0.02
        winstep = 0.01
        func(winlen, winstep, preemph=0, numcep=13, num_deltas=0)
        func(winlen, winstep, preemph=0, numcep=13, num_deltas=1)
        func(winlen, winstep, preemph=0, numcep=13, num_deltas=2)
        func(winlen, winstep, preemph=0.97, numcep=13, num_deltas=0)
        func(winlen, winstep, preemph=0.97, numcep=13, num_deltas=1)
        func(winlen, winstep, preemph=0.97, numcep=13, num_deltas=2)