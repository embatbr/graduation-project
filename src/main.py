"""Module to execute all important things from project.
"""


import sys

from useful import BASES_DIR
import corpus


# extract the MFCCs from base MIT and put in the correct format
if 'mit_features' in sys.argv:
    winlen = 0.02
    winstep = 0.01
    preemph = 0.97
    corpus.mit_features(winlen, winstep, preemph=preemph)