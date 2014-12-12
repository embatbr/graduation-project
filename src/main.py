"""Module to execute all important things from project.
"""


import sys
import os.path
import shutil

from useful import FEATURES_DIR
import corpus


# extract the MFCCs from base MIT and put in the correct format
if sys.argv[1] == 'extract-features':
    if os.path.exists(FEATURES_DIR):
        shutil.rmtree(FEATURES_DIR)

    print('FEATURE EXTRACTION')

    winlen = 0.02
    winstep = 0.01
    corpus.mit_features(winlen, winstep, 13, 0)
    corpus.mit_features(winlen, winstep, 13, 1)
    corpus.mit_features(winlen, winstep, 13, 2)

elif sys.argv[1] == 'train-gmms':
    pass

elif sys.argv[1] == 'train-ubm-gmm':
    genders = [None, 'f', 'm']
    for gender in genders:
        print(gender)
        #TODO após completar o módulo 'gmm', criar código para treinar os backgrounds