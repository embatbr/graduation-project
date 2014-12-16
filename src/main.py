#!/usr/bin/python3.4

"""Module to execute all important things from project.
"""


import sys
import os, os.path
import shutil
from useful import FEATURES_DIR, GMMS_DIR, IDENTIFY_DIR
import corpus, mixtures
import numpy as np
import pickle
import time


command = sys.argv[1]

#Extract the MFCCs from base MIT and put in the correct format.
if command == 'extract-features':
    if os.path.exists(FEATURES_DIR):
        shutil.rmtree(FEATURES_DIR)
    os.mkdir(FEATURES_DIR)

    print('FEATURE EXTRACTION')

    winlen = 0.02
    winstep = 0.01
    numcep = 13
    numdeltaslist =[0, 1, 2]
    for numdeltas in numdeltaslist:
        corpus.mit_features(winlen, winstep, numcep, numdeltas)


if command == 'train-ubm':
    if os.path.exists(GMMS_DIR):
        shutil.rmtree(GMMS_DIR)
    os.mkdir(GMMS_DIR)

    print('UBM TRAINING')

    numcep = 13
    numdeltaslist =[0, 1, 2]
    genders = ['f', 'm']
    M = 1024

    for numdeltas in numdeltaslist:
        print('numdeltas = %d' % numdeltas)
        gmmspath = '%smit_%d_%d/' % (GMMS_DIR, numcep, numdeltas)
        os.mkdir(gmmspath)
        numfeats = numcep*(numdeltas + 1)

        for gender in genders:
            print('gender = %s' % gender)
            mfccs = corpus.read_background_features(numcep, numdeltas, gender, False)
            print('mfccs.shape = %s\nUBM (%s) created, M = %d' % (str(mfccs.shape),
                                                                  gender, M))
            ubm = mixtures.GMM(M, mfccs.T)
            t = time.time()
            ubm.train(mfccs)
            t = time.time() - t
            print('UBM trained in %f seconds' % t)

            ubmpath = '%s/ubm_%s_%d.gmm' % (gmmspath, gender, M)
            ubmfile = open(ubmpath, 'wb')
            pickle.dump(ubm, ubmfile)
            ubmfile.close()

        print()