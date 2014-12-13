"""Module to execute all important things from project.
"""


import sys
import os, os.path
import shutil
from useful import FEATURES_DIR, GMMS_DIR
import corpus, gmm
import numpy as np
import pickle


# extract the MFCCs from base MIT and put in the correct format
if 'extract-features' in sys.argv:
    if os.path.exists(FEATURES_DIR):
        shutil.rmtree(FEATURES_DIR)

    print('FEATURE EXTRACTION')

    winlen = 0.02
    winstep = 0.01
    corpus.mit_features(winlen, winstep, 13, 0)
    corpus.mit_features(winlen, winstep, 13, 1)
    corpus.mit_features(winlen, winstep, 13, 2)

if 'train-gmms' in sys.argv:
    if os.path.exists(GMMS_DIR):
        shutil.rmtree(GMMS_DIR)
    os.mkdir(GMMS_DIR)

    print('SPEAKER GMMS TRAINING')

    numcep = 13
    numdeltaslist = [0, 1, 2]
    Ms = [8, 16, 32, 64]

    for numdeltas in numdeltaslist:
        gmmpath = '%smit_%d_%d/' % (GMMS_DIR, numcep, numdeltas)
        os.mkdir(gmmpath)
        print('GMM BASE: mit_%d_%d' % (numcep, numdeltas))
        speakers = os.listdir('%smit_%d_%d/enroll_1/' % (FEATURES_DIR, numcep,
                                                         numdeltas))
        speakers.sort()
        numfeats = numcep*(numdeltas + 1)

        for speaker in speakers:
            mfccs = corpus.read_speaker_features(numcep, numdeltas, speaker)
            print('SPEAKER: %s\nmfccs: %s' % (speaker, mfccs.shape))

            for M in Ms:
                print('M = %d' % M)
                oldgmm = gmm.create_gmm(M, numfeats)
                newgmm = gmm.train_gmm(oldgmm, mfccs)
                gmmfile = open('%s%s_%d.gmm' % (gmmpath, speaker, M), 'wb')
                pickle.dump(newgmm, gmmfile)

#if 'train-ubm-gmm' in sys.argv:
#    genders = [None, 'f', 'm']
#    for gender in genders:
#        print(gender)
#        #TODO após completar o módulo 'gmm', criar código para treinar os backgrounds