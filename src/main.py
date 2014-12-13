"""Module to execute all important things from project.
"""


import sys
import os, os.path
import shutil
from useful import FEATURES_DIR, GMMS_DIR, IDENTIFY_DIR
import corpus, mixtures
import numpy as np
import pickle


command = sys.argv[1]
params = sys.argv[2:]

# extract the MFCCs from base MIT and put in the correct format
if option == 'extract-features':
    if os.path.exists(FEATURES_DIR):
        shutil.rmtree(FEATURES_DIR)

    print('FEATURE EXTRACTION')

    winlen = 0.02
    winstep = 0.01
    corpus.mit_features(winlen, winstep, 13, 0)
    corpus.mit_features(winlen, winstep, 13, 1)
    corpus.mit_features(winlen, winstep, 13, 2)

elif option == 'train-gmms':
    if os.path.exists(GMMS_DIR):
        shutil.rmtree(GMMS_DIR)
    os.mkdir(GMMS_DIR)

    (numcep, numdeltas, M) = (int(params[0]), int(params[1]), int(params[2]))
    print('SPEAKER GMMS TRAINING\nGMM BASE: mit_%d_%d\nM = %d' % (numcep, numdeltas, M))

    gmmspath = '%smit_%d_%d/' % (GMMS_DIR, numcep, numdeltas)
    os.mkdir(gmmspath)
    speakers = os.listdir('%smit_%d_%d/enroll_1/' % (FEATURES_DIR, numcep,
                                                     numdeltas))
    speakers.sort()
    numfeats = numcep*(numdeltas + 1)

    for speaker in speakers:
        mfccs = corpus.read_speaker_features(numcep, numdeltas, speaker)
        print('SPEAKER: %s\nmfccs: %s' % (speaker, mfccs.shape))
        oldgmm = mixtures.create_gmm(M, numfeats)
        newgmm = mixtures.train_gmm(oldgmm, mfccs)
        gmmfile = open('%s%s_%d.gmm' % (gmmspath, speaker, M), 'wb')
        pickle.dump(newgmm, gmmfile)