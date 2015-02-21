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


commands = sys.argv[1:]


numdeltaslist = [0]#[0, 1, 2]

#Extract the MFCCs from base MIT and put in the correct format.
if 'extract-features' in commands:
    if os.path.exists(FEATURES_DIR):
        shutil.rmtree(FEATURES_DIR)
    os.mkdir(FEATURES_DIR)

    print('FEATURE EXTRACTION')

    winlen = 0.02
    winstep = 0.01
    numcep = 13
    for numdeltas in numdeltaslist:
        corpus.mit_features(winlen, winstep, numcep, numdeltas)


if 'train-gmms' in commands:
    if not os.path.exists(GMMS_DIR):
        os.mkdir(GMMS_DIR)

    print('GMM TRAINING')

    numcep = 13
    datasets = ['enroll_1', 'enroll_2', 'imposter']
    M = 8

    for numdeltas in numdeltaslist:
        print('numdeltas = %d' % numdeltas)
        GMMSPATH = '%smit_%d_%d/' % (GMMS_DIR, numcep, numdeltas)
        if not os.path.exists(GMMSPATH):
            os.mkdir(GMMSPATH)
        numfeats = numcep*(numdeltas + 1)

        for dataset in datasets:
            DATASETPATH = '%smit_%d_%d/%s/' % (FEATURES_DIR, numcep, numdeltas,
                                               dataset)
            speakers = os.listdir(DATASETPATH)
            speakers.sort()

            for speaker in speakers:
                mfccs = corpus.read_speaker_features(numcep, numdeltas, speaker)
                print(mfccs.shape)

                gmm = mixtures.GMM(M, mfccs)
                t = time.time()
                gmm.train(mfccs.T)
                t = time.time() - t
                print('GMM trained in %f seconds' % t)

                gmmpath = '%s/%s_%d.gmm' % (GMMSPATH, speaker, M)
                gmmfile = open(gmmpath, 'wb')
                pickle.dump(gmm, gmmfile)
                gmmfile.close()

            print()



if 'train-ubm' in commands:
    if not os.path.exists(GMMS_DIR):
        os.mkdir(GMMS_DIR)

    print('UBM TRAINING')

    numcep = 13
    genders = ['f', 'm']
    M = 1024

    for numdeltas in numdeltaslist:
        print('numdeltas = %d' % numdeltas)
        GMMSPATH = '%smit_%d_%d/' % (GMMS_DIR, numcep, numdeltas)
        if not os.path.exists(GMMSPATH):
            os.mkdir(GMMSPATH)
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

            ubmpath = '%s/ubm_%s_%d.gmm' % (GMMSPATH, gender, M)
            ubmfile = open(ubmpath, 'wb')
            pickle.dump(ubm, ubmfile)
            ubmfile.close()

        print()


if 'adap-gmms-from-ubm' in commands:
    pass