#!/usr/bin/python3.4

"""Module to execute all important things from project.
"""


import sys
import os, os.path
import shutil
import time
import pickle
from common import FEATURES_DIR, GMMS_DIR
import bases, mixtures


DEBUG = True
CHECK_ERROR = False


commands = sys.argv[1:]

delta_orders = [0, 1, 2]
Ms = [8, 16, 32, 64, 128, 256, 512, 1024]
datasets = ['enroll_1', 'enroll_2', 'imposter']
numcep = 13

#Extract the MFCCs from base MIT and put in the correct format.
if 'extract-features' in commands:
    if os.path.exists(FEATURES_DIR):
        shutil.rmtree(FEATURES_DIR)
    os.mkdir(FEATURES_DIR)

    if DEBUG: print('FEATURE EXTRACTION')

    winlen = 0.02
    winstep = 0.01
    for delta_order in delta_orders:
        bases.extract_mit(winlen, winstep, numcep, delta_order)


if 'train-gmms' in commands:
    if not os.path.exists(GMMS_DIR):
        os.mkdir(GMMS_DIR)

    if DEBUG: print('GMM TRAINING')
    t_tot = time.time()

    for M in Ms:
        if DEBUG: print('M = %d' % M)
        for delta_order in delta_orders:
            if DEBUG: print('delta_order = %d' % delta_order)
            GMMS_PATH = '%smit_%d_%d/' % (GMMS_DIR, numcep, delta_order)
            if not os.path.exists(GMMS_PATH):
                os.mkdir(GMMS_PATH)

            for dataset in datasets:
                if DEBUG: print(dataset)
                GMMS_DATASET_PATH = '%s%s/' % (GMMS_PATH, dataset)
                if not os.path.exists(GMMS_DATASET_PATH):
                    os.mkdir(GMMS_DATASET_PATH)

                DATASET_PATH = '%smit_%d_%d/%s/' % (FEATURES_DIR, numcep, delta_order,
                                                    dataset)
                speakers = os.listdir(DATASET_PATH)
                speakers.sort()

                for speaker in speakers:
                    if (DEBUG and CHECK_ERROR): (delta_order, dataset, speaker) = (1, 'enroll_1', 'f21')

                    featsvec = bases.read_mit_speaker_features(numcep, delta_order,
                                                               dataset, speaker)
                    if DEBUG: print('%s: %s' % (speaker, featsvec.shape))

                    gmm = mixtures.GMM(M, featsvec)
                    if DEBUG: t = time.time()
                    gmm.train(featsvec)
                    if DEBUG: t = time.time() - t
                    if DEBUG: print('GMM trained in %f seconds' % t)

                    GMM_PATH = '%s/%s_%d.gmm' % (GMMS_DATASET_PATH, speaker, M)
                    gmmfile = open(GMM_PATH, 'wb')
                    pickle.dump(gmm, gmmfile)
                    gmmfile.close()

                    if (DEBUG and CHECK_ERROR): sys.exit()

    t_tot = time.time() - t_tot
    print('Total time: %f seconds' % t_tot)


if 'verify' in commands:
    pass


if 'train-ubms' in commands:
    if not os.path.exists(GMMS_DIR):
        os.mkdir(GMMS_DIR)

    print('UBM TRAINING')

    genders = ['f', 'm']
    for M in Ms:
        if DEBUG: print('M = %d' % M)
        for delta_order in delta_orders:
            if DEBUG: print('delta_order = %d' % delta_order)
            GMMS_PATH = '%smit_%d_%d/' % (GMMS_DIR, numcep, delta_order)
            if not os.path.exists(GMMS_PATH):
                os.mkdir(GMMS_PATH)

            for gender in genders:
                featsvec = bases.read_mit_background_features(numcep, delta_order,
                                                              gender)
                if DEBUG: print('%s: %s' % (gender, featsvec.shape))

                ubm = mixtures.GMM(M, featsvec)
                if DEBUG: t = time.time()
                ubm.train(featsvec)
                if DEBUG: t = time.time() - t
                if DEBUG: print('UBM trained in %f seconds' % t)

                GMM_PATH = '%s/%s_%d.ubm' % (GMMS_PATH, gender, M)
                ubmfile = open(GMM_PATH, 'wb')
                pickle.dump(ubm, ubmfile)
                ubmfile.close()


if 'adap-gmms-from-ubm' in commands:
    pass


if 'verify' in commands:
    pass