#!/usr/bin/python3.4

"""Module to execute all important things from project.
"""


import sys
import os, os.path
import shutil
import time
import pickle
from common import FEATURES_DIR, GMMS_DIR, EXP_IDENTIFICATION_DIR
import bases, mixtures


DEBUG = True

commands = sys.argv[1:]

delta_orders = [0, 1, 2]
M = 8#1024
datasets = ['enroll_1', 'enroll_2', 'imposter']
numcep = 13

#Extract the MFCCs from base MIT and put in the correct format.
if 'extract-features' in commands:
    if os.path.exists(FEATURES_DIR):
        shutil.rmtree(FEATURES_DIR)
    os.mkdir(FEATURES_DIR)

    print('FEATURE EXTRACTION')

    winlen = 0.02
    winstep = 0.01
    t = time.time()
    for delta_order in delta_orders:
        bases.extract_mit(winlen, winstep, numcep, delta_order)
    t = time.time() - t
    print('Features extracted in %f seconds' % t)

if 'train-gmms' in commands:
    if not os.path.exists(GMMS_DIR):
        os.mkdir(GMMS_DIR)

    print('GMM TRAINING\nM = %d' % M)
    t_tot = time.time()

    for delta_order in delta_orders:
        print('delta_order = %d' % delta_order)
        GMMS_PATH = '%smit_%d_%d/' % (GMMS_DIR, numcep, delta_order)
        if not os.path.exists(GMMS_PATH):
            os.mkdir(GMMS_PATH)

        ENROLL_1_PATH = '%smit_%d_%d/enroll_1/' % (FEATURES_DIR, numcep, delta_order)
        speakers = os.listdir(ENROLL_1_PATH)
        speakers.sort()

        for speaker in speakers:
            featsvec = bases.read_mit_speaker_features(numcep, delta_order,
                                                       'enroll_1', speaker)
            if DEBUG: print('%s: %s' % (speaker, featsvec.shape))

            gmm = mixtures.GMM(M, featsvec)
            if DEBUG: t = time.time()
            gmm.train(featsvec)
            if DEBUG: t = time.time() - t
            if DEBUG: print('GMM trained in %f seconds' % t)

            GMM_PATH = '%s/%s_%d.gmm' % (GMMS_PATH, speaker, M)
            gmmfile = open(GMM_PATH, 'wb')
            pickle.dump(gmm, gmmfile)
            gmmfile.close()

    t_tot = time.time() - t_tot
    print('Total time: %f seconds' % t_tot)


if 'identify' in commands:
    if os.path.exists(EXP_IDENTIFICATION_DIR):
        shutil.rmtree(EXP_IDENTIFICATION_DIR)
    os.mkdir(EXP_IDENTIFICATION_DIR)

    print('SPEAKER IDENTIFICATION')

    if DEBUG: print('M = %d' % M)
    for delta_order in delta_orders:
        if DEBUG: print('delta_order = %d' % delta_order)
        # opening experiment file
        EXP_SET_PATH = '%smit_%d_%d.exp' % (EXP_IDENTIFICATION_DIR, numcep, delta_order)
        expfile = open(EXP_SET_PATH, 'a')

        # reading GMMs from 'enroll_1' to use as base
        ENROLL_1_PATH = '%smit_%d_%d/' % (GMMS_DIR, numcep, delta_order)
        gmmfilenames = os.listdir(ENROLL_1_PATH)
        gmmfilenames.sort()
        gmms_enroll_1 = list()
        for gmmfilename in gmmfilenames:
            GMM_PATH = '%s%s' % (ENROLL_1_PATH, gmmfilename)
            gmmfile = open(GMM_PATH, 'rb')
            gmm = pickle.load(gmmfile)
            gmmfile.close()
            gmms_enroll_1.append(gmm)

        # reading features from 'enroll_2'
        ENROLL_2_PATH = '%smit_%d_%d/enroll_2/' % (FEATURES_DIR, numcep, delta_order)
        speakers = os.listdir(ENROLL_2_PATH)
        speakers.sort()
        for speaker in speakers:
            if DEBUG: print(speaker)
            SPEAKER_PATH = '%s%s' % (ENROLL_2_PATH, speaker)
            features = os.listdir(SPEAKER_PATH)
            features.sort()

            hits = 0
            for feature in features:
                featsvec = bases.read_mit_features(numcep, delta_order, 'enroll_2',
                                                   speaker, int(feature[:2]))
                probs = [gmm_enroll_1.log_likelihood(featsvec) for gmm_enroll_1
                                                               in gmms_enroll_1]
                indentified = gmmfilenames[probs.index(max(probs))][:3]
                if indentified == speaker:
                    hits = hits + 1

            hits = hits / len(features)
            if DEBUG: print('hits =', hits)
            expfile.write('%d %s %3.2f\n' % (M, speaker, 100*hits))


if 'train-ubms' in commands:
    if not os.path.exists(GMMS_DIR):
        os.mkdir(GMMS_DIR)

    print('UBM TRAINING')

    t_tot = time.time()

    genders = ['f', 'm']
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

    t_tot = time.time() - t_tot
    print('Total time: %f seconds' % t_tot)


if 'adap-gmms-from-ubm' in commands:
    pass


if 'verify' in commands:
    pass