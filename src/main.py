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


commands = sys.argv[1:]

delta_orders = [0, 1, 2]
M = 64
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
            print('GMM %s: %s' % (speaker, featsvec.shape))

            gmm = mixtures.GMM(M, featsvec)
            featsvec_utt = bases.read_mit_features(numcep, delta_order, 'enroll_1', 'f00', 1)
            untrained_log_likelihood = gmm.log_likelihood(featsvec_utt)
            print('log_likelihood = %f' % untrained_log_likelihood)

            t = time.time()
            gmm.train(featsvec)
            t = time.time() - t
            print('Trained in %f seconds' % t)
            trained_log_likelihood = gmm.log_likelihood(featsvec_utt)
            print('log_likelihood = %f' % trained_log_likelihood)

            increase = 1 - (trained_log_likelihood / untrained_log_likelihood)
            print('increase = %2.2f%%' % (increase*100))

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

    print('SPEAKER IDENTIFICATION\nM = %d' % M)

    for delta_order in delta_orders:
        print('delta_order = %d' % delta_order)

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
        hitslist = list()
        for speaker in speakers:
            print(speaker)
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
            hitslist.append(hits)
            print('hits =', (hits*100))

        MFCC_DIR = '%smit_%d_%d/' % (EXP_IDENTIFICATION_DIR, numcep, delta_order)
        if not os.path.exists(MFCC_DIR):
            os.mkdir(MFCC_DIR)

        EXP_SET_PATH = '%sM_%d.exp' % (MFCC_DIR, M)
        expfile = open(EXP_SET_PATH, 'w')
        for (speaker, hits) in zip(speakers, hitslist):
            expfile.write('%s %3.2f\n' % (speaker, 100*hits))


if 'train-ubms' in commands:
    if not os.path.exists(GMMS_DIR):
        os.mkdir(GMMS_DIR)

    print('UBM TRAINING\nM = %d' % M)

    t_tot = time.time()

    genders = ['f', 'm']
    for delta_order in delta_orders:
        print('delta_order = %d' % delta_order)
        GMMS_PATH = '%smit_%d_%d/' % (GMMS_DIR, numcep, delta_order)
        if not os.path.exists(GMMS_PATH):
            os.mkdir(GMMS_PATH)

        for gender in genders:
            featsvec = bases.read_mit_background_features(numcep, delta_order,
                                                          gender)
            print('%s: %s' % (gender, featsvec.shape))

            ubm = mixtures.GMM(M, featsvec)
            t = time.time()
            ubm.train(featsvec)
            t = time.time() - t
            print('UBM trained in %f seconds' % t)

            GMM_PATH = '%s/ubm_%s_%d.gmm' % (GMMS_PATH, gender, M)
            ubmfile = open(GMM_PATH, 'wb')
            pickle.dump(ubm, ubmfile)
            ubmfile.close()

    t_tot = time.time() - t_tot
    print('Total time: %f seconds' % t_tot)


if 'adap-gmms-from-ubm' in commands:
    #TODO gmm files must ba named as 'adap_<speaker>_M.gmm', where speaker = 'fxx' or 'mxx'
    pass


if 'verify' in commands:
    pass