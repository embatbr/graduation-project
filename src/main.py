#!/usr/bin/python3.4

"""Module to execute all important things from project.
"""


import sys
import os, os.path
import shutil
import time
import pickle
import numpy as np
import pylab as pl
from common import FEATURES_DIR, GMMS_DIR, EXP_VERIFICATION_DIR
import bases, mixtures


commands = sys.argv[1:]

numceps = [13, 19, 26] # 26 is the default number of filters.
delta_orders = [0, 1, 2]
Ms = [8, 16, 32, 64, 128, 256, 512, 1024, 2048]

locations = ['office', 'hallway', 'intersection']
microphones = ['headset', 'internal']


if 'extract-features' in commands:
    if not os.path.exists(FEATURES_DIR):
        os.mkdir(FEATURES_DIR)

    print('FEATURE EXTRACTION')

    winlen = 0.02
    winstep = 0.01
    t = time.time()

    for numcep in numceps:
        for delta_order in delta_orders:
            bases.extract_mit(winlen, winstep, numcep, delta_order)

    t = time.time() - t
    print('Features extracted in %f seconds' % t)


if 'train-ubms' in commands:
    if not os.path.exists(GMMS_DIR):
        os.mkdir(GMMS_DIR)

    print('UBM TRAINING')
    t_tot = time.time()

    for M in Ms:
        print('M = %d' % M)
        for numcep in numceps:
            print('numcep = %d' % numcep)
            for delta_order in delta_orders:
                print('delta_order = %d' % delta_order)
                GMMS_PATH = '%smit_%d_%d/' % (GMMS_DIR, numcep, delta_order)
                if not os.path.exists(GMMS_PATH):
                    os.mkdir(GMMS_PATH)

                featsvec_f = bases.read_mit_background(numcep, delta_order, 'f')
                featsvec_m = bases.read_mit_background(numcep, delta_order, 'm')
                featsvec = np.vstack((featsvec_f, featsvec_m))

                print('UNISEX UBM')
                ubm_unisex = mixtures.GMM('unisex', M, featsvec)
                t = time.time()
                ubm_unisex.train(featsvec)
                t = time.time() - t
                print('Trained in %f seconds' % t)
                GMM_PATH = '%s/unisex_%d.ubm' % (GMMS_PATH, M)
                ubmfile = open(GMM_PATH, 'wb')
                pickle.dump(ubm_unisex, ubmfile)
                ubmfile.close()

                print('GENDER UBM')
                ubm_f = mixtures.GMM('gender', M//2, featsvec_f)
                ubm_m = mixtures.GMM('gender', M//2, featsvec_m)
                t = time.time()
                ubm_f.train(featsvec_f)
                ubm_m.train(featsvec_m)
                t = time.time() - t
                print('Trained in %f seconds' % t)
                # combination
                ubm_gender = ubm_f
                ubm_gender.M = 2*ubm_gender.M
                ubm_gender.weights = np.hstack((ubm_gender.weights, ubm_m.weights))
                ubm_gender.meansvec = np.vstack((ubm_gender.meansvec, ubm_m.meansvec))
                ubm_gender.variancesvec = np.vstack((ubm_gender.variancesvec,
                                                     ubm_m.variancesvec))
                GMM_PATH = '%s/gender_%d.ubm' % (GMMS_PATH, M)
                ubmfile = open(GMM_PATH, 'wb')
                pickle.dump(ubm_gender, ubmfile)
                ubmfile.close()

    t_tot = time.time() - t_tot
    print('Total time: %f seconds' % t_tot)


#TODO SEPARAR POR CONDITIONS
if 'verify' in commands:
    if not os.path.exists(EXP_VERIFICATION_DIR):
        os.mkdir(EXP_VERIFICATION_DIR)

    print('SPEAKER VERIFICATION')
    t_tot = time.time()

    for M in Ms:
        print('M = %d' % M)
        M_DIR = '%sM_%d/' % (EXP_VERIFICATION_DIR, M)
        if not os.path.exists(M_DIR):
            os.mkdir(M_DIR)

        for numcep in numceps:
            print('numcep = %d' % numcep)
            for delta_order in delta_orders:
                print('delta_order = %d' % delta_order)

                # reading UBMs trained using utterances from 'enroll_1'
                PATH = '%smit_%d_%d/' % (UBMS_DIR, numcep, delta_order)
                ubm_unisex_file = open('%sunisex_%d.gmm' % (PATH, M), 'rb')
                ubm_unisex = pickle.load(ubm_unisex_file)
                ubm_gender_file = open('%sgender_%d.gmm' % (PATH, M), 'rb')
                ubm_gender = pickle.load(ubm_gender_file)

                # reading GMMs trained using utterances from 'enroll_1'
                PATH = '%smit_%d_%d/' % (GMMS_DIR, numcep, delta_order)
                filenames = os.listdir(PATH)
                filenames = [filename for filename in filenames
                                      if filename.endswith('%d.gmm' % M)]
                filenames.sort()
                gmms = dict()
                for filename in filenames:
                    GMM_PATH = '%s%s' % (PATH, filename)
                    modelfile = open(GMM_PATH, 'rb')
                    model = pickle.load(modelfile)
                    modelfile.close()
                    gmms[filename[: 3]] = model

                M_DIR_NUMCEP_DELTA_PATH = '%smit_%d_%d/' % (M_DIR, numcep, delta_order)
                if not os.path.exists(M_DIR_NUMCEP_DELTA_PATH):
                    os.mkdir(M_DIR_NUMCEP_DELTA_PATH)

                # log-likelihood ratio test for claimed and imposter speakers
                CLAIMED_SPEAKERS_PATH = '%smit_%d_%d/enroll_2' % (FEATURES_DIR, numcep,
                                                                  delta_order)
                IMPOSTER_SPEAKERS_PATH = '%smit_%d_%d/imposter' % (FEATURES_DIR, numcep,
                                                                   delta_order)
                PATHS = [CLAIMED_SPEAKERS_PATH, IMPOSTER_SPEAKERS_PATH]
                datasets = ['enroll_2', 'imposter']
                for (SPEAKER_PATH, dataset) in zip(PATHS, datasets):
                    speakers = os.listdir(SPEAKER_PATH)
                    speakers.sort()
                    for speaker in speakers:
                        print(speaker)
                        gmm = gmms[speaker[: 3]]
                        speaker_features = bases.read_mit_features_list(numcep, delta_order,
                                                                        dataset, speaker)

                        scores = list()
                        for speaker_feature in speaker_features:
                            log_likeli_gmm = gmm.log_likelihood(speaker_feature)
                            log_likeli_ubm_unisex = ubm_unisex.log_likelihood(speaker_feature)
                            log_likeli_ubm_gender = ubm_gender.log_likelihood(speaker_feature)
                            score_unisex = log_likeli_gmm - log_likeli_ubm_unisex
                            score_gender = log_likeli_gmm - log_likeli_ubm_gender
                            scores.append((score_unisex, score_gender))

                        SCORE_PATH = '%s%s.score' % (M_DIR_NUMCEP_DELTA_PATH, speaker)
                        scorefile = open(SCORE_PATH, 'w')
                        for score in scores:
                            scorefile.write('%f\t%f\n' % (score[0], score[1]))
                        scorefile.close()

    t_tot = time.time() - t_tot
    print('Total time: %f seconds' % t_tot)