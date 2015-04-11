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

numceps = [19, 26] # 26 is the default number of filters.
delta_orders = [0, 1, 2]
Ms = [8, 16, 32, 64, 128, 256]
noisetypes = [('office', '01', '19'), ('hallway', '21', '39'), ('intersection', '41', '59'),
              ('all', '01', '59')]


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

                for (noise, downlim, uplim) in noisetypes:
                    print(noise)
                    featsvec_f = bases.read_mit_background(numcep, delta_order,
                                                           'f', downlim, uplim)
                    featsvec_m = bases.read_mit_background(numcep, delta_order,
                                                           'm', downlim, uplim)

                    # training
                    ubm_f = mixtures.GMM('f', M // 2, featsvec_f.shape[1])
                    ubm_m = mixtures.GMM('m', M // 2, featsvec_m.shape[1])
                    while(True):
                        try:
                            print('Training female GMM')
                            ubm_f.train(featsvec_f)
                            break
                        except mixtures.EmptyClusterError as e:
                            print('%s\nrebooting' % e.msg)
                    while(True):
                        try:
                            print('Training male GMM')
                            ubm_m.train(featsvec_m)
                            break
                        except mixtures.EmptyClusterError as e:
                            print('%s\nrebooting' % e.msg)

                    # combination
                    ubm = ubm_f
                    ubm.name = noise
                    ubm.M = 2 * M
                    ubm.weights = np.hstack((ubm.weights, ubm_m.weights))
                    ubm.meansvec = np.vstack((ubm.meansvec, ubm_m.meansvec))
                    ubm.variancesvec = np.vstack((ubm.variancesvec, ubm_m.variancesvec))

                    GMM_PATH = '%s/%s_%d.ubm' % (GMMS_PATH, noise, M)
                    ubmfile = open(GMM_PATH, 'wb')
                    pickle.dump(ubm, ubmfile)
                    ubmfile.close()

    t_tot = time.time() - t_tot
    print('Total time: %f seconds' % t_tot)


# TODO adaptar os GMMs a partir dos UBMs
if 'adapt-gmms' in commands:
    pass


# TODO colocar os resultados em formato json
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