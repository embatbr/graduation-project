#!/usr/bin/python3.4

"""Module to execute all important things from project.
"""


import sys
import os, os.path
import shutil
import time
import pickle
import json
import numpy as np
import pylab as pl
from common import FEATURES_DIR, UBMS_DIR, GMMS_DIR, VERIFY_DIR, MIN_VARIANCE
import bases, mixtures


command = sys.argv[1]
parameters = sys.argv[2 : ]

numceps = 19 # 26 is the default number of filters.
delta_orders = [0, 1, 2]
Ms = [8, 16, 32, 64, 128] # from 128, the EmptyClusterError starts to occur
configurations = {'office': ('01', '19'), 'hallway': ('21', '39'),
                  'intersection': ('41', '59'), 'all': ('01', '59')}


if command == 'extract-features':
    if not os.path.exists(FEATURES_DIR):
        os.mkdir(FEATURES_DIR)

    print('FEATURE EXTRACTION\nnumceps = %d' % numceps)

    winlen = 0.02
    winstep = 0.01
    t = time.time()

    for delta_order in delta_orders:
        bases.extract(winlen, winstep, numceps, delta_order)

    t = time.time() - t
    print('Features extracted in %f seconds' % t)


if command == 'train-ubms':
    if not os.path.exists(UBMS_DIR):
        os.mkdir(UBMS_DIR)

    print('UBM TRAINING\nnumceps = %d' % numceps)
    t_tot = time.time()

    for M in Ms:
        print('M = %d' % M)
        for delta_order in delta_orders:
            print('delta_order = %d' % delta_order)
            UBMS_PATH = '%smit_%d_%d/' % (UBMS_DIR, numceps, delta_order)
            if not os.path.exists(UBMS_PATH):
                os.mkdir(UBMS_PATH)

            for environment in configurations.keys():
                print(environment.upper())
                downlim = configurations[environment][0]
                uplim = configurations[environment][1]
                featsvec_f = bases.read_background(numceps, delta_order, 'f',
                                                   downlim, uplim)
                featsvec_m = bases.read_background(numceps, delta_order, 'm',
                                                   downlim, uplim)

                # training
                D = numceps * (1 + delta_order)
                print('D = %d' % D)
                ubm_f = mixtures.GMM('f', M // 2, D)
                ubm_m = mixtures.GMM('m', M // 2, D)
                for (ubm, featsvec, gender) in zip([ubm_f, ubm_m], [featsvec_f, featsvec_m],
                                                   ['female', 'male']):
                    while(True):
                        try:
                            print('Training %s GMM' % gender)
                            ubm.train(featsvec)
                            break
                        except mixtures.EmptyClusterError as e:
                            print('%s\nrebooting %s GMM' % (e.msg, gender))

                # combination
                ubm = ubm_f
                ubm.merge(ubm_m, '%s_%d' % (environment, M))

                UBM_PATH = '%s%s.ubm' % (UBMS_PATH, ubm.name)
                ubmfile = open(UBM_PATH, 'wb')
                pickle.dump(ubm, ubmfile)
                ubmfile.close()

    t_tot = time.time() - t_tot
    print('Total time: %f seconds' % t_tot)


if command == 'adapt-gmms':
    if not os.path.exists(GMMS_DIR):
        os.mkdir(GMMS_DIR)
    adaptations = parameters[0]
    adapt_gmmc_dir = '%sadapt_%s/' % (GMMS_DIR, adaptations)
    if not os.path.exists(adapt_gmmc_dir):
        os.mkdir(adapt_gmmc_dir)

    print('Adapting GMMs from UBM\nnumceps = %d' % numceps)
    print('adaptations: %s' % adaptations)
    t_tot = time.time()

    for M in Ms:
        print('M = %d' % M)
        for delta_order in delta_orders:
            print('delta_order = %d' % delta_order)
            ENROLL_1_PATH = '%smit_%d_%d/enroll_1/' % (FEATURES_DIR, numceps, delta_order)
            speakers = os.listdir(ENROLL_1_PATH)
            speakers.sort()

            UBMS_PATH = '%smit_%d_%d/' % (UBMS_DIR, numceps, delta_order)
            GMMS_PATH = '%smit_%d_%d/' % (adapt_gmmc_dir, numceps, delta_order)
            if not os.path.exists(GMMS_PATH):
                os.mkdir(GMMS_PATH)

            for environment in configurations.keys():
                print(environment.upper())
                ubmfile = open('%s%s_%d.ubm' % (UBMS_PATH, environment, M), 'rb')
                ubm = pickle.load(ubmfile)
                ubmfile.close()

                downlim = configurations[environment][0]
                uplim = configurations[environment][1]
                for speaker in speakers:
                    print(speaker)
                    featsvec = bases.read_speaker(numceps, delta_order, 'enroll_1',
                                                  speaker, downlim, uplim)
                    gmm = ubm.clone('%s_%s_%d' % (speaker, environment, M))
                    gmm.adapt_gmm(featsvec, adaptations=parameters)

                    GMM_PATH = '%s%s.gmm' % (GMMS_PATH, gmm.name)
                    gmmfile = open(GMM_PATH, 'wb')
                    pickle.dump(gmm, gmmfile)
                    gmmfile.close()


    t_tot = time.time() - t_tot
    print('Total time: %f seconds' % t_tot)


if command == 'verify':
    if not os.path.exists(VERIFY_DIR):
        os.mkdir(VERIFY_DIR)
    adaptations = parameters[0]
    verify_dir = '%sadapt_%s/' % (VERIFY_DIR, adaptations)
    if not os.path.exists(verify_dir):
        os.mkdir(verify_dir)
    adapt_gmmc_dir = '%sadapt_%s/' % (GMMS_DIR, adaptations)

    print('Verification\nnumceps = %d' % numceps)
    print('adaptations: %s' % adaptations)
    t_tot = time.time()

    for M in Ms:
        print('M = %d' % M)
        for delta_order in delta_orders:
            print('delta_order = %d' % delta_order)
            UBMS_PATH = '%smit_%d_%d/' % (UBMS_DIR, numceps, delta_order)
            GMMS_PATH = '%smit_%d_%d/' % (adapt_gmmc_dir, numceps, delta_order)
            EXP_PATH = '%smit_%d_%d/' % (verify_dir, numceps, delta_order)
            if not os.path.exists(EXP_PATH):
                os.mkdir(EXP_PATH)

            all_gmm_filenames = os.listdir(GMMS_PATH)
            all_gmm_filenames = [gmm_filename for gmm_filename in all_gmm_filenames
                                 if gmm_filename.endswith('_%d.gmm' % M)]
            all_gmm_filenames.sort()

            expdict = dict()
            for environment in configurations.keys():
                print(environment.upper())
                ubmfile = open('%s%s_%d.ubm' % (UBMS_PATH, environment, M), 'rb')
                ubm = pickle.load(ubmfile)
                ubmfile.close()

                ubm_key = 'UBM %s' % ubm.name.split('_')[0]
                expdict[ubm_key] = dict()
                for env in configurations.keys():
                    env_key = 'SCORES %s' % env
                    expdict[ubm_key][env_key] = dict()
                    expdict[ubm_key][env_key]['enrolled'] = list()
                    expdict[ubm_key][env_key]['imposter'] = list()

                gmm_filenames = [gmm_filename for gmm_filename in all_gmm_filenames
                                 if gmm_filename.endswith('_%s_%d.gmm' % (environment, M))]

                for gmm_filename in gmm_filenames:
                    print(gmm_filename)
                    gmmfile = open('%s%s' % (GMMS_PATH, gmm_filename), 'rb')
                    gmm = pickle.load(gmmfile)
                    gmmfile.close()

                    enrolled = gmm.name.split('_')[0]
                    gender = enrolled[0]
                    num_imposters = 17 if gender == 'f' else 23
                    speakers = [enrolled] + ['%s%02d_i' % (gender, i) for i in range(num_imposters)]

                    for i in range(len(speakers)):
                        speaker = speakers[i]
                        dataset = 'imposter' if i > 0 else 'enroll_2'
                        dataset_key = 'imposter' if i > 0 else 'enrolled'
                        featslist = bases.read_features_list(numceps, delta_order,
                                                             dataset, speaker)

                        for i in range(len(featslist)):
                            feats = featslist[i]
                            log_likeli_gmm = gmm.log_likelihood(feats)
                            log_likeli_ubm = ubm.log_likelihood(feats)
                            score = log_likeli_gmm - log_likeli_ubm

                            expdict[ubm_key]['SCORES all'][dataset_key].append(score)
                            if i < 18:
                                expdict[ubm_key]['SCORES office'][dataset_key].append(score)
                            if 18 <= i < 36:
                                expdict[ubm_key]['SCORES hallway'][dataset_key].append(score)
                            if 36 <= i < 54:
                                expdict[ubm_key]['SCORES intersection'][dataset_key].append(score)

            EXP_FILE_PATH = '%sM_%d.json' % (EXP_PATH, M)
            with open(EXP_FILE_PATH, 'w') as expfile:
                json.dump(expdict, expfile, indent=4, sort_keys=True)

    t_tot = time.time() - t_tot
    print('Total time: %f seconds' % t_tot)


# Códigos de correção para merdas que fiz anteriormente e demorariam muito tempo
# para refazer

if command == 'check':
    for M in Ms:
        for delta_order in delta_orders:
            for directory in [UBMS_DIR, GMMS_DIR]:
                PATH = '%smit_%d_%d/' % (directory, numceps, delta_order)
                filenames = os.listdir(PATH)
                filenames.sort()
                for filename in filenames:
                    gmmfile = open('%s%s' % (PATH, filename), 'rb')
                    gmm = pickle.load(gmmfile)
                    gmmfile.close()

                    if np.min(gmm.weights) <= 0 or np.min(gmm.weights) >= 1:
                        print(gmm.name)
                        print('Wrong weights')
                        print(gmm.weights)
                    if np.min(gmm.variancesvec) < MIN_VARIANCE:
                        print(gmm.name)
                        print('Wrong variancesvec')
                        print(gmm.variancesvec)


if command == 'correct-ubms-names':
    for M in Ms:
        for delta_order in delta_orders:
            UBMS_PATH = '%smit_%d_%d/' % (UBMS_DIR, numceps, delta_order)
            for environment in configurations.keys():
                ubmfile = open('%s%s_%d.ubm' % (UBMS_PATH, environment, M), 'rb')
                ubm = pickle.load(ubmfile)
                ubmfile.close()
                ubm.name = '%s_%d' % (environment, M)

                UBM_PATH = '%s%s.ubm' % (UBMS_PATH, ubm.name)
                ubmfile = open(UBM_PATH, 'wb')
                pickle.dump(ubm, ubmfile)
                ubmfile.close()

                print(UBM_PATH, ubm.name)