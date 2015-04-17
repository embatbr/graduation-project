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
from common import FEATURES_DIR, UBMS_DIR, GMMS_DIR
import bases, mixtures


commands = sys.argv[1 : ]

numceps = 19 # 26 is the default number of filters.
delta_orders = [0, 1, 2]
Ms = [8, 16, 32, 64, 128] # from 128, the EmptyClusterError starts to occur
configurations = {'office': ('01', '19'), 'hallway': ('21', '39'),
                  'intersection': ('41', '59'), 'all': ('01', '59')}


if 'extract-features' in commands:
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


if 'train-ubms' in commands:
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
                print(environment)
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


if 'adapt-gmms' in commands:
    if not os.path.exists(GMMS_DIR):
        os.mkdir(GMMS_DIR)

    print('Adapting GMMs from UBM\nnumceps = %d' % numceps)
    t_tot = time.time()

    for M in Ms:
        print('M = %d' % M)
        for delta_order in delta_orders:
            print('delta_order = %d' % delta_order)
            ENROLL_1_PATH = '%smit_%d_%d/enroll_1/' % (FEATURES_DIR, numceps, delta_order)
            speakers = os.listdir(ENROLL_1_PATH)
            speakers.sort()

            UBMS_PATH = '%smit_%d_%d/' % (UBMS_DIR, numceps, delta_order)
            GMMS_PATH = '%smit_%d_%d/' % (GMMS_DIR, numceps, delta_order)
            if not os.path.exists(GMMS_PATH):
                os.mkdir(GMMS_PATH)

            for environment in configurations.keys():
                print(environment)
                ubm_file = open('%s%s_%d.ubm' % (UBMS_PATH, environment, M), 'rb')
                ubm = pickle.load(ubm_file)

                downlim = configurations[environment][0]
                uplim = configurations[environment][1]
                for speaker in speakers:
                    print(speaker)
                    featsvec = bases.read_speaker(numceps, delta_order, 'enroll_1',
                                                  speaker, downlim, uplim)
                    gmm = ubm.clone('%s_%s_%d' % (speaker, environment, M))
                    gmm.adapt_gmm(featsvec)

                    GMM_PATH = '%s%s.gmm' % (GMMS_PATH, gmm.name)
                    gmmfile = open(GMM_PATH, 'wb')
                    pickle.dump(gmm, gmmfile)
                    gmmfile.close()


    t_tot = time.time() - t_tot
    print('Total time: %f seconds' % t_tot)


#TODO rodar quando terminar 'train-ubms' para corrigir os nomes
if 'correct-ubms-names' in commands:
    for M in Ms:
        for delta_order in delta_orders:
            UBMS_PATH = '%smit_%d_%d/' % (UBMS_DIR, numceps, delta_order)
            for environment in configurations.keys():
                ubm_file = open('%s%s_%d.ubm' % (UBMS_PATH, environment, M), 'rb')
                ubm = pickle.load(ubm_file)
                ubm.name = '%s_%d' % (environment, M)

                UBM_PATH = '%s%s.ubm' % (UBMS_PATH, ubm.name)
                ubmfile = open(UBM_PATH, 'wb')
                pickle.dump(ubm, ubmfile)
                ubmfile.close()