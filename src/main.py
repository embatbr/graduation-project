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

numceps = 19 # 26 is the default number of filters.
delta_orders = [0, 1, 2]
Ms = [8, 16, 32, 64, 128, 256]
noisetypes = [('office', '01', '19'), ('hallway', '21', '39'), ('intersection', '41', '59'),
              ('all', '01', '59')]


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
    if not os.path.exists(GMMS_DIR):
        os.mkdir(GMMS_DIR)

    print('UBM TRAINING\nnumceps = %d' % numceps)
    t_tot = time.time()

    for M in Ms:
        print('M = %d' % M)
        for delta_order in delta_orders:
            print('delta_order = %d' % delta_order)
            GMMS_PATH = '%smit_%d_%d/' % (GMMS_DIR, numceps, delta_order)
            if not os.path.exists(GMMS_PATH):
                os.mkdir(GMMS_PATH)

            for (environment, downlim, uplim) in noisetypes:
                print(environment)
                featsvec_f = bases.read_background(numceps, delta_order, 'f',
                                                   downlim, uplim)
                featsvec_m = bases.read_background(numceps, delta_order, 'm',
                                                   downlim, uplim)

                # training
                ubm_f = mixtures.GMM('f', M // 2, numceps)
                ubm_m = mixtures.GMM('m', M // 2, numceps)
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
                ubm.merge(ubm_m, environment)

                GMM_PATH = '%s/%s_%d.ubm' % (GMMS_PATH, environment, M)
                ubmfile = open(GMM_PATH, 'wb')
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
            ENROLL_1_PATH = '%smit_%d_%d/enroll_1/' % (FEATURES_DIR, numceps,
                                                       delta_order)
            GMMS_PATH = '%smit_%d_%d/' % (GMMS_DIR, numceps, delta_order)
            speakers = os.listdir(ENROLL_1_PATH)
            speakers.sort()

            for speaker in speakers:
                print(speaker)
                for (noise, downlim, uplim) in noisetypes:
                    featsvec = bases.read_speaker(numceps, delta_order, 'enroll_1',
                                                  speaker, downlim, uplim)

    t_tot = time.time() - t_tot
    print('Total time: %f seconds' % t_tot)