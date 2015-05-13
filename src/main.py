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
from common import EPS, calculate_eer, SPEAKERS_DIR, isequal
import bases, mixtures


command = sys.argv[1]
parameters = sys.argv[2 : ]

numceps = 19 # 26 is the default number of filters.
delta_orders = [0, 1, 2]
Ms = [8, 16, 32, 64, 128] # from 128, the EmptyClusterError starts to be frequent
configurations = {'office': ('01', '19'), 'hallway': ('21', '39'),
                  'intersection': ('41', '59'), 'all': ('01', '59')}
environments = ['office', 'hallway', 'intersection', 'all']


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
    if not os.path.exists(GMMS_DIR):
        os.mkdir(GMMS_DIR)
    if not os.path.exists(UBMS_DIR):
        os.mkdir(UBMS_DIR)

    print('UBM TRAINING\nnumceps = %d' % numceps)
    t = time.time()

    for M in Ms:
        print('M = %d' % M)
        for delta_order in delta_orders:
            print('delta_order = %d' % delta_order)
            UBMS_PATH = '%smit_%d_%d/' % (UBMS_DIR, numceps, delta_order)
            if not os.path.exists(UBMS_PATH):
                os.mkdir(UBMS_PATH)

            for environment in environments:
                print(environment.upper())
                downlim = configurations[environment][0]
                uplim = configurations[environment][1]
                featsvec_f = bases.read_background(numceps, delta_order, 'f',
                                                   downlim, uplim)
                featsvec_m = bases.read_background(numceps, delta_order, 'm',
                                                   downlim, uplim)

                # training
                D = numceps * (1 + delta_order)
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
                new_name = '%s_%d' % (environment, M)
                ubm.merge(ubm_m, new_name)

                UBM_PATH = '%s%s.ubm' % (UBMS_PATH, ubm.name)
                ubmfile = open(UBM_PATH, 'wb')
                pickle.dump(ubm, ubmfile)
                ubmfile.close()

    t = time.time() - t
    print('Total time: %f seconds' % t)


if command == 'train-speakers':
    if not os.path.exists(GMMS_DIR):
        os.mkdir(GMMS_DIR)
    if not os.path.exists(SPEAKERS_DIR):
        os.mkdir(SPEAKERS_DIR)

    print('SPEAKERS TRAINING\nnumceps = %d' % numceps)
    t = time.time()

    for M in Ms:
        print('M = %d' % M)
        for delta_order in delta_orders:
            print('delta_order = %d' % delta_order)
            ENROLL_1_PATH = '%smit_%d_%d/enroll_1/' % (FEATURES_DIR, numceps, delta_order)
            speakers = os.listdir(ENROLL_1_PATH)
            speakers.sort()

            PATH = '%smit_%d_%d/' % (SPEAKERS_DIR, numceps, delta_order)
            if not os.path.exists(PATH):
                os.mkdir(PATH)

            for speaker in speakers:
                print(speaker)
                for environment in environments:
                    downlim = configurations[environment][0]
                    uplim = configurations[environment][1]
                    featsvec = bases.read_speaker(numceps, delta_order, 'enroll_1',
                                                  speaker, downlim, uplim)

                    D = numceps * (1 + delta_order)
                    name = '%s_%s_%d' % (speaker, environment, M)
                    gmm = mixtures.GMM(name, M, D)
                    while(True):
                        try:
                            print('Training GMM %s' % gmm.name)
                            gmm.train(featsvec)
                            break
                        except mixtures.EmptyClusterError as e:
                            print('%s\nrebooting GMM %s' % (e.msg, gmm.name))

                    GMM_PATH = '%s%s.gmm' % (PATH, gmm.name)
                    gmmfile = open(GMM_PATH, 'wb')
                    pickle.dump(gmm, gmmfile)
                    gmmfile.close()

    t = time.time() - t
    print('Features extracted in %f seconds' % t)


if command == 'adapt-gmms':
    adaptations = parameters[0]
    top_C = None
    if len(parameters) > 1:
        top_C = int(parameters[1])

    if not os.path.exists(GMMS_DIR):
        os.mkdir(GMMS_DIR)

    if top_C is None:
        adapted_gmm_dir = '%sadapted_%s/' % (GMMS_DIR, adaptations)
    else:
        adapted_gmm_dir = '%sadapted_%s_C%d/' % (GMMS_DIR, adaptations, top_C)
    if not os.path.exists(adapted_gmm_dir):
        os.mkdir(adapted_gmm_dir)

    print('Adapting GMMs from UBM\nnumceps = %d' % numceps)
    print('adaptations: %s' % adaptations)
    if not top_C is None:
        print('top C: %d' % top_C)
    t = time.time()

    for M in Ms:
        print('M = %d' % M)
        for delta_order in delta_orders:
            print('delta_order = %d' % delta_order)
            ENROLL_1_PATH = '%smit_%d_%d/enroll_1/' % (FEATURES_DIR, numceps, delta_order)
            speakers = os.listdir(ENROLL_1_PATH)
            speakers.sort()

            UBMS_PATH = '%smit_%d_%d/' % (UBMS_DIR, numceps, delta_order)
            GMMS_PATH = '%smit_%d_%d/' % (adapted_gmm_dir, numceps, delta_order)
            if not os.path.exists(GMMS_PATH):
                os.mkdir(GMMS_PATH)

            for environment in environments:
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
                    gmm = ubm.clone('%s_%s_%d_%s' % (speaker, environment, M, adaptations))
                    gmm.adapt_gmm(featsvec, adaptations, top_C)

                    GMM_PATH = '%s%s.gmm' % (GMMS_PATH, gmm.name)
                    gmmfile = open(GMM_PATH, 'wb')
                    pickle.dump(gmm, gmmfile)
                    gmmfile.close()

    t = time.time() - t
    print('Total time: %f seconds' % t)


if command == 'verify':
    verify = 'speakers'
    adaptations = None
    if len(parameters) > 0:
        adaptations = parameters[0]
        verify = 'adapted_%s' % adaptations
    if len(parameters) > 1:
        C = parameters[1]
        verify = '%s_C%s' % (verify, C)
    verify_dir = '%s%s/' % (VERIFY_DIR, verify)
    gmm_dir = '%s%s/' % (GMMS_DIR, verify)

    if not os.path.exists(VERIFY_DIR):
        os.mkdir(VERIFY_DIR)
    if not os.path.exists(verify_dir):
        os.mkdir(verify_dir)

    print('Verification\nnumceps = %d' % numceps)
    print('verify: %s' % verify)
    t = time.time()

    for M in Ms:
        print('M = %d' % M)
        for delta_order in delta_orders:
            print('delta_order = %d' % delta_order)
            UBMS_PATH = '%smit_%d_%d/' % (UBMS_DIR, numceps, delta_order)
            GMMS_PATH = '%smit_%d_%d/' % (gmm_dir, numceps, delta_order)
            EXP_PATH = '%smit_%d_%d/' % (verify_dir, numceps, delta_order)
            if not os.path.exists(EXP_PATH):
                os.mkdir(EXP_PATH)

            all_gmm_filenames = os.listdir(GMMS_PATH)
            if adaptations is None:
                expr = '_%d.gmm' % M
            else:
                expr = '_%d_%s.gmm' % (M, adaptations)
            all_gmm_filenames = [gmm_filename for gmm_filename in all_gmm_filenames
                                 if gmm_filename.endswith(expr)]
            all_gmm_filenames.sort()

            expdict = dict()
            for environment in environments:
                print(environment.upper())
                ubmfile = open('%s%s_%d.ubm' % (UBMS_PATH, environment, M), 'rb')
                ubm = pickle.load(ubmfile)
                ubmfile.close()

                ubm_key = 'UBM %s' % ubm.name.split('_')[0]
                expdict[ubm_key] = dict()
                for env in environments:
                    env_key = 'SCORES %s' % env
                    expdict[ubm_key][env_key] = dict()
                    expdict[ubm_key][env_key]['enrolled'] = list()
                    expdict[ubm_key][env_key]['imposter'] = list()

                if adaptations is None:
                    expr = '_%s_%d.gmm' % (environment, M)
                else:
                    expr = '_%s_%d_%s.gmm' % (environment, M, adaptations)
                gmm_filenames = [gmm_filename for gmm_filename in all_gmm_filenames
                                 if gmm_filename.endswith(expr)]

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

                            # each environment has 18 utterances
                            expdict[ubm_key]['SCORES all'][dataset_key].append(score)
                            if i < 18:
                                expdict[ubm_key]['SCORES office'][dataset_key].append(score)
                            if 18 <= i < 36:
                                expdict[ubm_key]['SCORES hallway'][dataset_key].append(score)
                            if 36 <= i < 54:
                                expdict[ubm_key]['SCORES intersection'][dataset_key].append(score)

            EXP_FILE_PATH = '%sscores_M_%d.json' % (EXP_PATH, M)
            with open(EXP_FILE_PATH, 'w') as expfile:
                json.dump(expdict, expfile, indent=4, sort_keys=True)

    t = time.time() - t
    print('Total time: %f seconds' % t)


if command == 'calc-det-curve':
    verify = parameters[0]
    verify_dir = '%s%s/' % (VERIFY_DIR, verify)

    print('Calculating DET Curve\nnumceps = %d' % numceps)
    print('verify: %s' % verify)
    t = time.time()

    for delta_order in delta_orders:
        print('delta_order = %d' % delta_order)
        for M in Ms:
            print('M = %d' % M)
            PATH = '%smit_%d_%d/' % (verify_dir, numceps, delta_order)
            EXP_FILE_PATH = '%sscores_M_%d.json' % (PATH, M)
            expfile = open(EXP_FILE_PATH)
            expdict = json.load(expfile)

            detdict = dict()
            for environment in environments:
                ubm_key = 'UBM %s' % environment
                detdict[ubm_key] = dict()

                for environment in environments:
                    scores_key = 'SCORES %s' % environment
                    detdict[ubm_key][scores_key] = dict()

                    enrolled = np.array(expdict[ubm_key][scores_key]['enrolled'])
                    imposter = np.array(expdict[ubm_key][scores_key]['imposter'])
                    scores = np.sort(np.hstack((enrolled, imposter)))
                    scores = np.hstack((scores, np.max(scores) + 10*EPS))

                    detdict[ubm_key][scores_key]['false_detection'] = list()
                    detdict[ubm_key][scores_key]['false_rejection'] = list()

                    for score in scores:
                        false_detection = imposter[imposter >= score]
                        false_detection = (len(false_detection) / len(imposter)) * 100
                        detdict[ubm_key][scores_key]['false_detection'].append(false_detection)

                        false_rejection = enrolled[enrolled < score]
                        false_rejection = (len(false_rejection) / len(enrolled)) * 100
                        detdict[ubm_key][scores_key]['false_rejection'].append(false_rejection)

                    fd = detdict[ubm_key][scores_key]['false_detection']
                    fr = detdict[ubm_key][scores_key]['false_rejection']
                    (EER, EER_index) = calculate_eer(fd, fr)
                    detdict[ubm_key][scores_key]['EER'] = EER
                    detdict[ubm_key][scores_key]['EER_score'] = scores[EER_index]

            DET_FILE_PATH = '%sdet_M_%d.json' % (PATH, M)
            with open(DET_FILE_PATH, 'w') as detfile:
                json.dump(detdict, detfile, indent=4, sort_keys=True)

    t = time.time() - t
    print('Total time: %f seconds' % t)


if command == 'draw-det-curve':
    verify_dirs = os.listdir(VERIFY_DIR)
    verify_dirs.sort()

    print('Drawing DET Curve\nnumceps = %d' % numceps)
    t = time.time()

    for verify_dir in verify_dirs:
        print('verify_dir: %s' % verify_dir)
        for delta_order in delta_orders:
            print('delta_order = %d' % delta_order)
            for M in Ms:
                print('M = %d' % M)
                PATH = '%s%s/mit_%d_%d/' % (VERIFY_DIR, verify_dir, numceps, delta_order)
                DET_FILE_PATH = '%sdet_M_%d.json' % (PATH, M)
                detfile = open(DET_FILE_PATH)
                detdict = json.load(detfile)

                colors = ['b', 'g', 'r', 'k'] # colors: office, hallway, intersection and all
                position = 1
                ticks = np.arange(10, 100, 10)
                eer_line = np.linspace(0, 100, 101)

                pl.clf()
                for environment in environments:
                    ubm_key = 'UBM %s' % environment
                    ax = pl.subplot(2, 2, position)
                    ax.set_title(environment, fontsize=10)
                    pl.grid(True)
                    pl.xticks(ticks)
                    pl.yticks(ticks)

                    [tick.label.set_fontsize(7) for tick in ax.xaxis.get_major_ticks()]
                    [tick.label.set_fontsize(7) for tick in ax.yaxis.get_major_ticks()]

                    if position == 3 or position == 4:
                        pl.subplots_adjust(hspace=0.3)

                    position = position + 1
                    color_index = 0

                    for environment in environments:
                        scores_key = 'SCORES %s' % environment
                        false_detection = detdict[ubm_key][scores_key]['false_detection']
                        false_rejection = detdict[ubm_key][scores_key]['false_rejection']
                        pl.plot(false_detection, false_rejection, colors[color_index])
                        color_index = color_index + 1

                    pl.xlabel('false detection', fontsize=7)
                    pl.ylabel('false rejection', fontsize=7)

                pl.subplot(221)
                pl.legend(('office','hallway', 'intersection', 'all'),
                           loc='upper right', prop={'size':7})

                DET_IMG_PATH = '%sdet_M_%d.png' % (PATH, M)
                pl.savefig(DET_IMG_PATH, bbox_inches='tight')

    t = time.time() - t
    print('Total time: %f seconds' % t)


# Códigos de correção para merdas que fiz anteriormente e demorariam muito tempo
# para refazer

def check(directory):
    for M in Ms:
        print('M =', M)
        for delta_order in delta_orders:
            print('delta_order =', delta_order)
            PATH = '%s%s/mit_%d_%d/' % (GMMS_DIR, directory, numceps, delta_order)
            filenames = os.listdir(PATH)
            filenames.sort()

            for filename in filenames:
                gmmfile = open('%s%s' % (PATH, filename), 'rb')
                gmm = pickle.load(gmmfile)
                gmmfile.close()

                if np.min(gmm.weights) <= 0 or np.min(gmm.weights) >= 1:
                    print(PATH, gmm.name)
                    print('Some of the weights are not between 0 and 1')
                    print(PATH, gmm.weights)
                if isequal(np.sum(gmm.weights), 1):
                    print('Weights not summing to 1')
                    print('sum:', np.sum(gmm.weights))
                if np.min(gmm.variancesvec) < MIN_VARIANCE:
                    print(PATH, gmm.name)
                    print('Negative variancesvec')
                    print(gmm.variancesvec[gmm.variancesvec < MIN_VARIANCE])

if command == 'check':
    directory = parameters[0]
    print('Directory:', directory)
    check(directory)

if command == 'check-all':
    directories = os.listdir(GMMS_DIR)
    directories.sort()
    for directory in directories:
        print('Directory:', directory)
        check(directory)


if command == 'correct-ubms-names':
    for M in Ms:
        for delta_order in delta_orders:
            UBMS_PATH = '%smit_%d_%d/' % (UBMS_DIR, numceps, delta_order)
            for environment in environments:
                ubmfile = open('%s%s_%d.ubm' % (UBMS_PATH, environment, M), 'rb')
                ubm = pickle.load(ubmfile)
                ubmfile.close()
                ubm.name = '%s_%d' % (environment, M)

                UBM_PATH = '%s%s.ubm' % (UBMS_PATH, ubm.name)
                ubmfile = open(UBM_PATH, 'wb')
                pickle.dump(ubm, ubmfile)
                ubmfile.close()

                print(UBM_PATH, ubm.name)