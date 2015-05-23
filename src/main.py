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
from common import EPS, calculate_eer, SPEAKERS_DIR, isequal, CHECK_DIR
from common import FRAC_GMMS_DIR, FRAC_SPEAKERS_DIR, FRAC_UBMS_DIR, IDENTIFY_DIR
from common import NUM_ENROLLED_UTTERANCES, frange
import bases, mixtures


numceps = 19 # 26 is the default number of filters.
delta_orders = [0, 1, 2]
Ms = [8, 16, 32, 64, 128] # from 128, the EmptyClusterError starts to be frequent
configurations = {'office': ('01', '19'), 'hallway': ('21', '39'),
                  'intersection': ('41', '59'), 'all': ('01', '59')}
environments = ['office', 'hallway', 'intersection', 'all']
enrolled_speakers = ['f%02d' % i for i in range(22)] + ['m%02d' % i for i in range(26)]
rs = [1, 0.99, 1.01, 0.98, 1.02, 0.97, 1.03, 0.96, 1.04, 0.95, 1.05]

command = sys.argv[1]
parameters = sys.argv[2 : ]


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
    print('Total time: %f seconds' % t)


#UBM TRAINING SECTION

def train_ubms(gmms_dir, ubms_dir, r=None):
    for M in Ms:
        print('M = %d' % M)
        for delta_order in delta_orders:
            print('delta_order = %d' % delta_order)
            UBMS_PATH = '%smit_%d_%d/' % (ubms_dir, numceps, delta_order)
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
                ubm_f = mixtures.GMM('f', M // 2, D, featsvec_f, r=r)
                ubm_f.train(featsvec)
                ubm_m = mixtures.GMM('m', M // 2, D, featsvec_m, r=r)
                ubm_m.train(featsvec)

                # combination
                ubm = ubm_f
                r_apx = '' if r is None else '_%.02f' % r
                new_name = '%s_%d%s' % (environment, M, r_apx)
                ubm.absorb(ubm_m, new_name)

                UBM_PATH = '%s%s.gmm' % (UBMS_PATH, ubm.name)
                ubmfile = open(UBM_PATH, 'wb')
                pickle.dump(ubm, ubmfile)
                ubmfile.close()

if command == 'train-ubms':
    if not os.path.exists(GMMS_DIR):
        os.mkdir(GMMS_DIR)
    if not os.path.exists(UBMS_DIR):
        os.mkdir(UBMS_DIR)

    print('UBM TRAINING\nnumceps = %d' % numceps)
    t = time.time()

    train_ubms(GMMS_DIR, UBMS_DIR)

    t = time.time() - t
    print('Total time: %f seconds' % t)

if command == 'train-ubms-frac':
    if not os.path.exists(FRAC_GMMS_DIR):
        os.mkdir(FRAC_GMMS_DIR)
    if not os.path.exists(FRAC_UBMS_DIR):
        os.mkdir(FRAC_UBMS_DIR)

    print('UBM TRAINING\nnumceps = %d' % numceps)
    t = time.time()

    for r in rs:
        train_ubms(FRAC_GMMS_DIR, FRAC_UBMS_DIR, r=r)

    t = time.time() - t
    print('Total time: %f seconds' % t)


#SPEAKER'S GMMs TRAINING SECTION

def train_speakers(gmms_dir, speakers_dir, r=None, debug=False):
    for M in Ms:
        print('M = %d' % M)
        for delta_order in delta_orders:
            print('delta_order = %d' % delta_order)
            ENROLL_1_PATH = '%smit_%d_%d/enroll_1/' % (FEATURES_DIR, numceps, delta_order)
            speakers = os.listdir(ENROLL_1_PATH)
            speakers.sort()

            PATH = '%smit_%d_%d/' % (speakers_dir, numceps, delta_order)
            if not os.path.exists(PATH):
                os.mkdir(PATH)

            for speaker in speakers:
                print(speaker)
                for environment in environments:
                    downlim = configurations[environment][0]
                    uplim = configurations[environment][1]
                    featsvec = bases.read_speaker(numceps, delta_order, 'enroll_1',
                                                  speaker, downlim, uplim)

                    r_apx = '' if r is None else '_%.02f' % r
                    name = '%s_%s_%d%s' % (speaker, environment, M, r_apx)
                    D = numceps * (1 + delta_order)
                    gmm = mixtures.GMM(name, M, D, featsvec, r=r)
                    gmm.train(featsvec, r, debug=debug)

                    GMM_PATH = '%s%s.gmm' % (PATH, gmm.name)
                    gmmfile = open(GMM_PATH, 'wb')
                    pickle.dump(gmm, gmmfile)
                    gmmfile.close()

if command == 'train-speakers':
    if not os.path.exists(GMMS_DIR):
        os.mkdir(GMMS_DIR)
    if not os.path.exists(SPEAKERS_DIR):
        os.mkdir(SPEAKERS_DIR)

    print('SPEAKERS TRAINING\nnumceps = %d' % numceps)
    t = time.time()

    train_speakers(GMMS_DIR, SPEAKERS_DIR)

    t = time.time() - t
    print('Total time: %f seconds' % t)

if command == 'train-speakers-frac':
    if not os.path.exists(FRAC_GMMS_DIR):
        os.mkdir(FRAC_GMMS_DIR)
    if not os.path.exists(FRAC_SPEAKERS_DIR):
        os.mkdir(FRAC_SPEAKERS_DIR)

    print('SPEAKERS TRAINING (fractional)\nnumceps = %d' % numceps)
    print('r = %.02f' % r)
    t = time.time()

    rs.remove(1) # retirar isso depois
    for r in rs:
        print('r = %.02f' % r)
        train_speakers(FRAC_GMMS_DIR, FRAC_SPEAKERS_DIR, r=r, debug=True)

    t = time.time() - t
    print('Total time: %f seconds' % t)


#GMMs ADAPTATION SECTION

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
                ubmfile = open('%s%s_%d.gmm' % (UBMS_PATH, environment, M), 'rb')
                ubm = pickle.load(ubmfile)
                ubmfile.close()

                downlim = configurations[environment][0]
                uplim = configurations[environment][1]
                for speaker in speakers:
                    print(speaker)
                    featsvec = bases.read_speaker(numceps, delta_order, 'enroll_1',
                                                  speaker, downlim, uplim)

                    clone_name = '%s_%s_%d_%s' % (speaker, environment, M, adaptations)
                    gmm = ubm.clone(featsvec, clone_name)
                    gmm.adapt_gmm(featsvec, adaptations, top_C)

                    GMM_PATH = '%s%s.gmm' % (GMMS_PATH, gmm.name)
                    gmmfile = open(GMM_PATH, 'wb')
                    pickle.dump(gmm, gmmfile)
                    gmmfile.close()

    t = time.time() - t
    print('Total time: %f seconds' % t)


#TESTING SECTION

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
            all_gmm_filenames.sort()

            expdict = dict()
            for environment in environments:
                print(environment.upper())
                ubmfile = open('%s%s_%d.gmm' % (UBMS_PATH, environment, M), 'rb')
                ubm = pickle.load(ubmfile)
                ubmfile.close()

                ubm_key = 'UBM %s' % environment
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


if command == 'identify':
    if not os.path.exists(IDENTIFY_DIR):
        os.mkdir(IDENTIFY_DIR)

    r = None
    if len(parameters) > 0:
        r = float(parameters[0])

    identify = 'speakers'
    if r is None:
        identify_dir = '%s%s/' % (IDENTIFY_DIR, identify)
        gmm_dir = '%s%s/' % (GMMS_DIR, identify)
    else:
        identify_dir = '%s%s_%.02f/' % (IDENTIFY_DIR, identify, r)
        gmm_dir = '%s%s/' % (FRAC_GMMS_DIR, identify)

    if not os.path.exists(identify_dir):
        os.mkdir(identify_dir)

    print('Identification\nnumceps = %d' % numceps)
    if not r is None:
        print('r = %.02f' % r)
    t = time.time()

    for M in Ms:
        print('M = %d' % M)
        for delta_order in delta_orders:
            print('delta_order = %d' % delta_order)
            GMMS_PATH = '%smit_%d_%d/' % (gmm_dir, numceps, delta_order)
            EXP_PATH = '%smit_%d_%d/' % (identify_dir, numceps, delta_order)
            if not os.path.exists(EXP_PATH):
                os.mkdir(EXP_PATH)

            all_gmm_filenames = os.listdir(GMMS_PATH)
            all_gmm_filenames.sort()

            expdict = dict()
            for environment in environments:
                print(environment.upper())
                gmms_key = 'GMMs %s' % environment
                expdict[gmms_key] = dict()

                if r is None:
                    expr = '_%s_%d.gmm' % (environment, M)
                else:
                    expr = '_%s_%d_%.02f.gmm' % (environment, M, r)
                gmm_filenames = [gmm_filename for gmm_filename in all_gmm_filenames
                                 if gmm_filename.endswith(expr)]
                gmms = list()
                for gmm_filename in gmm_filenames:
                    gmmfile = open('%s%s' % (GMMS_PATH, gmm_filename), 'rb')
                    gmms.append(pickle.load(gmmfile))
                    gmmfile.close()

                for speaker in enrolled_speakers:
                    print(speaker)
                    expdict[gmms_key][speaker] = list()
                    featslist = bases.read_features_list(numceps, delta_order,
                                                         'enroll_2', speaker)
                    for feats in featslist:
                        log_likes = np.array([gmm.log_likelihood(feats) for gmm in gmms])
                        index = np.argsort(log_likes)[-1]
                        identity = enrolled_speakers[index]
                        expdict[gmms_key][speaker].append(identity)

            EXP_FILE_PATH = '%sidentities_M_%d.json' % (EXP_PATH, M)
            with open(EXP_FILE_PATH, 'w') as expfile:
                json.dump(expdict, expfile, indent=4, sort_keys=True)

    t = time.time() - t
    print('Total time: %f seconds' % t)


#CURVE GENERATION SECTION

if command == 'calc-det-curves':
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


def draw_det_curves(verify_dir):
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

if command == 'draw-det-curves':
    verify_dir = parameters[0]
    print('Drawing DET Curve\nnumceps = %d' % numceps)
    t = time.time()

    draw_det_curves(verify_dir)

    t = time.time() - t
    print('Total time: %f seconds' % t)

if command == 'draw-det-curves-all':
    verify_dirs = os.listdir(VERIFY_DIR)
    verify_dirs.sort()

    print('Drawing DET Curve\nnumceps = %d' % numceps)
    t = time.time()

    for verify_dir in verify_dirs:
        print('verify_dir: %s' % verify_dir)
        draw_det_curves(verify_dir)

    t = time.time() - t
    print('Total time: %f seconds' % t)


if command == 'calc-ident-curves':
    identify = parameters[0]
    identify_dir = '%s%s/' % (IDENTIFY_DIR, identify)

    print('Calculating Identification Curves\nnumceps = %d' % numceps)
    print('identify: %s' % identify)
    t = time.time()

    for delta_order in delta_orders:
        print('delta_order = %d' % delta_order)
        expdicts = list()
        for M in Ms:
            PATH = '%smit_%d_%d/' % (identify_dir, numceps, delta_order)
            EXP_FILE_PATH = '%sidentities_M_%d.json' % (PATH, M)
            expfile = open(EXP_FILE_PATH)
            expdicts.append(json.load(expfile))

        curvesdict = dict()
        for environment in environments:
            print(environment.upper())
            env_key = 'GMMs %s' % environment
            curvesdict[env_key] = dict()

            for (M, expdict) in zip(Ms, expdicts):
                speakers = list(expdict[env_key].keys())
                speakers.sort()

                numhits = 0
                for speaker in speakers:
                    identities = expdict[env_key][speaker]
                    identities = [identity for identity in identities if
                                  identity.startswith(speaker)]
                    numhits = numhits + len(identities)

                    # teste
                    if len(identities) == 0:
                        print(M, speaker, len(identities), identities)

                curvesdict[env_key][M] = (numhits / NUM_ENROLLED_UTTERANCES) * 100

        CURVE_FILE_PATH = '%scurves.json' % PATH
        with open(CURVE_FILE_PATH, 'w') as curvesfile:
            json.dump(curvesdict, curvesfile, indent=4, sort_keys=True)

    t = time.time() - t
    print('Total time: %f seconds' % t)


if command == 'draw-ident-curves':
    identify = parameters[0]
    identify_dir = '%s%s/' % (IDENTIFY_DIR, identify)
    print('Drawing Identification Curves\nnumceps = %d' % numceps)
    t = time.time()

    colors = ['b', 'g', 'r'] # colors: delta 0, delta 1 and delta 2
    position = 1
    xticks = np.array(Ms)
    yticks = np.arange(0, 101, 10)

    for environment in environments:
        print(environment.upper())
        gmms_key = 'GMMs %s' % environment
        ax = pl.subplot(2, 2, position)
        ax.set_title(environment, fontsize=10)
        pl.grid(True)

        if position == 3 or position == 4:
            pl.subplots_adjust(hspace=0.3)

        position = position + 1
        color_index = 0

        for delta_order in delta_orders:
            print('delta_order = %d' % delta_order)
            PATH = '%smit_%d_%d/' % (identify_dir, numceps, delta_order)
            CURVE_FILE_PATH = '%scurves.json' % PATH
            curvesfile = open(CURVE_FILE_PATH)
            curvesdict = json.load(curvesfile)

            keys = list(map(int, curvesdict[gmms_key].keys()))
            keys.sort()
            values = [curvesdict[gmms_key][str(key)] for key in keys]
            pl.plot(keys, values, '.-%s' % colors[color_index])
            color_index = color_index + 1

        pl.xticks(xticks)
        pl.yticks(yticks)

        [tick.label.set_fontsize(7) for tick in ax.xaxis.get_major_ticks()]
        [tick.label.set_fontsize(7) for tick in ax.yaxis.get_major_ticks()]

    pl.subplot(2, 2, 1)
    pl.legend(('delta 0','delta 1', 'delta 2'), loc='upper right', prop={'size':7})

    CURVES_IMG_PATH = '%scurves.png' % identify_dir
    pl.savefig(CURVES_IMG_PATH, bbox_inches='tight')

    t = time.time() - t
    print('Total time: %f seconds' % t)


# Códigos de correção para merdas que fiz anteriormente e demorariam muito tempo
# para refazer

def check(directory):
    if not os.path.exists(CHECK_DIR):
        os.mkdir(CHECK_DIR)

    CHECK_PATH = '%s%s.check' % (CHECK_DIR, directory)
    checkfile = open(CHECK_PATH, 'w')
    problems = list()

    adaptations = ''
    if directory.startswith('adapted'):
        adaptations = '_%s' % directory.split('_')[1]

    for M in Ms:
        for delta_order in delta_orders:
            PATH = '%s%s/mit_%d_%d/' % (GMMS_DIR, directory, numceps, delta_order)
            filenames = os.listdir(PATH)
            filenames.sort()

            if directory == 'ubms':
                speakers = [None] # GAMBI
            else:
                speakers = ['f%02d' % i for i in range(22)] + ['m%02d' % i for i in range(26)]

            for speaker in speakers:
                for environment in environments:
                    if directory == 'ubms':
                        GMM_PATH = '%s%s_%d%s.gmm' % (PATH, environment, M, adaptations)
                    else:
                        GMM_PATH = '%s%s_%s_%d%s.gmm' % (PATH, speaker, environment, M, adaptations)

                    if os.path.exists('%s' % GMM_PATH):
                        gmmfile = open('%s' % GMM_PATH, 'rb')
                        gmm = pickle.load(gmmfile)
                        gmmfile.close()

                        if np.min(gmm.weights) <= 0 or np.max(gmm.weights) >= 1:
                            problems.append('%s: exist weight not between 0 and 1' % GMM_PATH)
                        if isequal(np.sum(gmm.weights, axis=0), 1):
                            problems.append('%s%s: weights not summing to 1: %f' %
                                            (PATH, gmm.name, np.sum(gmm.weights)))
                        if np.min(gmm.variancesvec) < MIN_VARIANCE:
                            problems.append('%s: negative variances' % GMM_PATH)
                    else:
                        problems.append('%s: does not exist' % GMM_PATH)

    if len(problems) == 0:
        print('OK', file=checkfile)
    else:
        for problem in problems:
            print(problem, file=checkfile)

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
                ubmfile = open('%s%s_%d.gmm' % (UBMS_PATH, environment, M), 'rb')
                ubm = pickle.load(ubmfile)
                ubmfile.close()
                ubm.name = '%s_%d' % (environment, M)

                UBM_PATH = '%s%s.gmm' % (UBMS_PATH, ubm.name)
                ubmfile = open(UBM_PATH, 'wb')
                pickle.dump(ubm, ubmfile)
                ubmfile.close()

                print(UBM_PATH, ubm.name)


if command == 'delete-ubm-extensions':
    for M in Ms:
        for delta_order in delta_orders:
            UBMS_PATH = '%smit_%d_%d/' % (UBMS_DIR, numceps, delta_order)
            for environment in environments:
                os.remove('%s%s_%d.ubm' % (UBMS_PATH, environment, M))