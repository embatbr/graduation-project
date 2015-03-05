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
from common import FEATURES_DIR, GMMS_DIR, UBMS_DIR
from common import EXP_IDENTIFICATION_DIR, EXP_VERIFICATION_DIR, RESULTS_DIR
import bases, mixtures


commands = sys.argv[1:]


numceps = [6]#[6, 13, 19]
delta_orders = [0]#[0, 1, 2]
M = 128


#Extract the MFCCs from base MIT and put in the correct format.
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


if 'train-gmms' in commands:
    if not os.path.exists(GMMS_DIR):
        os.mkdir(GMMS_DIR)

    print('GMM TRAINING\nM = %d' % M)
    t_tot = time.time()

    for numcep in numceps:
        print('numcep = %d' % numcep)
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
    if not os.path.exists(EXP_IDENTIFICATION_DIR):
        os.mkdir(EXP_IDENTIFICATION_DIR)

    print('SPEAKER IDENTIFICATION\nM = %d' % M)

    M_DIR = '%sM_%d/' % (EXP_IDENTIFICATION_DIR, M)
    if not os.path.exists(M_DIR):
        os.mkdir(M_DIR)

    t_tot = time.time()

    for numcep in numceps:
        print('numcep = %d' % numcep)
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
                t = time.time()
                for feature in features:
                    featsvec = bases.read_mit_features(numcep, delta_order, 'enroll_2',
                                                       speaker, int(feature[:2]))
                    probs = [gmm_enroll_1.log_likelihood(featsvec) for gmm_enroll_1
                                                                   in gmms_enroll_1]
                    indentified = gmmfilenames[probs.index(max(probs))][:3]
                    if indentified == speaker:
                        hits = hits + 1

                t = time.time() - t
                hits = (hits / len(features)) * 100
                hitslist.append(hits)
                print('hits = %3.2f%%' % hits)
                print('Indentified in %f seconds' % t)


            EXP_SET_PATH = '%smit_%d_%d.exp' % (M_DIR, numcep, delta_order)
            expfile = open(EXP_SET_PATH, 'w')
            for (speaker, hits) in zip(speakers, hitslist):
                expfile.write('%s %3.2f\n' % (speaker, hits))
            expfile.close()

    t_tot = time.time() - t_tot
    print('Total time: %f seconds' % t_tot)


if 'train-ubms' in commands:
    if not os.path.exists(UBMS_DIR):
        os.mkdir(UBMS_DIR)

    print('UBM TRAINING\nM = %d' % M)

    t_tot = time.time()

    for numcep in numceps:
        print('numcep = %d' % numcep)
        for delta_order in delta_orders:
            print('delta_order = %d' % delta_order)
            GMMS_PATH = '%smit_%d_%d/' % (UBMS_DIR, numcep, delta_order)
            if not os.path.exists(GMMS_PATH):
                os.mkdir(GMMS_PATH)

            featsvec_f = bases.read_mit_background_features(numcep, delta_order, 'f')
            featsvec_m = bases.read_mit_background_features(numcep, delta_order, 'm')
            featsvec = np.vstack((featsvec_f, featsvec_m))

            ubm_unisex = mixtures.GMM(M, featsvec_f)
            print('UBM unisex created. Training...')
            t = time.time()
            ubm_unisex.train(featsvec_f)
            t = time.time() - t
            print('UBM unisex trained in %f seconds' % t)
            GMM_PATH = '%s/unisex_%d.gmm' % (GMMS_PATH, M)
            ubmfile = open(GMM_PATH, 'wb')
            pickle.dump(ubm_unisex, ubmfile)
            ubmfile.close()

            ubm_f = mixtures.GMM(M//2, featsvec_f)
            ubm_m = mixtures.GMM(M//2, featsvec_m)
            print('UBM gender created. Training...')
            t = time.time()
            ubm_f.train(featsvec_f)
            ubm_m.train(featsvec_m)
            t = time.time() - t
            # combination
            ubm_gender = ubm_f
            ubm_gender.M = 2*ubm_gender.M
            ubm_gender.weights = np.hstack((ubm_gender.weights, ubm_m.weights))
            ubm_gender.meansvec = np.vstack((ubm_gender.meansvec, ubm_m.meansvec))
            ubm_gender.variancesvec = np.vstack((ubm_gender.variancesvec, ubm_m.variancesvec))
            print('UBM gender trained in %f seconds' % t)
            GMM_PATH = '%s/gender_%d.gmm' % (GMMS_PATH, M)
            ubmfile = open(GMM_PATH, 'wb')
            pickle.dump(ubm_gender, ubmfile)
            ubmfile.close()

    t_tot = time.time() - t_tot
    print('Total time: %f seconds' % t_tot)


if 'verify' in commands:
    if not os.path.exists(EXP_VERIFICATION_DIR):
        os.mkdir(EXP_VERIFICATION_DIR)

    print('SPEAKER VERIFICATION\nM = %d' % M)

    M_DIR = '%sM_%d/' % (EXP_VERIFICATION_DIR, M)
    if not os.path.exists(M_DIR):
        os.mkdir(M_DIR)

    t_tot = time.time()

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


if 'results-identify' in commands:
    if not os.path.exists(RESULTS_DIR):
        os.mkdir(RESULTS_DIR)

    print('WRITING RESULTS')

    t_tot = time.time()

    resultsfile = open('%sindentify' % RESULTS_DIR, 'w')

    graph = np.zeros((3,3,3))
    i = 0

    Ms = np.array([32, 64, 128])
    for m in Ms:
        j = 0
        print('M = %d' % m)
        print('M_%d' % m, file=resultsfile)
        for numcep in numceps:
            k = 0
            print('numcep = %d' % numcep)
            for delta_order in delta_orders:
                print('delta_order = %d' % delta_order)
                print('mit_%d_%d' % (numcep, delta_order), file=resultsfile)

                EXP_ID_PATH = '%sM_%d/mit_%d_%d.exp' % (EXP_IDENTIFICATION_DIR,
                                                        m, numcep, delta_order)
                exp_id_file = open(EXP_ID_PATH)
                results = dict()
                for line in exp_id_file:
                    line = line.split()
                    (speaker, result) = (line[0], line[1])
                    results[speaker]= float(result)

                values = np.array(list(results.values()))
                mean = np.mean(values)
                std = np.std(values) # desvio padrão
                amax = np.amax(values)
                amin = np.amin(values)

                graph[i,j,k] = mean

                print('values', values, file=resultsfile)
                print('mean: %.2f' % mean, file=resultsfile)
                print('std: %.2f' % std, file=resultsfile)
                print('maximum: %.2f' % amax, file=resultsfile)
                print('minimum: %.2f' % amin, file=resultsfile)

                k += 1
            j += 1
        i += 1

    print(graph)
    pl.plot(Ms, graph[:, 0, 0])
    pl.show()

    t_tot = time.time() - t_tot
    print('Total time: %f seconds' % t_tot)