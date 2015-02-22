#!/usr/bin/python3.4

"""Module to execute all important things from project.
"""


import sys
import os, os.path
import shutil
from common import FEATURES_DIR, GMMS_DIR
import corpus


commands = sys.argv[1:]

delta_orders = [0, 1, 2]
Ms = [8, 16, 32, 64, 128, 256, 512, 1024]
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
    for delta_order in delta_orders:
        corpus.extract_mit(winlen, winstep, numcep, delta_order)


if 'train-gmms' in commands:
    if not os.path.exists(GMMS_DIR):
        os.mkdir(GMMS_DIR)

    print('GMM TRAINING')

    for M in Ms:
        print('M = %d' % M)
        for delta_order in delta_orders:
            print('delta_order = %d' % delta_order)
            GMMS_PATH = '%smit_%d_%d/' % (GMMS_DIR, numcep, delta_order)
            if not os.path.exists(GMMS_PATH):
                os.mkdir(GMMS_PATH)

            numfeats = numcep*(delta_order + 1)
            for dataset in datasets:
                print(dataset)
                DATASET_PATH = '%smit_%d_%d/%s/' % (FEATURES_DIR, numcep, delta_order,
                                                    dataset)
                speakers = os.listdir(DATASET_PATH)
                speakers.sort()

                for speaker in speakers:
                    feats = corpus.read_mit_speaker_features(numcep, delta_order,
                                                             dataset, speaker)
                    print('%s = %s' % (speaker, feats.shape))