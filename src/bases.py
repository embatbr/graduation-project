"""Module with code to handle the base. Extract data, rewrite and etc.
"""


import os, os.path
import shutil
import scipy.io.wavfile as wavf
import numpy as np
from common import CORPORA_DIR, FEATURES_DIR
from features import mfcc


def extract_mit(winlen, winstep, numcep, delta_order=0):
    """Extracts features from base MIT. The features are saved in a directory with
    name given by '../bases/features/mit_<numcep>_<delta_order>'.

    @param winlen: the length of the analysis window in seconds.
    @param winstep: the step between successive windows in seconds.
    @param numcep: the number of cepstrum to return.
    @param delta_order: number of delta calculations.
    """
    if not os.path.exists(FEATURES_DIR):
        os.mkdir(FEATURES_DIR)

    #Feature base: mit_<numcep>_<delta_order>
    PATH_FEATS = '%smit_%d_%d/' % (FEATURES_DIR, numcep, delta_order)
    print('CREATING %s' % PATH_FEATS)
    if os.path.exists(PATH_FEATS):
        shutil.rmtree(PATH_FEATS)
    os.mkdir(PATH_FEATS)

    PATH_DATASETS = '%smit/' % CORPORA_DIR
    datasets = os.listdir(PATH_DATASETS)
    datasets.sort()

    for dataset in datasets:
        print(dataset)
        os.mkdir('%s%s' % (PATH_FEATS, dataset))
        speakers = os.listdir('%s%s' % (PATH_DATASETS, dataset))
        speakers.sort()

        for speaker in speakers:
            print(speaker)
            #reading list of utterances from each speaker
            PATH_SPEAKER = '%s%s/%s' % (PATH_DATASETS, dataset, speaker)
            utterances = os.listdir(PATH_SPEAKER)
            utterances.sort()
            utterances = [utt for utt in utterances if utt.endswith('.wav')]

            #path to write in features
            PATH_SPEAKER_FEAT = '%s%s/%s' % (PATH_FEATS, dataset, speaker)
            os.mkdir('%s' % PATH_SPEAKER_FEAT)
            for utt in utterances:
                PATH_UTT = '%s/%s' % (PATH_SPEAKER, utt)

                (samplerate, signal) = wavf.read(PATH_UTT)
                featsvec = mfcc(signal, winlen, winstep, samplerate, numcep=numcep,
                                delta_order=delta_order)
                np.save('%s/%s' % (PATH_SPEAKER_FEAT, utt[6:8]), featsvec)


def read_mit_features(numcep, delta_order, dataset, speaker, featurenum):
    """Reads a feature from a speaker.

    @param numcep: number of cepstral coefficients (used to access the base).
    @param delta_order: order of deltas (used to access the base).
    @param dataset: the dataset from where to extract the feature.
    @param speaker: the speaker to read the features.
    @param featurenum: the feature number.

    @returns: The features from the utterance given by (dataset, speaker, featurenum).
    """
    PATH = '%smit_%d_%d/%s/%s/%02d.npy' % (FEATURES_DIR, numcep, delta_order,
                                           dataset, speaker, featurenum)
    feats = np.load(PATH)
    return feats


def read_mit_speaker(numcep, delta_order, dataset, speaker):
    """Reads the features files from database for a given speaker and concatenate
    in a single matrix of features.

    @param numcep: number of cepstral coefficients (used to access the base).
    @param delta_order: order of deltas (used to access the base).
    @param dataset: the dataset from where to extract the feature.
    @param speaker: the speaker to read the features.

    @returns: a matrix of order NUMFRAMESTOTAL x numcep representing the speaker's
    features.
    """
    PATH_SPEAKER = '%smit_%d_%d/%s/%s' % (FEATURES_DIR, numcep, delta_order,
                                          dataset, speaker)
    featurenums = os.listdir(PATH_SPEAKER)
    featurenums.sort()
    featsvec = None

    for featurenum in featurenums:
        featurenum = int(featurenum[ : 2])
        feats = read_mit_features(numcep, delta_order, dataset, speaker, featurenum)
        if featsvec is None:
            featsvec = feats
        else:
            featsvec = np.vstack((featsvec, feats))

    return featsvec


def read_mit_background(numcep, delta_order, gender):
    """Returns the concatenated MFCCs of a gender from dataset 'enroll_1'.

    @param numcep: number of cepstral coefficients (used to access the base).
    @param delta_order: order of deltas (used to access the base).
    @param gender: tells the gender of the background ('f' or 'm').

    @returns: a matrix of order NUM_FRAMES_TOTAL x numcep representing the MFCCs
    for the background model.
    """
    ENROLL_1_PATH = '%smit_%d_%d/enroll_1' % (FEATURES_DIR, numcep, delta_order)
    speakers = os.listdir(ENROLL_1_PATH)
    speakers = [speaker for speaker in speakers if speaker.startswith(gender)]
    speakers.sort()

    featsvec = None
    for speaker in speakers:
        feats = read_mit_speaker(numcep, delta_order, 'enroll_1', speaker)
        if featsvec is None:
            featsvec = feats
        else:
            featsvec = np.vstack((featsvec, feats))

    return featsvec