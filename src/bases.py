"""Module with code to handle the base. Extract data, rewrite and etc.
"""


import os, os.path
import shutil
import scipy.io.wavfile as wavf
import numpy as np
from common import CORPORA_DIR, FEATURES_DIR
from features import mfcc


def extract(winlen, winstep, numceps, delta_order):
    """Extracts features from base MIT. The features are saved in a directory with
    name given by '../bases/features/mit_<numceps>_<delta_order>'.

    @param winlen: the length of the analysis window in seconds.
    @param winstep: the step between successive windows in seconds.
    @param numceps: the number of cepstrum to return.
    @param delta_order: number of delta calculations.
    """
    if not os.path.exists(FEATURES_DIR):
        os.mkdir(FEATURES_DIR)

    #Feature base: mit_<numceps>_<delta_order>
    PATH_FEATS = '%smit_%d_%d/' % (FEATURES_DIR, numceps, delta_order)
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
                featsvec = mfcc(signal, winlen, winstep, samplerate, numceps=numceps,
                                delta_order=delta_order)
                np.save('%s/%s' % (PATH_SPEAKER_FEAT, utt[6:8]), featsvec)


def read_features(numceps, delta_order, dataset, speaker, featurename):
    """Reads a feature from a speaker.

    @param numceps: number of cepstral coefficients (used to access the base).
    @param delta_order: order of deltas (used to access the base).
    @param dataset: the dataset from where to extract the feature.
    @param speaker: the speaker to read the features.
    @param featurename: the feature number.

    @returns: The features from the utterance given by (dataset, speaker, featurename).
    """
    PATH = '%smit_%d_%d/%s/%s/%s' % (FEATURES_DIR, numceps, delta_order,
                                           dataset, speaker, featurename)
    feats = np.load(PATH)
    return feats


def read_features_list(numceps, delta_order, dataset, speaker, downlim='01', uplim='59'):
    """Reads a list of features from a speaker.

    @param numceps: number of cepstral coefficients (used to access the base).
    @param delta_order: order of deltas (used to access the base).
    @param dataset: the dataset from where to extract the feature.
    @param speaker: the speaker to read the features.
    @param downlim: the bottom limit for signal reading.
    @param uplim: the top limit for signal reading.

    @returns: a matrix of order NUMFRAMESTOTAL x numceps representing the speaker's
    features.
    """
    PATH_SPEAKER = '%smit_%d_%d/%s/%s' % (FEATURES_DIR, numceps, delta_order,
                                          dataset, speaker)
    featurenames = os.listdir(PATH_SPEAKER)
    featurenames.sort()

    featslist = list()
    for featurename in featurenames:
        if featurename[:2] >= downlim and featurename[:2] <= uplim:
            feats = read_features(numceps, delta_order, dataset, speaker, featurename)
            featslist.append(feats)

    return featslist


def read_speaker(numceps, delta_order, dataset, speaker, downlim='01', uplim='59'):
    """Reads the features files from database for a given speaker and concatenate
    in a single matrix of features.

    @param numceps: number of cepstral coefficients (used to access the base).
    @param delta_order: order of deltas (used to access the base).
    @param dataset: the dataset from where to extract the feature.
    @param speaker: the speaker to read the features.
    @param downlim: the bottom limit for signal reading.
    @param uplim: the top limit for signal reading.

    @returns: a matrix of order NUMFRAMESTOTAL x numceps representing the speaker's
    features.
    """
    PATH_SPEAKER = '%smit_%d_%d/%s/%s' % (FEATURES_DIR, numceps, delta_order,
                                          dataset, speaker)
    featurenames = os.listdir(PATH_SPEAKER)
    featurenames.sort()
    featsvec = None

    for featurename in featurenames:
        if featurename[:2] >= downlim and featurename[:2] <= uplim:
            feats = read_features(numceps, delta_order, dataset, speaker, featurename)
            if featsvec is None:
                featsvec = feats
            else:
                featsvec = np.vstack((featsvec, feats))

    return featsvec


def read_background(numceps, delta_order, gender=None, downlim='01', uplim='59'):
    """Returns the concatenated MFCCs of a gender from dataset 'enroll_1'.

    @param numceps: number of cepstral coefficients (used to access the base).
    @param delta_order: order of deltas (used to access the base).
    @param gender: tells the gender of the background ('f' or 'm'). Default None.
    @param downlim: the bottom limit for signal reading.
    @param uplim: the top limit for signal reading.

    @returns: a matrix of order NUM_FRAMES_TOTAL x numceps representing the MFCCs
    for the background model.
    """
    ENROLL_1_PATH = '%smit_%d_%d/enroll_1' % (FEATURES_DIR, numceps, delta_order)
    speakers = os.listdir(ENROLL_1_PATH)
    if not gender is None:
        speakers = [speaker for speaker in speakers if speaker.startswith(gender)]
    speakers.sort()

    featsvec = None
    for speaker in speakers:
        feats = read_speaker(numceps, delta_order, 'enroll_1', speaker, downlim,
                             uplim)
        if featsvec is None:
            featsvec = feats
        else:
            featsvec = np.vstack((featsvec, feats))

    return featsvec