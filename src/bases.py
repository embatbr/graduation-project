"""Module with code to handle the base. Extract data, rewrite and etc.
"""


import os, os.path
import shutil
import scipy.io.wavfile as wavf
import numpy as np
from common import CORPORA_DIR, FEATURES_DIR
from features import mfcc


def get_conditions(filename):
    txt_file = open(filename)
    lines = txt_file.readlines()
    location = [line for line in lines if line.lower().startswith('location:')][0]
    location = location[len('location:') : ].strip().lower()
    microphone = [line for line in lines if line.lower().startswith('microphone:')][0]
    microphone = microphone[len('microphone:') : ].strip().lower()

    return (location, microphone)

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
            texts = [utt for utt in utterances if utt.endswith('.txt')]
            utterances = [utt for utt in utterances if utt.endswith('.wav')]

            #path to write in features
            PATH_SPEAKER_FEAT = '%s%s/%s' % (PATH_FEATS, dataset, speaker)
            os.mkdir('%s' % PATH_SPEAKER_FEAT)
            for (utt, txt) in zip(utterances, texts):
                PATH_UTT = '%s/%s' % (PATH_SPEAKER, utt)
                PATH_TXT = '%s/%s' % (PATH_SPEAKER, txt)
                (location, microphone) = get_conditions(PATH_TXT)

                (samplerate, signal) = wavf.read(PATH_UTT)
                featsvec = mfcc(signal, winlen, winstep, samplerate, numcep=numcep,
                             delta_order=delta_order)
                np.save('%s/%s_%s_%s' % (PATH_SPEAKER_FEAT, utt[6:8], location,
                                         microphone), featsvec)

def read_mit_features(numcep, delta_order, dataset, speaker, featurename):
    """Reads a feature from a speaker.

    @param numcep: number of cepstral coefficients (used to access the base).
    @param delta_order: order of deltas (used to access the base).
    @param dataset: the dataset from where to extract the feature.
    @param speaker: the speaker to read the features.
    @param featurename: the feature name, such as '01_office_headset.npy'.

    @returns: The features from the utterance given by (dataset, speaker, featurename).
    """
    PATH = '%smit_%d_%d/%s/%s/%s' % (FEATURES_DIR, numcep, delta_order, dataset,
                                     speaker, featurename)
    feats = np.load(PATH)

    return feats

def read_mit_features_list(numcep, delta_order, dataset, speaker):
    """Reads all features from a speaker.

    @returns: The features from speaker listed by utterance.
    """
    PATH_SPEAKER = '%smit_%d_%d/%s/%s' % (FEATURES_DIR, numcep, delta_order,
                                          dataset, speaker)
    features = os.listdir(PATH_SPEAKER)
    features.sort()
    featslist = list()

    for feature in features:
        featsnum = int(feature[:2])
        feats = read_mit_features(numcep, delta_order, dataset, speaker, featsnum)
        featslist.append(feats)

    return featslist

def read_mit_speaker_features(numcep, delta_order, dataset, speaker, environment,
                              microphone):
    """Reads the features files from database for each speaker and concatenate
    in a single matrix of features.

    @param numcep: number of cepstral coefficients (used to access the base).
    @param delta_order: order of deltas (used to access the base).
    @param dataset: the dataset from where to extract the feature.
    @param speaker: the speaker to read the features.
    @param environment: one of three environments: 'office', 'hallway' or 'intersection'.
    @param microphone: one of two microphones: 'headset' or 'internal'.

    @returns: a matrix of order NUMFRAMESTOTAL x numcep representing the speaker's
    features.
    """
    PATH_SPEAKER = '%smit_%d_%d/%s/%s' % (FEATURES_DIR, numcep, delta_order,
                                          dataset, speaker)
    featurenames = os.listdir(PATH_SPEAKER)
    signature = '_%s_%s.npy' % (environment, microphone)
    featurenames = [featurename for featurename in featurenames
                    if featurename.endswith(signature)]
    featurenames.sort()
    featsvec = None

    for featurename in featurenames:
        feats = read_mit_features(numcep, delta_order, dataset, speaker, featurename)
        if featsvec is None:
            featsvec = feats
        else:
            featsvec = np.vstack((featsvec, feats))

    return featsvec

def read_mit_background_features(numcep, delta_order, gender):
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
        feats = read_mit_speaker_features(numcep, delta_order, 'enroll_1', speaker)
        if featsvec is None:
            featsvec = feats
        else:
            featsvec = np.vstack((featsvec, feats))

    return featsvec


#TESTS
if __name__ == '__main__':
    import sys
    import scipy.io.wavfile as wavf
    import os, os.path, shutil
    import pylab as pl

    from common import CORPORA_DIR

    commands = sys.argv[1:]


    winlen = 0.02
    winstep = 0.01
    numcep = 13
    if 'extract_mit' in commands:
        extract_mit(winlen, winstep, numcep)

    # Reading speech signal
    (samplerate, signal) = wavf.read('%smit/enroll_1/f00/phrase02_16k.wav' %
                                     CORPORA_DIR)
    numsamples = len(signal)
    duration = numsamples/samplerate
    time = np.linspace(1/samplerate, duration, numsamples)

    # figure 1
    featsvec = mfcc(signal, winlen, winstep, samplerate, numcep=numcep)
    pl.subplot(211)
    pl.grid(True)
    pl.plot(featsvec)
    featsf = read_mit_features(numcep, 0, 'enroll_1', 'f00', 2)
    pl.subplot(212)
    pl.grid(True)
    pl.plot(featsf)

    equal = 'Yes' if np.array_equal(featsvec, featsf) else 'No'
    print('Equal? %s.' % equal)

    pl.figure()
    for i in range(6):
        position = 231 + i
        pl.subplot(position)
        pl.plot(featsvec[:, i], featsvec[:, i + 1], '.')
        pl.xlabel('feature %d' % i)
        pl.ylabel('feature %d' % (i + 1))

    read_mit_background_features(numcep, 0, 'f')
    read_mit_background_features(numcep, 0, 'm')

    pl.show()