"""Module with code to handle the base. Extract data, rewrite and etc.
"""


import os, os.path
import shutil
import scipy.io.wavfile as wavf
import numpy as np
from useful import CORPORA_DIR, FEATURES_DIR
import features


def mit_features(winlen=0.02, winstep=0.01, numcep=13, numdeltas=2):
    """Extracts features from base MIT. The features are saved in a directory with
    name given by '../bases/features/mit_<numcep>_<numdeltas>'

    @param winlen: the length of the analysis window in seconds. Default is
    0.02s (20 milliseconds).
    @param winstep: the step between successive windows in seconds. Default is
    0.01s (10 milliseconds).
    @param numcep: the number of cepstrum to return, default 13.
    @param numdeltas: number of delta calculations.
    """
    if not os.path.exists(FEATURES_DIR):
        os.mkdir(FEATURES_DIR)

    #Feature base: mit_<numcep>_<numdeltas>
    pathfeat = '%smit_%d_%d/' % (FEATURES_DIR, numcep, numdeltas)
    print('CREATING %s' % pathfeat)
    if os.path.exists(pathfeat):
        shutil.rmtree(pathfeat)
    os.mkdir(pathfeat)

    pathdatasets = '%smit/' % CORPORA_DIR
    datasets = os.listdir(pathdatasets)
    datasets.sort()

    for dataset in datasets:
        print(dataset)
        os.mkdir('%s%s' % (pathfeat, dataset))
        speakers = os.listdir('%s%s' % (pathdatasets, dataset))
        speakers.sort()

        for speaker in speakers:
            print(speaker)
            #reading list of utterances from each speaker
            pathspeaker = '%s%s/%s' % (pathdatasets, dataset, speaker)
            utterances = os.listdir(pathspeaker)
            utterances.sort()
            utterances = [utt for utt in utterances if utt.endswith('.wav')]

            #path to write in features
            pathspeaker_feat = '%s%s/%s' % (pathfeat, dataset, speaker)
            os.mkdir('%s' % pathspeaker_feat)

            for utt in utterances:
                path_utt = '%s/%s' % (pathspeaker, utt)
                (samplerate, signal) = wavf.read(path_utt)
                mfccs_deltas = features.mfcc_delta(signal, winlen, winstep, samplerate,
                                                   numcep, numdeltas=numdeltas)
                mfccs_deltas = mfccs_deltas.transpose()
                np.save('%s/%s' % (pathspeaker_feat, utt), mfccs_deltas)

def read_features(numcep, numdeltas, dataset, speaker, uttnum, transpose=True):
    """Reads a feature from dataset 'enroll_2' or 'imposter'.

    @returns: The features from the utterance given by (dataset, speaker, uttnum).
    """
    mfccs = np.load('%smit_%d_%d/%s/%s/phrase%02d_16k.wav.npy' % (FEATURES_DIR,
                    numcep, numdeltas, dataset, speaker, uttnum))

    if transpose:
        return mfccs.transpose()
    else:
        return mfccs