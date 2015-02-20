import scipy.io.wavfile as wavf
import numpy as np
import os, os.path, shutil
from mfcc import mfcc


BASES_DIR = '../../bases/'
CORPORA_DIR = '%scorpora/' % BASES_DIR
FEATURES_DIR = '%sfeatures/' % BASES_DIR


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
                (ceps, _, _) = mfcc(signal)
                ceps = ceps.transpose()
                np.save('%s/%s' % (pathspeaker_feat, utt[6:8]), ceps)


if os.path.exists(FEATURES_DIR):
        shutil.rmtree(FEATURES_DIR)
os.mkdir(FEATURES_DIR)

print('FEATURE EXTRACTION')

winlen = 0.02
winstep = 0.01
numcep = 13
mit_features(winlen, winstep, numcep, 0)