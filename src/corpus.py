"""Module with code to handle the base. Extract data, rewrite and etc.
"""


import os, os.path
import shutil

import scipy.io.wavfile as wavf
import numpy as np

from useful import CORPORA_DIR, FEATURES_DIR
import features


def mit_features(winlen, winstep, preemph, numcep, num_deltas):
    if not os.path.exists(FEATURES_DIR):
        os.mkdir(FEATURES_DIR)

    pathfeat = '%smit_numcep_%d_deltas_%d_preemph_%f/' % (FEATURES_DIR, numcep,
                                                          num_deltas, preemph)
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
            if dataset == 'enroll_1':    #enroll_1 concatenate all utterances from speaker
                mfccs_deltas_list = list()
            else:                       #others, save each utterance as a file
                os.mkdir('%s' % pathspeaker_feat)

            for utt in utterances:
                path_utt = '%s/%s' % (pathspeaker, utt)
                (samplerate, signal) = wavf.read(path_utt)
                mfccs_deltas = features.mfcc_delta(signal, winlen, winstep, samplerate,
                                                   numcep=numcep, preemph=preemph,
                                                   num_deltas=num_deltas)
                if dataset == 'enroll_1':
                    mfccs_deltas_list.append(mfccs_deltas.transpose())
                else:
                    mfccs_deltas = mfccs_deltas.transpose()
                    np.save('%s/%s' % (pathspeaker_feat, utt), mfccs_deltas)

            if dataset == 'enroll_1':
                mfccs_deltas = np.array(mfccs_deltas_list)
                mfccs_deltas = np.concatenate(mfccs_deltas)
                mfccs_deltas = mfccs_deltas.transpose()
                np.save('%senroll_1/%s' % (pathfeat, speaker), mfccs_deltas)

def read_features(numcep, num_deltas, preemph, dataset, speaker, uttnum):
    mfccs = np.load('%smit_numcep_%d_deltas_%d_preemph_%f/%s/%s/phrase%2d_16k.wav.npy' %
                    (FEATURES_DIR, numcep, num_deltas, preemph, dataset, speaker, uttnum))
    return mfccs.transpose()


#TEST
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import features

    winlen = 0.02
    winstep = 0.01
    numcep = 13
    num_deltas = 2
    preemph = 0.97

    (samplerate, signal) = wavf.read('%smit/enroll_2/f08/phrase54_16k.wav' %
                                     CORPORA_DIR)
    print('signal:')
    print(signal)
    fig = plt.figure()
    plt.grid(True)
    plt.plot(signal) #figure 1
    fig.suptitle('signal')
    plt.xlabel('time (samples)')

    mfccs_deltas = features.mfcc_delta(signal, 0.02, 0.01, samplerate, preemph=preemph,
                                       num_deltas=num_deltas)
    print('mfccs_deltas', len(mfccs_deltas), 'x', len(mfccs_deltas[0]))
    print(mfccs_deltas)
    fig = plt.figure()
    plt.grid(True)
    for melfeat_deltas in mfccs_deltas: #figure 2
        plt.plot(melfeat_deltas)
    fig.suptitle('%d mfccs = %d cepstra + %d deltas\n(calculated)' %
                 (len(mfccs_deltas), numcep, num_deltas))
    plt.xlabel('frame')
    plt.ylabel('feature value')

    mfccs_deltas = read_features(numcep, num_deltas, preemph, 'enroll_2', 'f08', 54)
    print('mfccs_deltas (loaded)', len(mfccs_deltas), 'x', len(mfccs_deltas[0]))
    print(mfccs_deltas)
    fig = plt.figure()
    plt.grid(True)
    for melfeat_deltas in mfccs_deltas: #figure 3
        plt.plot(melfeat_deltas)
    fig.suptitle('%d mfccs = %d cepstra + %d deltas\n(from database)' %
                 (len(mfccs_deltas), numcep, num_deltas))
    plt.xlabel('feature')
    plt.ylabel('feature value')

    plt.show()