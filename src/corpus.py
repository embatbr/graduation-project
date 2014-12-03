"""Module with code to handle the base. Extract data, rewrite and etc.
"""


import os, os.path
import shutil

import scipy.io.wavfile as wavf
import numpy as np

from useful import BASES_DIR
import features


def mit_features(winlen, winstep, preemph=0.97):
    pathfeat = '%smit/features/' % BASES_DIR
    if os.path.exists(pathfeat):
        shutil.rmtree(pathfeat)
    os.mkdir(pathfeat)

    pathcorp = '%smit/corpuses/' % BASES_DIR
    corpuses = os.listdir(pathcorp)
    corpuses.sort()

    for corpus in corpuses:
        print(corpus)
        os.mkdir('%s%s' % (pathfeat, corpus))
        speakers = os.listdir('%s%s' % (pathcorp, corpus))
        speakers.sort()

        for speaker in speakers:
            print(speaker)
            #reading list of utterances from each speaker
            pathspeaker = '%s%s/%s' % (pathcorp, corpus, speaker)
            utterances = os.listdir(pathspeaker)
            utterances.sort()
            utterances = [utt for utt in utterances if utt.endswith('.wav')]

            #path to write in features
            pathspeaker_feat = '%s%s/%s' % (pathfeat, corpus, speaker)
            if corpus == 'enroll_1':    #enroll_1 concatenate all utterances from speaker
                mfccs_deltas_list = list()
            else:                       #others, save each utterance as a file
                os.mkdir('%s' % pathspeaker_feat)

            for utt in utterances:
                path_utt = '%s/%s' % (pathspeaker, utt)
                (samplerate, signal) = wavf.read(path_utt)
                mfccs_deltas = features.mfcc_delta(signal, winlen, winstep, samplerate,
                                                   preemph=preemph)
                if corpus == 'enroll_1':
                    mfccs_deltas_list.append(mfccs_deltas.transpose())
                else:
                    mfccs_deltas = mfccs_deltas.transpose()
                    np.save('%s/%s' % (pathspeaker_feat, utt), mfccs_deltas)

            if corpus == 'enroll_1':
                mfccs_deltas = np.array(mfccs_deltas_list)
                mfccs_deltas = np.concatenate(mfccs_deltas)
                mfccs_deltas = mfccs_deltas.transpose()
                np.save('%senroll_1/%s' % (pathfeat, speaker), mfccs_deltas)

def read_features(corpus, speaker, uttnum):
    mfccs = np.load('%smit/features/%s/%s/phrase%d_16k.wav.npy' % (BASES_DIR,
                    corpus, speaker, uttnum))
    return mfccs.transpose()


#TEST
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import features

    winlen = 0.02
    winstep = 0.01
    preemph = 0.97

    (samplerate, signal) = wavf.read('%smit/corpuses/enroll_2/f08/phrase54_16k.wav' %
                                     BASES_DIR)
    print('signal:')
    print(signal)
    fig = plt.figure()
    plt.grid(True)
    plt.plot(signal) #figure 1
    fig.suptitle('signal')
    plt.xlabel('time (samples)')

    mfccs = features.mfcc_delta(signal, 0.02, 0.01, samplerate, preemph=preemph)
    print('mfccs', len(mfccs), 'x', len(mfccs[0]))
    print(mfccs)
    recovered = np.array(list())
    for mfcc in mfccs:
        recovered = np.concatenate((recovered, mfcc))
    fig = plt.figure()
    plt.grid(True)
    plt.plot(recovered) #figure 2
    fig.suptitle('mfccs')
    plt.xlabel('time (samples)')

    mfccs = read_features('enroll_2', 'f08', 54)
    print('mfccs (loaded)', len(mfccs), 'x', len(mfccs[0]))
    print(mfccs)
    recovered = np.array(list())
    for mfcc in mfccs:
        recovered = np.concatenate((recovered, mfcc))
    fig = plt.figure()
    plt.grid(True)
    plt.plot(recovered) #figure 3
    fig.suptitle('mfccs (read from file)')
    plt.xlabel('time (samples)')

    plt.show()