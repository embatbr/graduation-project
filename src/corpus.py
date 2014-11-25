"""Module with code to handle the base. Extract data, rewrite and etc.
"""


import os, os.path
import shutil

import scipy.io.wavfile as wavf
import numpy as np

import features.base


BASES_DIR = '../bases/'


def mit_to_mfcc():
    pathfeat = '%s%s' % (BASES_DIR, 'mit/features/')
    if os.path.exists(pathfeat):
        shutil.rmtree(pathfeat)
    os.mkdir(pathfeat)
    os.mkdir('%senroll_1' % (pathfeat))
    os.mkdir('%senroll_2' % (pathfeat))
    os.mkdir('%simposter' % (pathfeat))

    #corpus 'enroll_1' will have all mfccs from utterances concatenated
    path_enroll_1 = '%smit/utterances/enroll_1/' % BASES_DIR
    speakers = os.listdir(path_enroll_1)
    speakers.sort()
    for speaker in speakers:
        path_speaker = '%s/%s' % (path_enroll_1, speaker)
        utterances = os.listdir(path_speaker)
        utterances = [utt for utt in utterances if utt.endswith('.wav')]
        utterances.sort()

        mfccs_deltas_list = list()
        for utt in utterances:
            path_utt = '%s/%s' % (path_speaker, utt)
            (samplerate, signal) = wavf.read(path_utt)
            mfccs_deltas = features.base.mfcc_delta(signal, samplerate)
            mfccs_deltas_list.append(mfccs_deltas.transpose())

        mfccs_deltas = np.array(mfccs_deltas_list)
        mfccs_deltas = np.concatenate(mfccs_deltas)
        mfccs_deltas = mfccs_deltas.transpose()
        np.save('%senroll_1/%s.feat' % (pathfeat, speaker), mfccs_deltas)

        print('speaker:', speaker)
        print('mfccs_deltas', len(mfccs_deltas), 'x', len(mfccs_deltas[0]))


#TEST
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import features.sigproc

    coeff = 0.95

    (samplerate, signal) = wavf.read('%smit/utterances/enroll_2/f08/phrase54_16k.wav' % BASES_DIR)
    print('signal:')
    print(signal)
    plt.grid(True)
    plt.plot(signal) #figure 1

    presignal = features.sigproc.preemphasis(signal, coeff=coeff)
    print('preemphasis:')
    print(presignal)
    plt.figure()
    plt.grid(True)
    plt.plot(presignal) #figure 2

    mit_to_mfcc()

    #plt.show()