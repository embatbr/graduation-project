#!/usr/bin/python3.4


"""Tests for module 'features'.
"""


import numpy as np
import scipy.io.wavfile as wavf
import matplotlib.pyplot as plt
import sys

import math

from useful import CORPORA_DIR, testplot
import features
import sigproc


option = sys.argv[1]
args = sys.argv[2:]

(samplerate, signal) = wavf.read('%smit/enroll_2/f08/phrase54_16k.wav' % CORPORA_DIR)
preemph = 0.97
presignal = sigproc.preemphasis(signal, preemph=preemph)

nfilt = 26
NFFT = 512
fftbins = math.floor(NFFT/2 + 1)    #fft bins == 'caixas' de FFT
freq = np.linspace(0, samplerate/2, fftbins)

winlen = 0.02
winstep = 0.01

if option == 'filterbank':
    #Filterbank
    fbank = features.filterbank(samplerate=samplerate, nfilt=nfilt, NFFT=NFFT)
    fig = plt.figure()
    fig.suptitle('%d-filterbank, each with %d FFT bins' % (nfilt, fftbins))
    for f in fbank:
        testplot(freq, f, newfig=False, xlabel='frequency (Hz)', ylabel='filter[f]')

    #Pre emphasized signal's squared magnitude spectrum
    powspec = sigproc.powspec(presignal, NFFT=NFFT)
    testplot(freq, powspec, suptitle='Squared magnitude spectrum\n(preemph = %.2f, NFFT = %d)' %
             (preemph, NFFT), xlabel='frequency (Hz)', ylabel='powspec[f]', fill=True)

    #Pre emphasized signal's squared magnitude spectrum after 21st filter (index 20)
    filter_index = 20
    fspec = np.multiply(powspec, fbank[filter_index])
    testplot(freq, fspec, xlabel='frequency (Hz)', ylabel='powspec[f]',
             fill=True, suptitle='Squared magnitude spectrum at filter[%d]' % filter_index)
    testplot(freq, fbank[filter_index], xlabel='frequency (Hz)', ylabel='filter[f]',
             suptitle='Filter[%d]' % filter_index)

    #Pre emphasized signal's squared magnitude spectrum after filterbank
    fig = plt.figure()
    fig.suptitle('Squared magnitude spectrum after %d-filterbank' % nfilt)
    fspecfull = np.zeros(len(fspec))
    for f in fbank:
        fspec = np.multiply(powspec, f)
        fspecfull = np.maximum(fspecfull, fspec)
    testplot(freq, fspecfull, xlabel='frequency (Hz)', ylabel='powspec[f]',
             fill=True, newfig=False)

elif option == 'filtersignal':
    #Squared magnitude spectrum of pre emphasized signal
    powspec = sigproc.powspec(presignal, NFFT=NFFT)
    testplot(freq, powspec, suptitle='Squared magnitude spectrum\n(preemph = %.2f, NFFT = %d)' %
             (preemph, NFFT), xlabel='frequency (Hz)', ylabel='powspec[f]', fill=True)

    fsignal = features.filtersignal(signal, winlen, winstep, samplerate, nfilt,
                                    NFFT, preemph)
    (feats, energy) = fsignal
    numframes = len(energy)
    frameindices = np.linspace(1, numframes, numframes)
    print(feats.shape)
    print(energy.shape)

    featsfull = np.zeros(numframes)
    for (feat, n) in zip(feats.T, range(numframes)):
        featsfull = np.maximum(featsfull, feat)
        logfeat = np.log10(feat)
        if 'features' in args:
            testplot(frameindices, feat, xlabel='frames', ylabel='feature[frame]',
                     suptitle='Feature %d' % n)
            testplot(frameindices, logfeat, xlabel='frames', ylabel='log(feature[frame])',
                     suptitle='Log of feature %d' % n)

    if not 'features' in args:
        testplot(frameindices, featsfull, fill=True, xlabel='frames',
                 ylabel='max(feature[frame])', suptitle='Maximum feature value per frame')

    #Energy per frame
    testplot(frameindices, energy, fill=True, xlabel='frames', ylabel='energy[frame]',
             suptitle='Energy per frame')

plt.show()