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
nfilt = 26
NFFT = 512
coeff = 0.97

if option == 'filterbank':
    fbank = features.filterbank(samplerate=samplerate, nfilt=nfilt, NFFT=NFFT)
    fftbins = math.floor(NFFT/2 + 1)    #fft bins == 'caixas' de FFT
    freq = np.linspace(0, samplerate/2, fftbins)

    fig = plt.figure()
    fig.suptitle('%d filterbank, each with %d FFT bins' % (nfilt, fftbins))
    for f in fbank:
        testplot(freq, f, newfig=False, xlabel='frequency (Hz)', ylabel='filter[f]')

    presignal = sigproc.preemphasis(signal, coeff=coeff)
    pspec = sigproc.powspec(presignal, NFFT=NFFT)
    testplot(freq, pspec, suptitle='Squared magnitude of spectrum\n(preemph = %.2f, NFFT = %d)' %
             (coeff, NFFT), xlabel='frequency (Hz)', ylabel='powspec[f]', fill=True)

    filter_index = 20
    fspec = np.multiply(pspec, fbank[filter_index])
    testplot(freq, fspec, xlabel='frequency (Hz)', ylabel='powspec[f]',
             fill=True, suptitle='Squared magnitude spectrum at %dº filter' % filter_index)
    testplot(freq, fbank[filter_index], xlabel='frequency (Hz)', ylabel='filter[f]',
             suptitle='Filter %d' % filter_index)

    fig = plt.figure()
    fig.suptitle('Squared magnitude spectrum filtered')
    fspecfull = np.zeros(len(fspec))
    for f in fbank:
        fspec = np.multiply(pspec, f)
        fspecfull = np.maximum(fspecfull, fspec)
    testplot(freq, fspecfull, xlabel='frequency (Hz)', ylabel='powspec[f]',
             fill=True, newfig=False)

plt.show()