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


option = sys.argv[1]
args = sys.argv[2:]

(samplerate, signal) = wavf.read('%smit/enroll_2/f08/phrase54_16k.wav' % CORPORA_DIR)
nfilt = 26
NFFT = 512

if option == 'filterbank':
    filters = features.filterbank(samplerate=samplerate, nfilt=nfilt, NFFT=NFFT)
    fftbins = math.floor(NFFT/2 + 1)    #fft bins == 'caixas' de FFT
    freq = np.linspace(0, samplerate/2, fftbins)

    fig = plt.figure()
    fig.suptitle('%d filterbank, each with %d FFT bins' % (nfilt, fftbins))
    for f in filters:
        testplot(freq, f, newfig=False, xlabel='frequency (Hz)', ylabel='filter[f]')

plt.show()