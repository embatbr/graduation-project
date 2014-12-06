#!/usr/bin/python3.4


"""Tests for module 'sigproc'.
"""


import numpy as np
import scipy.io.wavfile as wavf
import matplotlib.pyplot as plt
import sys

import math

from useful import CORPORA_DIR, testplot
import sigproc


option = sys.argv[1]
args = sys.argv[2:]

(samplerate, signal) = wavf.read('%smit/enroll_2/f08/phrase54_16k.wav' % CORPORA_DIR)
numsamples = len(signal)
samples = np.linspace(1, numsamples, numsamples)
NFFT = 512
freq = np.linspace(0, samplerate/2, num=math.floor(NFFT/2 + 1))

#Pre emphasized signal plotting.
if option == 'preemphasis':
    coeffs = [0, 0.25, 0.5, 0.75, 1]

    for coeff in coeffs:
        presignal = sigproc.preemphasis(signal, coeff=coeff)
        testplot(samples, presignal, suptitle='Signal (preemph = %.2f)' % coeff,
                 xlabel='time (samples)', ylabel='signal[sample]')

        #Magnitude of presignal's spectrum
        if 'magspec' in args:
            magsig = sigproc.magspec(presignal, NFFT=NFFT)
            testplot(freq, magsig, suptitle='Magnitude of spectrum\n(preemph = %.2f)' %
                     coeff, xlabel='frequency (Hz)', ylabel='magspec[f]', fill=True)

        #Squared magnitude of presignal's spectrum
        if 'powspec' in args:
            powsig = sigproc.powspec(presignal, NFFT=NFFT)
            testplot(freq, powsig, suptitle='Squared magnitude of spectrum\n(preemph = %.2f)' %
                     coeff, xlabel='frequency (Hz)', ylabel='powspec[f]', fill=True)

#Common code for option 'frames'
elif option == 'frames':
    frame_len = 0.02*samplerate     #sec * (samples/sec)
    frame_step = 0.01*samplerate    #sec * (samples/sec)
    coeffs = [0, 1]
    winfuncs = [lambda x: np.ones((1, x)), lambda x: np.hamming(x)]
    winnames = ['rectangular', 'hamming']

    for coeff in coeffs:
        #Pre emphasized signal
        presignal = sigproc.preemphasis(signal, coeff=coeff)
        testplot(samples, presignal, suptitle='Signal (preemph = %.2f)' % coeff,
                 xlabel='time (samples)', ylabel='presignal[sample]')

        for (winfunc, winname) in zip(winfuncs, winnames):
            frames = sigproc.frame_signal(presignal, frame_len, frame_step, winfunc)
            #Framed signal plotting.
            concatsig = np.array(list())
            for frame in frames:
                concatsig = np.concatenate((concatsig, frame))
            numconcatsamples = len(concatsig)
            concatsamples = np.linspace(1, numconcatsamples, numconcatsamples)
            testplot(concatsamples, concatsig, suptitle='Frames\n(preemph = %.2f, win = %s)' %
                     (coeff, winname), xlabel='time (samples)', ylabel='concatsig[sample]')

            #Magnitude spectrum
            if 'magspec' in args:
                magframes = sigproc.magspec(frames, NFFT)
                magspec = np.zeros(len(magframes[0]))
                for magframe in magframes:
                    magspec = np.maximum(magspec, magframe)
                testplot(freq, magspec, xlabel='frequency (Hz)', ylabel='magspec[f]', fill=True,
                         suptitle='Magnitude of framed spectrum\n(preemph = %.2f, win = %s)' %
                                    (coeff, winname))

            #Squared magnitude spectrum
            if 'powspec' in args:
                powframes = sigproc.powspec(frames, NFFT)
                powspec = np.zeros(len(powframes[0]))
                for powframe in powframes:
                    powspec = np.maximum(powspec, powframe)
                testplot(freq, powspec, xlabel='frequency (Hz)', ylabel='powspec[f]', fill=True,
                         suptitle='Squared magnitude of framed spectrum\n(preemph = %.2f, win = %s)' %
                                    (coeff, winname))

plt.show()