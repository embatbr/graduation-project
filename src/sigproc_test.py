#!/usr/bin/python3.4


"""Tests for module 'sigproc'.
"""


import numpy as np
import scipy.io.wavfile as wavf
import matplotlib.pyplot as plt
import sys

from useful import CORPORA_DIR, testplot

from sigproc import *


args = sys.argv[1:]
show = 'noshow' not in args
if not show:
    args.remove('noshow')
runall = args == []

(samplerate, signal) = wavf.read('%smit/enroll_2/f08/phrase54_16k.wav' %
                                 CORPORA_DIR)
numsamples = len(signal)
samples = np.linspace(1, numsamples, numsamples)

#Pre emphasized signal plotting.
if runall or ('preemphasis' in args):
    coeffs = [0, 0.25, 0.5, 0.75, 1]

    for coeff in coeffs:
        presignal = preemphasis(signal, coeff=coeff)
        testplot(samples, presignal, suptitle='Signal (preemph = %.2f)' % coeff,
                 xlabel='time (samples)', ylabel='signal[sample]')

#Common code for options 'frames', 'magnitude' and 'power'.
if runall or (any(option in ['frames', 'magnitude', 'power'] for option in args)):
    frame_len = 0.02*samplerate     #sec * (samples/sec)
    frame_step = 0.01*samplerate    #sec * (samples/sec)
    coeffs = [0, 1]
    winfuncs = [lambda x: np.ones((1, x)), lambda x: np.hamming(x)]   #Hamming Window
    winnames = ['rectangular', 'hamming']

    #Common variable for options 'magnitude' and 'power'.
    if runall or (any(option in ['magnitude', 'power'] for option in args)):
        NFFT = 512

    for coeff in coeffs:
        #Pre emphasized signal
        presignal = preemphasis(signal, coeff=coeff)
        testplot(samples, presignal, suptitle='Signal (preemph = %.2f)' % coeff,
                 xlabel='time (samples)', ylabel='presignal[sample]')

        for (winfunc, winname) in zip(winfuncs, winnames):
            frames = frame_signal(presignal, frame_len, frame_step, winfunc)

            #Framed signal protting.
            if runall or ('frames' in args):
                concatsig = np.array(list())
                for frame in frames:
                    concatsig = np.concatenate((concatsig, frame))
                numconcatsamples = len(concatsig)
                concatsamples = np.linspace(1, numconcatsamples, numconcatsamples)
                testplot(concatsamples, concatsig, suptitle='Frames\n(preemph = %.2f, window = %s)' %
                         (coeff, winname), xlabel='time (samples)', ylabel='concatsig[sample]')

            #Magnitude spectrum
            if runall or ('magnitude' in args):
                magsignal = magspec(frames, NFFT)
                freq = np.linspace(0, samplerate/2, num=math.floor(NFFT/2 + 1))
                fig = plt.figure()
                fig.suptitle('Magnitude of spectrum\n(preemph = %.2f, window = %s)' %
                             (coeff, winname))
                for magsig in magsignal:
                    testplot(freq, magsig, newfig=False, xlabel='frequency (Hz)',
                             ylabel='magspec[f]', options='r')

            #Squared magnitude spectrum
            if runall or ('power' in args):
                powsignal = powspec(frames, NFFT)
                freq = np.linspace(0, samplerate/2, num=math.floor(NFFT/2 + 1))
                fig = plt.figure()
                fig.suptitle('Squared magnitude of spectrum\n(preemph = %.2f, window = %s)' %
                             (coeff, winname))
                for powsig in powsignal:
                    testplot(freq, powsig, newfig=False, xlabel='frequency (Hz)',
                             ylabel='powspec[f]', options='r')

if show:
    plt.show()