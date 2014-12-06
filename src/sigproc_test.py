#!/usr/bin/python3.4


"""Tests for module 'sigproc'.
"""


import numpy as np
import scipy.io.wavfile as wavf
import matplotlib.pyplot as plt
import sys

from useful import CORPORA_DIR, testplot

from sigproc import *


args = sys.argv

(samplerate, signal) = wavf.read('%smit/enroll_2/f08/phrase54_16k.wav' %
                                 CORPORA_DIR)

#Pre emphasized signal plotting.
if 'preemphasis' in args:
    numsamples = len(signal)
    samples = np.linspace(1, numsamples, numsamples)
    coeffs = [0, 0.25, 0.5, 0.75, 1]

    for coeff in coeffs:
        presignal = preemphasis(signal, coeff=coeff)
        testplot(samples, presignal, suptitle='Signal (preemph = %.2f)' % coeff,
                 xlabel='time (samples)', ylabel='signal[sample]')


#Common code for options 'frames', 'magnitude' and 'power'.
frame_len = 0.02*samplerate     #sec * samples/sec
frame_step = 0.01*samplerate    #sec * samples/sec
coeffs = [0, 1]
winfuncs = [lambda x: np.ones((1, x)), lambda x: np.hamming(x)]   #Hamming Window
winnames = ['rectangular', 'hamming']

#Framed signal protting.
if 'frames' in args:
    for coeff in coeffs:
        #Pre emphasized signal
        presignal = preemphasis(signal, coeff=coeff)
        numsamples = len(presignal)
        samples = np.linspace(1, numsamples, numsamples)
        testplot(samples, presignal, suptitle='Signal (preemph = %.2f)' % coeff,
                 xlabel='time (samples)', ylabel='presignal[sample]')

        for (winfunc, winname) in zip(winfuncs, winnames):
            frames = frame_signal(presignal, frame_len, frame_step, winfunc)
            concatsig = np.array(list())
            for frame in frames:
                concatsig = np.concatenate((concatsig, frame))

            numsamples = len(concatsig)
            samples = np.linspace(1, numsamples, numsamples)
            testplot(samples, concatsig, suptitle='Frames\n(preemph = %.2f, window = %s)' %
                     (coeff, winname), xlabel='time (samples)', ylabel='concatsig[sample]')


#Common variable for options 'magnitude' and 'power'.
NFFT = 512

#Magnitude spectrum
if 'magnitude' in args:
    for coeff in coeffs:
        #Pre emphasized signal
        presignal = preemphasis(signal, coeff=coeff)
        numsamples = len(presignal)
        samples = np.linspace(1, numsamples, numsamples)
        testplot(samples, presignal, suptitle='Signal (preemph = %.2f)' % coeff,
                 xlabel='sample', ylabel='presignal[sample]')

        for (winfunc, winname) in zip(winfuncs, winnames):
            frames = frame_signal(presignal, frame_len, frame_step, winfunc)
            magsignal = magspec(frames, NFFT)
            freq = np.linspace(0, samplerate/2, num=math.floor(NFFT/2 + 1))
            fig = plt.figure()
            fig.suptitle('Magnitude of spectrum\n(preemph = %.2f, window = %s)' %
                         (coeff, winname))
            for magsig in magsignal:
                testplot(freq, magsig, newfig=False, xlabel='frequency (Hz)',
                         ylabel='magspec[f]', options='r')


#Magnitude spectrum
if 'power' in args:
    for coeff in coeffs:
        #Pre emphasized signal
        presignal = preemphasis(signal, coeff=coeff)
        numsamples = len(presignal)
        samples = np.linspace(1, numsamples, numsamples)
        testplot(samples, presignal, suptitle='Signal (preemph = %.2f)' % coeff,
                 xlabel='sample', ylabel='signal(sample)')

        for (winfunc, winname) in zip(winfuncs, winnames):
            frames = frame_signal(presignal, frame_len, frame_step, winfunc)
            powsignal = powspec(frames, NFFT)
            freq = np.linspace(0, samplerate/2, num=math.floor(NFFT/2 + 1))
            fig = plt.figure()
            fig.suptitle('Magnitude of spectrum\n(preemph = %.2f, window = %s)' %
                         (coeff, winname))
            for powsig in powsignal:
                testplot(freq, powsig, newfig=False, xlabel='frequency (Hz)',
                         ylabel='powspec[f]', options='r')


plt.show()