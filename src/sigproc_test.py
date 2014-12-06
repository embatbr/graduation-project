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


#Pre emphasized signal plotting.
if 'preemphasis' in args:
    (_, signal) = wavf.read('%smit/enroll_2/f08/phrase54_16k.wav' % CORPORA_DIR)
    numsamples = len(signal)
    samples = np.linspace(1, numsamples, numsamples)

    coeffs = [0, 0.25, 0.5, 0.75, 1]
    for coeff in coeffs:
        presignal = preemphasis(signal, coeff=coeff)
        testplot(samples, presignal, suptitle='Signal (preemph = %.2f)' % coeff,
                 xlabel='sample', ylabel='signal(sample)')


#Framed signal protting.
if 'frame' in args:
    (samplerate, signal) = wavf.read('%smit/enroll_2/f08/phrase54_16k.wav' %
                                     CORPORA_DIR)
    frame_len = 0.02*samplerate     #sec * samples/sec
    frame_step = 0.01*samplerate    #sec * samples/sec
    coeffs = [0, 1]
    winfuncs = [lambda x: np.ones((1, x)), lambda x: np.hamming(x)]   #Hamming Window
    winnames = ['rectangular', 'hamming']

    for coeff in coeffs:
        #Pre emphasized signal
        presignal = preemphasis(signal, coeff=coeff)
        numsamples = len(presignal)
        samples = np.linspace(1, numsamples, numsamples)
        testplot(samples, presignal, suptitle='Signal (preemph = %.2f)' % coeff,
                 xlabel='sample', ylabel='signal(sample)')

        for (winfunc, winname) in zip(winfuncs, winnames):
            frames = frame_signal(presignal, frame_len, frame_step, winfunc)
            concatenated = np.array(list())
            for frame in frames:
                concatenated = np.concatenate((concatenated, frame))

            numsamples = len(concatenated)
            samples = np.linspace(1, numsamples, numsamples)
            testplot(samples, concatenated, suptitle='Frames (preemph = %.2f, %s)' %
                     (coeff, winname), xlabel='sample', ylabel='concatenated(sample)')


plt.show()