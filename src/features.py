"""This module provides classes and functions to perform feature extraction,
such as Mel Frequency Cepstral Coefficents (MFCCs).
"""


import numpy as np

import bases


def preemphasis(signal, coeff=0.95):
    """A highpass filter, with 0 < coeff <= 1. It is basically a differentiation
    in time (or in sample domain, to be more precise).
    """
    x = signal.samples
    wave = list()
    for n in range(1, signal.length()):
        y = x[n] - coeff*x[n - 1]
        wave.append(y)

    x = np.array(wave)
    return bases.Signal((signal.sample_rate, x))


# TEST
if __name__ == '__main__':
    signal = bases.read_signal('mit-preproc/enroll_1/f00.wav')

    import matplotlib.pyplot as plt

    plt.grid(True)
    plt.plot(signal.samples)
    signal = preemphasis(signal)
    plt.plot(signal.samples)

    plt.show()