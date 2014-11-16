"""This module provides classes and functions to perform feature extraction,
such as MFCC.
"""


import numpy as np

import bases


def pre_emphasis(signal, alpha=1):
    """A highpass filter, with 0 < alpha <= 1.
    """
    samples = signal.samples
    wave = list()
    for n in range(1, len(samples)):
        y = samples[n] - alpha*samples[n - 1]
        wave.append(y)

    samples = np.array(wave)
    return bases.Signal((signal.sample_rate, samples))


# TEST
if __name__ == '__main__':
    signal = bases.read_signal('mit-preproc/enroll_1/f00.wav')

    import matplotlib.pyplot as plt

    plt.grid(True)
    plt.plot(signal.samples)
    signal = pre_emphasis(signal)
    plt.plot(signal.samples)

    plt.show()