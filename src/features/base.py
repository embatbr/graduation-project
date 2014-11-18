"""This module was made "from scratch" using the codes provided by James Lyons
at the url "github.com/jameslyons/python_speech_features". It includes routines
to calculate the MFCCs extraction.

Most part of the code is similar to the "inspiration". What I did was read his
code and copy what I understood. The idea is to do the same he did, using his
code as a guide.
"""


import numpy as np
from scipy.fftpack import dct

import sigproc


def hz2mel(hz):
    """Convert a value in Hertz to Mels

    :param hz: a value in Hz. This can also be a numpy array, conversion proceeds
    element-wise.

    :returns: a value in Mels. If an array was passed in, an identical sized array
    is returned.
    """
    return (2595 * np.log10(1 + hz/700.0))

def mel2hz(mel):
    """Convert a value in Mels to Hertz

    :param mel: a value in Mels. This can also be a numpy array, conversion
    proceeds element-wise.

    :returns: a value in Hertz. If an array was passed in, an identical sized
    array is returned.
    """
    return (700 * (10**(mel/2595.0) - 1))

def get_filterbanks(nfilt=26, nfft=512, samplerate=16000, lowfreq=0, highfreq=None):
    """Compute a Mel-filterbank. The filters are stored in the rows, the columns
    correspond to fft bins. The filters are returned as an array of size
    nfilt x (nfft/2 + 1)

    :param nfilt: the number of filters in the filterbank, default 26.
    :param nfft: the FFT size. Default is 512.
    :param samplerate: the samplerate of the signal we are working with. Affects
    mel spacing.
    :param lowfreq: lowest band edge of mel filters, default 0 Hz
    :param highfreq: highest band edge of mel filters, default samplerate/2

    :returns: A numpy array of size nfilt x (nfft/2 + 1) containing filterbank.
    Each row holds 1 filter.
    """
    if (not highfreq) or (highfreq > samplerate/2):
        highfreq = samplerate/2

    lowmel = hz2mel(lowfreq)
    highmel = hz2mel(highfreq)
    melpoints = np.linspace(lowmel, highmel, nfilt + 2)
    hzpoints = mel2hz(melpoints)
    bin = np.floor((nfft + 1) * hzpoints / samplerate)

    fbank = np.zeros([nfilt, nfft/2 + 1])
    for m in range(0, nfilt):
        bin_m = int(bin[m])         # f(m - 1)
        bin_m_1 = int(bin[m + 1])   # f(m)
        bin_m_2 = int(bin[m + 2])   # f(m + 1)

        # for (k < bin_m) and (k > bin_m_2), fbank[m, k] is already ZERO
        for k in range(bin_m, bin_m_1):
            fbank[m, k] = (k - bin[m]) / (bin[m + 1] - bin[m])
        for k in range(bin_m_1, bin_m_2):
            fbank[m, k] = (bin[m + 2] - k) / (bin[m + 2] - bin[m + 1])

    return fbank


# TEST
if __name__ == '__main__':
    import scipy.io.wavfile as wavf

    (rate, signal) = wavf.read("file.wav")
    print('signal:')
    print(signal)
    fbank = get_filterbanks(nfilt=10, lowfreq=300)
    print('fbank', len(fbank), 'x', len(fbank[0]))
    print(fbank)

    import matplotlib.pyplot as plt

    plt.grid(True)
    plt.plot(signal)
    plt.figure()
    plt.grid(True)
    for i in range(len(fbank)):
        plt.plot(fbank[i])

    plt.show()