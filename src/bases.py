"""This module contains basic structures to represent the utterances extracted
from any base used. The final representation is a Signal object, that represents
some fields from a .wav file (or an acoustic signal).
"""


import scipy.io.wavfile as wavf
import numpy as np

from defines import *


class Signal(object):
    """An acoustic signal, containing the sample rate and the samples.
    """
    def __init__(self, wave):
        """By default the .wav file uses a 16 bits integer. These data are
        converted to a 64 bits integer.
        """
        self.sample_rate = wave[0]
        self.samples = wave[1].astype(np.int64, copy=False)

    def __str__(self):
        ret = 'rate: %d\nlength: %d\nsamples: %s, dtype=%s' % (self.sample_rate,
              self.length(), self.samples, self.samples.dtype)
        return ret

    def length(self):
        """Number os samples.
        """
        return len(self.samples)

    def to_wavfile(self):
        """Converts the Signal object to a tuple returned by scipy.io.wavfile.read.
        """
        return (self.sample_rate, self.samples.astype(np.int16, copy=False))


#TODO ler o .json
def read_base():
    pass


#TEST
if __name__ == '__main__':
    filename = 'mit/enroll_1/f00/phrase01_16k.wav'
    wavfile = wavf.read('%s%s' % (BASES_DIR, filename))
    signal = Signal(wavfile)
    print(signal)
    wavfile2 = signal.to_wavfile()
    print(wavfile2)
