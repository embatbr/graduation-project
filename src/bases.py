"""This module contains basic structures to represent the utterances extracted
from any base used. The final representation is a Signal object, equivalent to
some fields from a .wav file (an acoustic signal).
"""


import os, os.path

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


def read_base_mit():
    """Reads a base and returns a dictionary with the utterances in hierarchy.
    """
    basepath = '%s%s' % (BASES_DIR, 'mit')
    basedict = dict()

    for baseset in os.listdir(basepath):
        basedict[baseset] = dict()
        basesetpath = '%s/%s' % (basepath, baseset)
        speakers = os.listdir(basesetpath)

        for speaker in speakers:
            basedict[baseset][speaker] = dict()
            speakerpath = '%s/%s' % (basesetpath, speaker)
            uttnames = os.listdir(speakerpath)

            for uttname in uttnames:
                if uttname.endswith('.wav'):
                    uttpath = '%s/%s' % (speakerpath, uttname)
                    wave = wavf.read(uttpath)
                    signal = Signal(wave)
                    basedict[baseset][speaker][uttname] = signal

    return basedict


#TEST
if __name__ == '__main__':
    basedict = read_base_mit()
    print(basedict)