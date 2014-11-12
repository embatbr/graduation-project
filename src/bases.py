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


def base_to_dict(basename):
    """Reads a base and returns a dictionary with the hierarchy.
    """
    basepath = '%s%s' % (BASES_DIR, basename)
    basedict = dict()

    for baseset in os.listdir(basepath):
        basedict[baseset] = dict()

        speakers = os.listdir('%s/%s' % (basepath, baseset))
        speakers.sort()
        for speaker in speakers:
            utterances = os.listdir('%s/%s/%s' % (basepath, baseset, speaker))
            utterances.sort()
            utterances = [u for u in utterances if u.endswith('.wav')]
            basedict[baseset][speaker] = utterances

    return basedict

def read_base(basename):
    basedict = base_to_dict(basename)


#TEST
if __name__ == '__main__':
    if not os.path.exists('base.out'):
        os.mkdir('base.out')

    basename = 'mit'
    basedict = base_to_dict(basename)
    read_base(basename)

    import json
    basefile = open('base.out/%s.json' % basename, 'w')
    json.dump(basedict, basefile, indent=4, sort_keys=True)