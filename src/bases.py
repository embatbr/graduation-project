"""This module contains basic structures to represent the utterances extracted
from any base used. The final representation is a Signal object, that represents
some fields from a .wav file (or an acoustic signal).
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


def dir_to_dict(filepath):
    """A recursive function to turn a directory tree into a dictionary.
    """
    if os.path.isfile(filepath) and (not os.path.islink(filepath)):
        filename = os.path.basename(filepath)
        if filename.endswith('.txt'):
            return None
        return filename

    subdirs = os.listdir(filepath)
    subdirs.sort()
    filename = os.path.basename(filepath)
    filedict = {filename : list()}

    for subdir in subdirs:
        subdict = dir_to_dict('%s/%s' % (filepath, subdir))
        if subdict is not None:
            filedict[filename].append(subdict)

    return filedict

def base_to_dict(basename):
    """Reads a base of utterances names and returns a dictionary containing
    the hierarchy.
    """
    basepath = '%s%s' % (BASES_DIR, basename)
    return dir_to_dict(basepath)


#TEST
if __name__ == '__main__':
    basename = 'mit'
    base = base_to_dict(basename)
    basefile = open('%s.json' % basename, 'w')

    import json
    json.dump(base, basefile, indent=4)