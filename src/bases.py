"""This module contains basic structures to represent the utterances extracted
from any base used. The final representation is a Signal object, equivalent to
some fields from a .wav file (an acoustic signal).
"""


import os, os.path
import shutil

import scipy.io.wavfile as wavf
import numpy as np


BASES_DIR = '../bases/'


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


def preproc_mit_base():
    """Performs the pre-processing of the MIT base, removing silences and noise
    (if needed), and joining the utterances of each speaker in one.
    """
    preproc_basepath = '%s%s' % (BASES_DIR, 'mit-preproc')
    if os.path.exists(preproc_basepath):
        shutil.rmtree(preproc_basepath)
    os.mkdir(preproc_basepath)

    basepath = '%s%s' % (BASES_DIR, 'mit')
    basesets = os.listdir(basepath)

    for baseset in basesets:
        preproc_basesetpath = '%s/%s' % (preproc_basepath, baseset)
        os.mkdir(preproc_basesetpath)

        basesetpath = '%s/%s' % (basepath, baseset)
        speakers = os.listdir(basesetpath)

        for speaker in speakers:
            speakerpath = '%s/%s' % (basesetpath, speaker)
            uttnames = os.listdir(speakerpath)
            uttnames.sort()

            waves = list()
            sample_rate = 20000
            for uttname in uttnames:
                if uttname.endswith('.wav'):
                    uttpath = '%s/%s' % (speakerpath, uttname)
                    wave = wavf.read(uttpath)
                    #if necessary, the VAD goes here
                    sample_rate = wave[0]
                    waves.append(wave[1])

            signal = Signal((sample_rate, np.concatenate(waves)))
            wavfile = signal.to_wavfile()
            preproc_speakerpath = '%s/%s.wav' % (preproc_basesetpath, speaker)
            wavf.write(preproc_speakerpath, wavfile[0], wavfile[1])


#TEST
if __name__ == '__main__':
    preproc_mit_base()