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
    basepath_2 = '%s%s' % (BASES_DIR, 'mit-2')
    if os.path.exists(basepath_2):
        shutil.rmtree(basepath_2)
    os.mkdir(basepath_2)

    basepath = '%s%s' % (BASES_DIR, 'mit')
    basesets = os.listdir(basepath)

    for baseset in basesets:
        basesetpath_2 = '%s/%s' % (basepath_2, baseset)
        os.mkdir(basesetpath_2)

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
            speakerpath_2 = '%s/%s.wav' % (basesetpath_2, speaker)
            wavf.write(speakerpath_2, wavfile[0], wavfile[1])




def read_mit_base():
    """Reads the MIT base and returns a dictionary with the utterances in the
    directory's hierarchy.

    OBS: uses too much resources. The system becomes slow.
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
    #basedict = read_mit_base()
    #signal = basedict['enroll_1']['f00']['phrase01_16k.wav']
    #print(signal)

    preproc_mit_base()