"""This module was made "from scratch" using the codes provided by James Lyons
at the url "github.com/jameslyons/python_speech_features". Most part of the code
is similar to the "inspiration". I just read his work, copy what I understood and
improved some parts.

It includes routines for basic signal processing, such as framing and computing
magnitude and squared magnitude of spectrum of framed signal.
"""


import numpy as np
import math


def preemphasis(signal, preemph=0.97):
    """Performs pre emphasis on the input signal. The formula for the pre emphasized
    signal y is y[n] = x[n] - preemph*x[n-1].

    @param signal: The signal to filter.
    @param preemph: The pre emphasis coefficient. 0 is no filter. Default is 0.97.

    @returns: the filtered signal.
    """
    return np.append(signal[0], signal[1 : ] - preemph*signal[ : -1])

def framing(signal, framelen=320, framestep=160, winfunc=lambda x:np.hamming(x)):
    """Divides a signal into overlapping frames.

    @param signal: the audio signal to divide in frames.
    @param framelen: length of each frame measured in samples.
    @param framestep: number of samples after the start of the previous frame
    that the next frame should begin.
    @param winfunc: the analysis window to apply to each frame. By default it's
    the rectangular window.

    @returns: an array of frames. Size is (NUMFRAMES x framelen).
    """
    signal_len = len(signal)
    framelen = int(round(framelen))
    framestep = int(round(framestep))
    if signal_len <= framelen:
        numframes = 1
    else:
        num_additional_frames = float(signal_len - framelen) / framestep
        num_additional_frames = int(math.ceil(num_additional_frames))
        numframes = 1 + num_additional_frames

    padsignal_len = (numframes - 1)*framestep + framelen
    zeros = np.zeros((padsignal_len - signal_len))
    padsignal = np.concatenate((signal, zeros))  # addition of zeros at the end

    # indices of samples in frames (0:0->320, 1:160->480, ...)
    indices = np.tile(np.arange(0, framelen), (numframes, 1)) +\
              np.tile(np.arange(0, numframes*framestep, framestep), (framelen, 1)).T
    indices = indices.astype(np.int32, copy=False)
    frames = padsignal[indices]
    win = np.tile(winfunc(framelen), (numframes, 1))

    return (frames*win)

def magspec(frames, NFFT=512):
    """Computes the magnitude spectrum of each frame in frames. If frames is an
    N*D matrix, output will be (N x (NFFT/2)). Can be used in a one dimensional
    numpy array.

    @param frames: the array of frames. Each row is a frame.
    @param NFFT: the FFT length to use. If NFFT > framelen, the frames are
    zero-padded.

    @returns: If frames is an N*D matrix, output will be (N x (NFFT/2)). Each row will
    be the magnitude spectrum of the corresponding frame.
    """
    complex_spec = np.fft.rfft(frames, NFFT)
    return np.absolute(complex_spec)    # from a + jb to |z| (cuts in half due to simmetry)

def powspec(frames, NFFT=512):
    """Computes the power spectrum (periodogram estimate) of each frame in frames.
    If frames is an N*D matrix, output will be (N x (NFFT/2)). Can be used in a
    one dimensional numpy array.

    @param frames: the array of frames. Each row is a frame.
    @param NFFT: the FFT length to use. If NFFT > framelen, the frames are
    zero-padded.

    @returns: If frames is an N*D matrix, output will be (N x (NFFT/2)). Each row will
    be the power spectrum of the corresponding frame.
    """
    magframes = magspec(frames, NFFT)
    return ((1.0/NFFT) * np.square(magframes))


#TESTS
if __name__ == '__main__':
    import scipy.io.wavfile as wavf
    import os, os.path, shutil
    from useful import CORPORA_DIR, TESTS_DIR, plotfigure
    import pylab as pl

    # Reading signal from base

    (samplerate, signal) = wavf.read('%smit/enroll_1/f00/phrase01_16k.wav' %
                                     CORPORA_DIR)
    numsamples = len(signal)
    duration = numsamples/samplerate
    time = np.linspace(1/samplerate, duration, numsamples)

    # Plotting signal and 0.97 preemph signal, frames, magspec and powspec
    for sig in [signal, preemphasis(signal)]:
        pl.figure()

        pl.subplot(321)
        pl.plot(time, sig, 'b')
        magsig = magspec(sig)
        pl.subplot(323)
        pl.plot(magsig, 'r')
        powsig = powspec(sig)
        pl.subplot(325)
        pl.plot(powsig, 'g')

        frames = framing(sig)
        pl.subplot(322)
        pl.plot(frames, 'b')
        mag_frames = magspec(frames)
        pl.subplot(324)
        for mag_frame in mag_frames:
            pl.plot(mag_frame, 'r')
        pow_frames = powspec(frames)
        pl.subplot(326)
        for pow_frame in pow_frames:
            pl.plot(pow_frame, 'g')

    pl.show()