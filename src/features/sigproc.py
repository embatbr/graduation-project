"""This module was made "from scratch" using the codes provided by James Lyons
at the url "github.com/jameslyons/python_speech_features". It includes routines
for basic signal processing, such as framing and computing power spectra.

Most part of the code is similar to the "inspiration". What I did was read his
code and copy what I understood. The idea is to do the same he did, using his
code as a guide.
"""


import numpy as np
import math


def preemphasis(signal, coeff=0.95):
    """Perform preemphasis on the input signal.

    :param signal: The signal to filter.
    :param coeff: The preemphasis coefficient. 0 is no filter, default is 0.95.

    :returns: the filtered signal.
    """
    return np.append(signal[0], signal[1 : ] - coeff*signal[ : -1])

def frame_signal(signal, frame_len, frame_step, winfunc=lambda x:np.hamming(x)):
    """Frame a signal into overlapping frames.

    :param signal: the audio signal to frame.
    :param frame_len: length of each frame measured in samples.
    :param frame_step: number of samples after the start of the previous frame
    that the next frame should begin (in samples).
    :param winfunc: the analysis window to apply to each frame. By default no
    window is applied (signal is multiplied by 1).

    :returns: an array of frames. Size is NUMFRAMES x frame_len.
    """
    signal_len = len(signal)
    frame_len = int(round(frame_len))
    frame_step = int(round(frame_step))
    if signal_len <= frame_len:
        numframes = 1
    else:
        num_additional_frames = float(signal_len - frame_len) / frame_step
        num_additional_frames = int(math.ceil(num_additional_frames))
        numframes = 1 + num_additional_frames

    padsignal_len = (numframes - 1)*frame_step + frame_len
    zeros = np.zeros((padsignal_len - signal_len))
    padsignal = np.concatenate((signal, zeros))  # addition of zeros at the end

    # indices of samples in frames (0:0->399, 1:160->559, ..., n:26880->27279)
    indices = np.tile(np.arange(0, frame_len), (numframes, 1)) +\
              np.tile(np.arange(0, numframes*frame_step, frame_step),
                      (frame_len, 1)).T
    indices = indices.astype(np.int32, copy=False)
    frames = padsignal[indices]
    win = np.tile(winfunc(frame_len), (numframes, 1))

    return (frames*win)

def magspec(frames, NFFT):
    """Compute the magnitude spectrum of each frame in frames. If frames is an
    NxD matrix, output will be NxNFFT.

    :param frames: the array of frames. Each row is a frame.
    :param NFFT: the FFT length to use. If NFFT > frame_len, the frames are
    zero-padded.

    :returns: If frames is an NxD matrix, output will be NxNFFT. Each row will
    be the magnitude spectrum of the corresponding frame.
    """
    complex_spec = np.fft.rfft(frames, NFFT)    # the window is multiplied in frame_signal()
    return np.absolute(complex_spec)

def powspec(frames, NFFT):
    """Compute the power spectrum (periodogram estimate) of each frame in frames.
    If frames is an NxD matrix, output will be NxNFFT.

    :param frames: the array of frames. Each row is a frame.
    :param NFFT: the FFT length to use. If NFFT > frame_len, the frames are
    zero-padded.

    :returns: If frames is an NxD matrix, output will be NxNFFT. Each row will
    be the power spectrum of the corresponding frame.
    """
    return ((1.0/NFFT) * np.square(magspec(frames, NFFT)))


# TEST
if __name__ == '__main__':
    import scipy.io.wavfile as wavf

    (samplerate, signal) = wavf.read("file.wav")
    print('signal:')
    print(signal)
    signal = preemphasis(signal)
    print('preemphasis:')
    print(signal)
    framedsig = frame_signal(signal, 0.025*16000, 0.01*16000)
    print('framedsig', len(framedsig), 'x', len(framedsig[0]))
    print(framedsig)
    magsig = magspec(framedsig, 512)
    print('magsig', len(magsig), 'x', len(magsig[0]))
    print(magsig)
    powsig = powspec(framedsig, 512)
    print('powsig', len(powsig), 'x', len(powsig[0]))
    print(powsig)

    import matplotlib.pyplot as plt

    plt.grid(True)
    plt.plot(wavf.read("file.wav")[1])
    plt.plot(signal)
    plt.figure()
    plt.grid(True)
    plt.plot(magsig[50])
    plt.plot(powsig[50])
    plt.figure()
    plt.grid(True)
    plt.plot(magsig)
    plt.plot(powsig)

    plt.show()