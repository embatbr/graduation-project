"""This module was made "from scratch" using the codes provided by James Lyons
at the url "github.com/jameslyons/python_speech_features". It includes routines
for basic signal processing, such as framing and computing power spectra.

Most part of the code is similar to the "inspiration". What I did was read his
work, copy what I understood and improve some parts.
"""


import numpy as np
import math


def preemphasis(signal, coeff=0.97):
    """Performs preemphasis on the input signal.

    @param signal: The signal to filter.
    @param coeff: The preemphasis coefficient. 0 is no filter. Default is 0.97.

    @returns: the higher frequencies of signal.
    """
    return np.append(signal[0], signal[1 : ] - coeff*signal[ : -1])

def frame_signal(signal, frame_len, frame_step, winfunc=lambda x:np.hamming(x)):
    """Frames a signal into overlapping frames.

    @param signal: the audio signal to frame.
    @param frame_len: length of each frame measured in samples.
    @param frame_step: number of samples after the start of the previous frame
    that the next frame should begin (in samples).
    @param winfunc: the analysis window to apply to each frame. By default it's
    the Hamming window.

    @returns: an array of frames. Size is (NUMFRAMES x frame_len).
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

    # indices of samples in frames (0:0->320, 1:160->480, ...)
    indices = np.tile(np.arange(0, frame_len), (numframes, 1)) +\
              np.tile(np.arange(0, numframes*frame_step, frame_step),
                      (frame_len, 1)).T
    indices = indices.astype(np.int32, copy=False)
    frames = padsignal[indices]
    win = np.tile(winfunc(frame_len), (numframes, 1))

    return (frames*win)

def magspec(frames, NFFT=512):
    """Computes the magnitude spectrum of each frame in frames. If frames is an
    N*D matrix, output will be (N x (NFFT/2)).

    @param frames: the array of frames. Each row is a frame.
    @param NFFT: the FFT length to use. If NFFT > frame_len, the frames are
    zero-padded.

    @returns: If frames is an N*D matrix, output will be (N x (NFFT/2)). Each row will
    be the magnitude spectrum of the corresponding frame.
    """
    complex_spec = np.fft.rfft(frames, NFFT)    # the window is multiplied in frame_signal()
    return np.absolute(complex_spec)            # cuts half of the array off

def powspec(frames, NFFT=512):
    """Computes the power spectrum (periodogram estimate) of each frame in frames.
    If frames is an N*D matrix, output will be (N x (NFFT/2)).

    @param frames: the array of frames. Each row is a frame.
    @param NFFT: the FFT length to use. If NFFT > frame_len, the frames are
    zero-padded.

    @returns: If frames is an N*D matrix, output will be (N x (NFFT/2)). Each row will
    be the power spectrum of the corresponding frame.
    """
    return ((1.0/NFFT) * np.square(magspec(frames, NFFT=NFFT)))


# TEST
if __name__ == '__main__':
    import scipy.io.wavfile as wavf
    import matplotlib.pyplot as plt

    (samplerate, signal) = wavf.read('../bases/mit/corpuses/enroll_2/f08/phrase54_16k.wav')

    frame_len = 0.02*samplerate
    frame_step = 0.01*samplerate
    preemph = 0.97
    NFFT = 512

    print('signal:')
    print(signal)
    fig = plt.figure()
    plt.grid(True)
    plt.plot(signal) #figure 1
    fig.suptitle('signal')
    plt.xlabel('time (samples)')

    presignal = preemphasis(signal, coeff=preemph)
    print('preemphasis:')
    print(presignal)
    fig = plt.figure()
    plt.grid(True)
    plt.plot(presignal) #figure 2
    fig.suptitle('presignal')
    plt.xlabel('time (samples)')

    frames = frame_signal(presignal, frame_len, frame_step)
    print('frames', len(frames), 'x', len(frames[0]))
    print(frames)
    recovered = np.array(list())
    for frame in frames:
        recovered = np.concatenate((recovered, frame))
    fig = plt.figure()
    plt.grid(True)
    plt.plot(recovered) #figure 3
    fig.suptitle('frames')
    plt.xlabel('time (samples)')

    magsig = magspec(frames, NFFT)
    print('magsig', len(magsig), 'x', len(magsig[0]))
    print(magsig)
    recovered = np.array(list())
    for mag in magsig:
        recovered = np.concatenate((recovered, mag))
    fig = plt.figure()
    plt.grid(True)
    plt.plot(recovered) #figure 4
    fig.suptitle('magsig')
    plt.xlabel('frquency (Hz)')

    powsig = powspec(frames, NFFT)
    print('powsig', len(powsig), 'x', len(powsig[0]))
    print(powsig)
    for pwr in powsig:
        recovered = np.concatenate((recovered, pwr))
    fig = plt.figure()
    plt.grid(True)
    plt.plot(recovered) #figure 5
    fig.suptitle('powsig')
    plt.xlabel('frquency (Hz)')

    plt.show()