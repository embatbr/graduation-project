"""This module was made "from scratch" using the codes provided by James Lyons
at the url "github.com/jameslyons/python_speech_features". Most part of the code
is similar to the "inspiration". I just read his work, copy what I understood and
improved some parts.

It includes routines for basic signal processing, such as framing and computing
magnitude and squared magnitude spectrum of framed signal.
"""


import numpy as np
import math


def preemphasis(signal, preemph=0.97):
    """Performs pre emphasis on the input signal.

    @param signal: The signal to filter.
    @param preemph: The pre emphasis coefficient. 0 is no filter. Default is 0.97.

    @returns: the highpass filtered signal.
    """
    return np.append(signal[0], signal[1 : ] - preemph*signal[ : -1])

def framing(signal, framelen, framestep, winfunc=lambda x:np.hamming(x)):
    """Divides a signal into overlapping frames.

    @param signal: the audio signal to frame.
    @param framelen: length of each frame measured in samples.
    @param framestep: number of samples after the start of the previous frame
    that the next frame should begin (in samples).
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
    from useful import CORPORA_DIR, TEST_IMAGES_DIR, plotfile


    if not os.path.exists(TEST_IMAGES_DIR):
        os.mkdir(TEST_IMAGES_DIR)

    IMAGES_SIGPROC_DIR = '%ssigproc/' % TEST_IMAGES_DIR

    if os.path.exists(IMAGES_SIGPROC_DIR):
            shutil.rmtree(IMAGES_SIGPROC_DIR)
    os.mkdir(IMAGES_SIGPROC_DIR)

    filecounter = 0
    filename = '%sfigure' % IMAGES_SIGPROC_DIR

    #Reading signal from base and plotting
    voice = ('enroll_2', 'f08', 54)
    (enroll, speaker, speech) = voice
    (samplerate, signal) = wavf.read('%smit/%s/%s/phrase%02d_16k.wav' %
                                     (CORPORA_DIR, enroll, speaker, speech))
    numsamples = len(signal)
    time = np.linspace(0, numsamples/samplerate, numsamples, False)
    ###figure000
    filecounter = plotfile(time, signal, '%s\n%d Hz' % (voice, samplerate),
                           't (seconds)', 'signal[t]', filename, filecounter)

    NFFT = 512
    numfftbins = math.floor(NFFT/2 + 1)    #fft bins == 'caixas' de FFT
    freq = np.linspace(0, samplerate/2, numfftbins)

    #Magnitude of signal's spectrum
    magsig = magspec(signal, NFFT)
    ###figure001
    filecounter = plotfile(freq, magsig, '%s\n%d Hz, |FFT|' % (voice, samplerate),
                           'f (Hz)', '|FFT[f]|', filename, filecounter, 'red', True)

    #Squared magnitude of signal's spectrum
    powsig = powspec(signal, NFFT)
    ###figure002
    filecounter = plotfile(freq, powsig, '%s\n%d Hz, |FFT|²' % (voice, samplerate),
                           'f (Hz)', '|FFT[f]|²', filename, filecounter, 'red', True)

    #Pre emphasized signal with coefficient 0.97
    presignal = preemphasis(signal)
    ###figure003
    filecounter = plotfile(time, presignal, '%s\n%d Hz, preemph 0.97' % (voice,
                           samplerate), 't (seconds)', 'presignal[t]', filename,
                           filecounter)

    #Inteisities of lower frequences reduced and of higher, increased
    #wavf.write('%s-%s-%02d-%dHz-preemph0.97.wav' % (enroll, speaker, speech, samplerate),
    #           samplerate, np.int16(presignal))

    #Magnitude of presignal's spectrum
    magpresig = magspec(presignal, NFFT)
    ###figure004
    filecounter = plotfile(freq, magpresig, '%s\n%d Hz, preemph 0.97, |FFT|' %
                           (voice, samplerate), 'f (Hz)', '|FFT[f]|', filename,
                           filecounter, 'red', True)

    #Squared magnitude of presignal's spectrum
    powpresig = powspec(presignal, NFFT)
    ###figure005
    filecounter = plotfile(freq, powpresig, '%s\n%d Hz, preemph 0.97, |FFT|²' %
                           (voice, samplerate), 'f (Hz)', '|FFT[f]|²', filename,
                           filecounter, 'red', True)

    #samples = sec * (samples/sec)
    framelen = 0.02
    framestep = 0.01

    #Framing with rectangular window
    frames = framing(presignal, framelen*samplerate, framestep*samplerate,
                     winfunc=lambda x:np.ones((1, x)))
    magframes = magspec(frames, NFFT)
    powframes = powspec(frames, NFFT)
    numframes = len(frames)
    print('Framing with rectangular window\n#frames = %d' % numframes)
    for i in range(0, numframes):
        frametime = np.linspace(i*framestep, (i*framestep + framelen),
                                framelen*samplerate, False)
        #Framed pre emphasized signal using a Rect window
        filecounter = plotfile(frametime, frames[i], '%s\n%d Hz, preemph 0.97, Rect #%d' %
                               (voice, samplerate, i), 't (seconds)',
                               'presignal[t] * rect', filename, filecounter)

        #Magnitude of the framed spectrum
        filecounter = plotfile(freq, magframes[i], '%s\n%d Hz, preemph 0.97, |FFT(Rect #%d)|' %
                               (voice, samplerate, i), 'f (Hz)', '|FFT[f]|',
                               filename, filecounter, 'red', True)

        #Squared magnitude of the framed spectrum
        filecounter = plotfile(freq, powframes[i], '%s\n%d Hz, preemph 0.97, |FFT(Rect #%d)|²' %
                               (voice, samplerate, i), 'f (Hz)', '|FFT[f]|²',
                               filename, filecounter, 'red', True)

    #Framing with Hamming window
    frames = framing(presignal, framelen*samplerate, framestep*samplerate)
    magframes = magspec(frames, NFFT)
    powframes = powspec(frames, NFFT)
    numframes = len(frames)
    print('Framing with Hamming window\n#frames = %d' % numframes)
    for i in range(0, numframes):
        frametime = np.linspace(i*framestep, (i*framestep + framelen),
                                framelen*samplerate, False)
        #Framed pre emphasized signal using a Hamming window
        filecounter = plotfile(frametime, frames[i], '%s\n%d Hz, preemph 0.97, Hamming #%d' %
                               (voice, samplerate, i), 't (seconds)',
                               'presignal[t] * hamming', filename, filecounter)

        #Magnitude of the framed spectrum
        filecounter = plotfile(freq, magframes[i], '%s\n%d Hz, preemph 0.97, |FFT(Hamming #%d)|' %
                               (voice, samplerate, i), 'f (Hz)', '|FFT[f]|',
                               filename, filecounter, 'red', True)

        #Squared magnitude of the framed spectrum
        filecounter = plotfile(freq, powframes[i], '%s\n%d Hz, preemph 0.97, |FFT(Hamming #%d)|²' %
                               (voice, samplerate, i), 'f (Hz)', '|FFT[f]|²',
                               filename, filecounter, 'red', True)