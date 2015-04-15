"""Contains functions to extract MFCCs from a speech signal.

This module was made "from scratch" using the codes provided by James Lyons
at the url "github.com/jameslyons/python_speech_features". Most part of the code
is similar to the "inspiration" (aka it's the same). I just read his work, copied
what I understood and improved some parts.
"""


import numpy as np
import math
from scipy.fftpack import dct


def preemphasis(signal, preemph=0.97):
    """Performs pre-emphasis on the input signal. The formula for the pre emphasized
    signal y is y[n] = x[n] - preemph*x[n-1].

    @param signal: The signal to filter.
    @param preemph: The pre-emphasis coefficient. 0 is no filter. Default is 0.97.

    @returns: the signal with the high frequencies emphasized.
    """
    return np.append(signal[0], signal[1 : ] - preemph*signal[ : -1])

def framesignal(signal, framelen, framestep):
    """Divides a signal into overlapping frames, using the Hamming window.

    @param signal: the audio signal to frame.
    @param framelen: length of each frame in samples.
    @param framestep: frame shift in samples.

    @returns: an array of frames of size NUMFRAMES x framelen.
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
    padsignal = np.append(signal, zeros)  # addition of zeros at the end

    # indices of samples in frames (0:0->framelen, 1:framestep->(framelen + framestep), ...)
    # the 'tile' usage is really smart, but difficult to get at first
    indices = np.tile(np.arange(0, framelen), (numframes, 1)) +\
              np.tile(np.arange(0, numframes*framestep, framestep), (framelen, 1)).T
    indices = indices.astype(np.int32, copy=False)
    frames = padsignal[indices]
    window = np.tile(np.hamming(framelen), (numframes, 1))

    return frames*window

def magspec(frames, NFFT=512):
    """Computes the magnitude spectrum of each frame in frames. If frames is an
    N*D matrix, output will be (N x (NFFT/2)). np.fft.rfft computes the FFT on
    the most internal axis.

    @param frames: the array of frames. Each row is a frame.
    @param NFFT: the FFT length to use. If NFFT > len(frames[k]), the frames are
    zero-padded. Else, if NFFT < len(frames[k]), frames[k] are cropped.

    @returns: If frames is an N*D matrix, output will be (N x (NFFT/2)). Each row
    will be the magnitude spectrum of the corresponding frame.
    """
    complex_spec = np.fft.rfft(frames, NFFT)
    return np.absolute(complex_spec)    # from a + jb to |z| (cuts in half due to simmetry)

def powspec(frames, NFFT=512):
    """Computes the power spectrum (periodogram estimate) of each frame from frames.
    If frames is an N*D matrix, output will be (N x (NFFT/2)).

    @param frames: the array of frames. Each row is a frame.
    @param NFFT: the FFT length to use. If NFFT > len(frames[k]), the frames are
    zero-padded. Else, if NFFT < len(frames[k]), frames[k] are cropped.

    @returns: If frames is an N*D matrix, output will be (N x (NFFT/2)). Each row
    will be the power spectrum of the corresponding frame.
    """
    magframes = magspec(frames, NFFT)
    return ((1.0/NFFT) * np.square(magframes))

def hz2mel(hz):
    """Converts a value in Hertz to Mels

    @param hz: a value in Hz. This can also be a numpy array, conversion proceeds
    element-wise.

    @returns: a value in Mels. If an array was passed in, an identical sized array
    is returned.
    """
    return (2595 * np.log10(1 + hz/700))

def mel2hz(mel):
    """Converts a value in Mels to Hertz

    @param mel: a value in Mels. This can also be a numpy array, conversion
    proceeds element-wise.

    @returns: a value in Hertz. If an array was passed in, an identical sized
    array is returned.
    """
    return (700 * (10**(mel/2595) - 1))

def filterbank(samplerate, nfilt=26, NFFT=512):
    """Creates an filterbank in the mel scale. The filters are stored in the rows,
    the columns correspond to fft bins. The filters are returned as an array of
    size nfilt x (NFFT/2 + 1).

    @param samplerate: the samplerate of the signal we are working with. Affects
    mel spacing.
    @param nfilt: the number of filters in the filterbank. Default 26.
    @param NFFT: the FFT size. Default 512.

    @returns: A numpy array of size (nfilt x (floor(NFFT/2) + 1)) containing the
    filterbank. Each row is one filter.
    """
    lowfreq_mel = 0
    highfreq_mel = hz2mel(samplerate/2)
    melpoints = np.linspace(lowfreq_mel, highfreq_mel, nfilt + 2) # equally spaced in mel scale
    hzpoints = mel2hz(melpoints)
    bin = np.floor((NFFT + 1) * hzpoints / samplerate)  #'bin', from FFT bin = 'caixa' de FFT

    fbank = np.zeros((nfilt, math.floor(NFFT/2 + 1)))
    for m in range(1, nfilt + 1):
        f_m_minus_1 = int(bin[m - 1])   # f(m - 1)
        f_m = int(bin[m])               # f(m)
        f_m_plus_1 = int(bin[m + 1])    # f(m + 1)

        # for (k < f_m_minus_1) and (k > f_m_plus_1), fbank[m, k] is already ZERO
        for k in range(f_m_minus_1, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus_1):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])

    return fbank

def lifter(cepstra, L=22):
    """Apply a lifter (filter for cepstrum) to the matrix of cepstra. This has
    the effect of increase the magnitude of the high frequency DCT coeffs.

    @param cepstra: the matrix of mel-cepstra, with size numframes*numceps.
    @param L: the liftering coefficient to use. Default is 22. L <= 0 disables
    lifter.

    @return: the liftered cepstra.
    """
    if L > 0:
        (nframes, ncoeff) = np.shape(cepstra)
        n = np.arange(ncoeff)
        lift = 1 + (L/2)*np.sin(np.pi * n / L)
        return (lift*cepstra)
    else:
        return cepstra

def append_deltas(featsvec, numceps, N=2):
    """Calculates and appends the deltas for the last 'numceps' features from
    frames 'featsvec[:]'. OBS: this method only calculates 1st order deltas. To
    higher orders, use it recursively.

    @param featsvec: the original features.
    @param numceps: the number of cepstral coefficients (added at the end of each
    frame 'featsvec[:]').
    @param N: complexity of delta. Default 2.

    @returns: a new array containing the original features with deltas appended.
    """
    numfeats = len(featsvec[0, :])
    numframes = len(featsvec)
    new_feats = np.zeros((numframes, numfeats + numceps))
    new_feats[:, : numfeats] = featsvec[:,:]    #copy old features

    denom = 2 * sum([n*n for n in range(1, N + 1)])
    for t in range(numframes):
        delta = np.zeros(numceps)
        for n in range(1, N + 1):
            after = featsvec[t + n, numfeats - numceps :] if ((t + n) < numframes) else 0
            before = featsvec[t - n, numfeats - numceps :] if ((t - n) >= 0) else 0

            delta = delta + n*(after - before)

        new_feats[t, numfeats :] = delta / denom

    return new_feats

def mfcc(signal, winlen, winstep, samplerate, nfilt=26, NFFT=512, preemph=0.97,
         numceps=19, ceplifter=22, append_energy=True, applyCMS=True, delta_order=0, N=2):
    """Extracts features from an audio signal using the MFCC algorithm.

    @param signal: the audio signal from which to extract the features. Should
    be an N*1 array
    @param winlen: the length of the analysis window in seconds.
    @param winstep: the step between successive windows in seconds.
    @param samplerate: the samplerate of the signal we are working with.
    @param nfilt: the number of filters in the filterbank. Default 26.
    @param NFFT: the FFT size. Default is 512.
    @param preemph: apply preemphasis filter with preemph as coefficient. 0 is
    no filter. Default is 0.97.
    @param numceps: the number of cepstrum to return, default 13
    @param ceplifter: apply a lifter to final cepstral coefficients. 0 is no
    lifter. Default is 22.
    @param append_energy: if this is true, the zeroth cepstral coefficient is
    replaced by the log of the frame energy.
    @param applyCMS: if True (default), applies the Cepstral Mean Subtraction.
    @param delta_order: the number of delta calculations. Default 0 (no delta).
    @param N: complexity of delta. Default 2.

    @returns: A numpy array of size NUMFRAMES x numfeats containing features.
    Each row holds a numfeats-dimensional vector and each column holds 1 feature
    over time, where numfeats = (1 + delta_order)*numceps. Ex:

    |f_1_1 f_1_2 ... f_1_c|
    |f_2_1 f_2_2 ... f_2_c|
    |...                  |
    |f_T_1 f_T_2 ... f_T_c|

    where 'c' is the number of features and 'T' the number of frames.
    """
    emph_signal = preemphasis(signal, preemph)
    frames = framesignal(emph_signal, winlen*samplerate, winstep*samplerate)
    powframes = powspec(frames, NFFT)
    fbank = filterbank(samplerate, nfilt, NFFT)

    featsvec = np.dot(powframes, fbank.T)
    featsvec = 20*np.log10(featsvec) #dB
    featsvec = dct(featsvec, type=2, axis=1, norm='ortho')[ : , : numceps]
    featsvec = lifter(featsvec, ceplifter)

    if append_energy:
        energy = np.sum(powframes, axis=1) # stores the total energy of each frame
        energy = 20*np.log10(energy) #dB
        featsvec[ : , 0] = energy

    if applyCMS:
        featsvec = featsvec - np.mean(featsvec, axis=0) # CMS reduces the effect of noise

    while delta_order > 0:
        featsvec = append_deltas(featsvec, numceps, N)
        delta_order = delta_order - 1

    return featsvec