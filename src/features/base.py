"""This module was made "from scratch" using the codes provided by James Lyons
at the url "github.com/jameslyons/python_speech_features". It includes routines
to calculate the MFCCs extraction.

Most part of the code is similar to the "inspiration". What I did was read his
code and copy what I understood. The idea is to do the same he did, using his
code as a guide.
"""


import numpy as np
from scipy.fftpack import dct

import sigproc


def hz2mel(hz):
    """Convert a value in Hertz to Mels

    @param hz: a value in Hz. This can also be a numpy array, conversion proceeds
    element-wise.

    @returns: a value in Mels. If an array was passed in, an identical sized array
    is returned.
    """
    return (2595 * np.log10(1 + hz/700.0))

def mel2hz(mel):
    """Convert a value in Mels to Hertz

    @param mel: a value in Mels. This can also be a numpy array, conversion
    proceeds element-wise.

    @returns: a value in Hertz. If an array was passed in, an identical sized
    array is returned.
    """
    return (700 * (10**(mel/2595.0) - 1))

def filterbanks(nfilt=26, nfft=512, samplerate=16000, lowfreq=0, highfreq=None):
    """Compute a Mel-filterbank. The filters are stored in the rows, the columns
    correspond to fft bins. The filters are returned as an array of size
    nfilt x (nfft/2 + 1)

    @param nfilt: the number of filters in the filterbank, default 26.
    @param nfft: the FFT size. Default is 512.
    @param samplerate: the samplerate of the signal we are working with. Affects
    mel spacing.
    @param lowfreq: lowest band edge of mel filters, default 0 Hz
    @param highfreq: highest band edge of mel filters, default samplerate/2

    @returns: A numpy array of size nfilt*(nfft/2 + 1) containing filterbank.
    Each row holds 1 filter.
    """
    if (not highfreq) or (highfreq > samplerate/2):
        highfreq = samplerate/2

    lowmel = hz2mel(lowfreq)
    highmel = hz2mel(highfreq)
    melpoints = np.linspace(lowmel, highmel, nfilt + 2)
    hzpoints = mel2hz(melpoints)
    bin = np.floor((nfft + 1) * hzpoints / samplerate)

    fbank = np.zeros([nfilt, nfft/2 + 1])
    for m in range(0, nfilt):
        bin_m = int(bin[m])         # f(m - 1)
        bin_m_1 = int(bin[m + 1])   # f(m)
        bin_m_2 = int(bin[m + 2])   # f(m + 1)

        # for (k < bin_m) and (k > bin_m_2), fbank[m, k] is already ZERO
        for k in range(bin_m, bin_m_1):
            fbank[m, k] = (k - bin[m]) / (bin[m + 1] - bin[m])
        for k in range(bin_m_1, bin_m_2):
            fbank[m, k] = (bin[m + 2] - k) / (bin[m + 2] - bin[m + 1])

    return fbank

def filterbank_signal(signal, samplerate=16000, winlen=0.025, winstep=0.01,
                        nfilt=26, nfft=512, lowfreq=0, highfreq=None, preemph=0.97):
    """Compute Mel-filterbank energy features from an audio signal.

    @param signal: the audio signal from which to compute features. Should be an
    N*1 array
    @param samplerate: the samplerate of the signal we are working with.
    @param winlen: the length of the analysis window in seconds. Default is 0.025s
    (25 milliseconds)
    @param winstep: the step between seccessive windows in seconds. Default is
    0.01s (10 milliseconds)
    @param nfilt: the number of filters in the filterbank, default 26.
    @param nfft: the FFT size. Default is 512.
    @param lowfreq: lowest band edge of mel filters. In Hz, default is 0.
    @param highfreq: highest band edge of mel filters. In Hz, default is samplerate/2
    @param preemph: apply preemphasis filter with preemph as coefficient. 0 is no
    filter. Default is 0.97.

    @returns: 2 values. The first is a numpy array of size (NUMFRAMES*nfilt)
    containing features. Each row holds 1 feature vector. The second is the energy
    in each frame (total energy, unwindowed)
    """
    signal = sigproc.preemphasis(signal, preemph)
    frames = sigproc.frame_signal(signal, winlen*samplerate, winstep*samplerate)
    pspec = sigproc.powspec(frames, nfft)

    signal_fb = filterbanks(nfilt, nfft, samplerate, lowfreq, highfreq)
    feat = np.dot(pspec, signal_fb.T)    # computes the filterbank energies
    energy = np.sum(pspec, 1)            # this stores the total energy in each frame

    return (feat, energy)

def lifter(cepstra, L=22):
    """Apply a cepstral lifter to the matrix of cepstra. This has the effect of
    increasing the magnitude of the high frequency DCT coeffs.

    @param cepstra: the matrix of mel-cepstra, will be numframes*numcep in size.
    @param L: the liftering coefficient to use. Default is 22. L <= 0 disables
    lifter.
    """
    if L > 0:
        (nframes, ncoeff) = np.shape(cepstra)
        n = np.arange(ncoeff)
        lift = 1 + (L/2)*np.sin(np.pi * n / L)
        return (lift*cepstra)
    else:
        # values of L <= 0, do nothing
        return cepstra

def mfcc(signal, samplerate=16000, winlen=0.025, winstep=0.01, numcep=13, nfilt=26,
         nfft=512, lowfreq=0, highfreq=None, preemph=0.97, ceplifter=22, appendEnergy=True):
    """Compute MFCC features from an audio signal.

    @param signal: the audio signal from which to compute features. Should be an
    N*1 array
    @param samplerate: the samplerate of the signal we are working with.
    @param winlen: the length of the analysis window in seconds. Default is
    0.025s (25 milliseconds)
    @param winstep: the step between successive windows in seconds. Default is
    0.01s (10 milliseconds)
    @param numcep: the number of cepstrum to return, default 13
    @param nfilt: the number of filters in the filterbank, default 26.
    @param nfft: the FFT size. Default is 512.
    @param lowfreq: lowest band edge of mel filters. In Hz, default is 0.
    @param highfreq: highest band edge of mel filters. In Hz, default is samplerate/2
    @param preemph: apply preemphasis filter with preemph as coefficient. 0 is
    no filter. Default is 0.97.
    @param ceplifter: apply a lifter to final cepstral coefficients. 0 is no
    lifter. Default is 22.
    @param appendEnergy: if this is true, the zeroth cepstral coefficient is
    replaced with the log of the total frame energy.

    @returns: A numpy array of size (NUMFRAMES*numcep) containing features.
    Each row holds 1 feature vector.
    """
    (feat, energy) = filterbank_signal(signal, samplerate, winlen, winstep, nfilt,
                                       nfft, lowfreq, highfreq, preemph)
    feat = np.log(feat)
    feat = dct(feat, type=2, axis=1, norm='ortho')[ : , : numcep]
    feat = lifter(feat, ceplifter)
    if appendEnergy:
        # replace first cepstral coefficient with log of frame energy
        feat[ : , 0] = np.log(energy)

    return feat

def delta(coeffs, t, numframes, N, denom, begin, end):
    """Calculates the delta from the mfcc coefficients.

    @param coeffs: coefficients of order 0 (mfccs) or higher.
    @param t: index of frame which delta is calculated.
    @param numframes: number of frames.
    @param N: complexity of delta (by default, 2).
    @param denom: 2 * (sum_1^N n*n).
    @param begin: beginning of coeffs[t]. begin = order*numcep.
    @param end: ending of coeffs[t]. end = (order + 1)*numcep.

    returns: the deltas from coeffs.
    """
    delta_coeffs = list()

    for k in range(begin, end):
        delta = 0
        for n in range(1, N + 1):
            if (t + n) < numframes:
                after = coeffs[t + n][k]
            else:
                after = 0

            if (t - n) >= 0:
                before = coeffs[t - n][k]
            else:
                before = 0

            delta = delta + n*(after - before)

        delta = delta / denom
        delta_coeffs.append(delta)

    return delta_coeffs

def mfcc_delta(mfccs, N = 2, double=True):
    """Calculates the Delta and Delta-Delta for a matrix of mfccs (frame x mfccs).

    @param mfccs: the original mfccs calculated by mfcc().
    @param N: complexity of delta (by default, 2).
    @param double: if True, calculates the Delta-Delta.
    """
    numcep = len(mfccs[0])
    denom = 2 * sum([n*n for n in range(1, N + 1)])
    numframes = len(mfccs)
    new_coeffs = list()

    for t in range(numframes):  #for each frame, take all 'numcep' coefficients
        delta_coeffs = delta(mfccs, t, numframes, N, denom, 0, numcep)
        new_coeffs.append(mfccs[t].tolist() + delta_coeffs)

    #TODO juntar os 2 for em um for duplo (no hurry)
    if double:
        for t in range(numframes):  #for each frame, take all 'numcep' coefficients
            delta_coeffs = delta(new_coeffs, t, numframes, N, denom, numcep, 2*numcep)
            new_coeffs[t] = new_coeffs[t] + delta_coeffs

    return np.array(new_coeffs)

# TEST
if __name__ == '__main__':
    import scipy.io.wavfile as wavf
    import matplotlib.pyplot as plt

    frame_len = 0.025*16000
    frame_step = 0.01*16000
    preemph_coeff = 0.95

    fbank = filterbanks()
    print('fbank', len(fbank), 'x', len(fbank[0]))
    print(fbank)
    plt.grid(True)
    for i in range(len(fbank)): #figure 1
        plt.plot(fbank[i], 'b')

    (samplerate, signal) = wavf.read("file.wav")
    signal_fb = filterbank_signal(signal)
    print('signal_fb', len(signal_fb))
    print('signal_fb[0] (features)', len(signal_fb[0]), 'x', len(signal_fb[0][0]))
    print(signal_fb[0])
    plt.figure()
    plt.grid(True)
    plt.plot(signal_fb[0]) #figure 2
    print('signal_fb[1] (energy)', len(signal_fb[1]))
    print(signal_fb[1])
    plt.figure()
    plt.grid(True)
    plt.plot(signal_fb[1]) #figure 3

    logsig = np.log(signal_fb[0])
    print('log signal_fb[0]', len(logsig), 'x', len(logsig[0]))
    print(logsig)
    plt.figure()
    plt.grid(True)
    plt.plot(logsig) #figure 4

    mfccs = mfcc(signal, samplerate, preemph=preemph_coeff)
    print('mfccs', len(mfccs), 'x', len(mfccs[0]))
    print(mfccs)
    plt.figure()
    plt.grid(True)
    plt.plot(mfccs) #figure 5
    plt.figure()
    plt.grid(True)
    for i in range(len(mfccs)): #figure 6
        plt.plot(mfccs[i])

    mfccs_delta = mfcc_delta(mfccs)
    print('mfccs_delta', len(mfccs_delta), 'x', len(mfccs_delta[0]))
    print(mfccs_delta)
    plt.figure()
    plt.grid(True)
    plt.plot(mfccs_delta) #figure 7
    plt.figure()
    plt.grid(True)
    for i in range(len(mfccs_delta)): #figure 8
        plt.plot(mfccs_delta[i])

    plt.show()