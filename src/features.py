"""This module was made "from scratch" using the codes provided by James Lyons
at the url "github.com/jameslyons/python_speech_features". It includes routines
to calculate the MFCCs extraction.

Most part of the code is similar to the "inspiration". What I did was read his
code and copy what I understood. The idea is to do the same he did, using his
code as a guide.
"""


import numpy as np
from scipy.fftpack import dct
import math

from useful import CORPORA_DIR
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

def filterbanks(samplerate=16000, nfilt=26, NFFT=512):
    """Computes a Mel-filterbank. The filters are stored in the rows, the columns
    correspond to fft bins. The filters are returned as an array of size
    (nfilt x (NFFT/2 + 1))

    @param samplerate: the samplerate of the signal we are working with. Affects
    mel spacing.
    @param nfilt: the number of filters in the filterbank, default 26.
    @param NFFT: the FFT size. Default is 512.

    @returns: A numpy array of size (nfilt x (floor(NFFT/2) + 1)) containing filterbank.
    Each row holds 1 filter.
    """
    lowfreq_mel = 0
    highfreq_mel = hz2mel(samplerate / 2)
    melpoints = np.linspace(lowfreq_mel, highfreq_mel, nfilt + 2)
    hzpoints = mel2hz(melpoints)
    bin = np.floor((NFFT + 1) * hzpoints / samplerate)

    fbank = np.zeros([nfilt, NFFT/2 + 1])
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

def filterbank_signal(signal, winlen, winstep, samplerate=16000, nfilt=26,
                      NFFT=512, preemph=0.97):
    """Computes Mel-filterbank energy features from an audio signal.

    @param signal: the audio signal from which to compute features. Should be an
    N*1 array
    @param winlen: the length of the analysis window in seconds. Default is 0.02s
    (20 milliseconds)
    @param winstep: the step between seccessive windows in seconds. Default is
    0.01s (10 milliseconds)
    @param samplerate: the samplerate of the signal we are working with.
    @param nfilt: the number of filters in the filterbank, default 26.
    @param NFFT: the FFT size. Default is 512.
    @param lowfreq: lowest band edge of mel filters. In Hz, default is 0.
    @param highfreq: highest band edge of mel filters. In Hz, default is samplerate/2
    @param preemph: apply preemphasis filter with preemph as coefficient. 0 is no
    filter. Default is 0.97.

    @returns: 2 values. The first is a numpy array of size (NUMFRAMES x nfilt)
    containing features. Each row holds 1 feature vector. The second is the energy
    in each frame (total energy, unwindowed)
    """
    signal = sigproc.preemphasis(signal, preemph)
    frames = sigproc.frame_signal(signal, winlen*samplerate, winstep*samplerate,
                                  winfunc=lambda x:np.hamming(x))
    pspec = sigproc.powspec(frames, NFFT)

    signal_fb = filterbanks(samplerate, nfilt, NFFT)
    feat = np.dot(pspec, signal_fb.T)    # computes the filterbank energies
    energy = np.sum(pspec, 1)            # this stores the total energy in each frame

    return (feat, energy)

def lifter(cepstra, L=22):
    """Apply a cepstral lifter to the matrix of cepstra. This has the effect of
    increasing the magnitude of the high frequency DCT coeffs.

    @param cepstra: the matrix of mel-cepstra, with size numframes*numcep.
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
        # values of L <= 0, do nothing
        return cepstra

def mfcc(signal, winlen, winstep, samplerate=16000, numcep=13, nfilt=26, NFFT=512,
         preemph=0.97, ceplifter=22, appendEnergy=True, normalize=False):
    """Computes MFCC features from an audio signal.

    @param signal: the audio signal from which to compute features. Should be an
    N*1 array
    @param winlen: the length of the analysis window in seconds. Default is
    0.02s (20 milliseconds)
    @param winstep: the step between successive windows in seconds. Default is
    0.01s (10 milliseconds)
    @param samplerate: the samplerate of the signal we are working with.
    @param numcep: the number of cepstrum to return, default 13
    @param nfilt: the number of filters in the filterbank, default 26.
    @param NFFT: the FFT size. Default is 512.
    @param preemph: apply preemphasis filter with preemph as coefficient. 0 is
    no filter. Default is 0.97.
    @param ceplifter: apply a lifter to final cepstral coefficients. 0 is no
    lifter. Default is 22.
    @param appendEnergy: if this is true, the zeroth cepstral coefficient is
    replaced with the log of the total frame energy.

    @returns: A numpy array of size (numcep x NUMFRAMES) (transposed) containing
    features. Each row holds 1 feature (with NUMFRAMES "timestamps") and each
    column holds 1 vector (with numcep features). Ex:

    |f_1_1 f_1_2 ... f_1_T|
    |f_2_1 f_2_2 ... f_2_T|
    |...                  |
    |f_c_1 f_c_2 ... f_c_T|

    where 'c' is the number of features and 'T' the number of frames.
    """
    (feats, energy) = filterbank_signal(signal, winlen, winstep, samplerate, nfilt,
                                       NFFT, preemph)
    feats = np.log(feats)
    feats = dct(feats, type=2, axis=1, norm='ortho')[ : , : numcep]
    feats = lifter(feats, ceplifter)
    if appendEnergy:
        # replace first cepstral coefficient with log of frame energy
        feats[ : , 0] = np.log(energy)

    feats = feats.transpose()

    #Cepstral Mean Normalization
    if normalize:
        feats = np.array([feat - np.mean(feat) for feat in feats])

    return feats

#TODO efetuar Cepstral Mean Subtraction (CMS) antes de calcular os deltas

def add_delta(mfccs, N = 2, num_deltas=2):
    """Calculates the deltas for a matrix of mfccs (frame x mfccs).

    @param mfccs: the original mfccs calculated by mfcc().
    @param N: complexity of delta (by default, 2).
    @param double: if True, calculates the Delta-Delta.

    @returns: a new array containing the original MFCCs and the deltas calculated.
    """
    numcep = len(mfccs[:, 0])
    denom = 2 * sum([n*n for n in range(1, N + 1)])
    numframes = len(mfccs[0])
    new_coeffs = np.zeros((numcep*(1 + num_deltas), numframes))
    new_coeffs[: numcep, :] = mfccs[:,:]    #copy MFCCs array

    for order in range(num_deltas):
        for k in range(order*numcep, (order + 1)*numcep): #index of coeff(0 to 12, 13 to 25 and 26 to 38)
            coeff = new_coeffs[k]
            for t in range(numframes):
                delta = 0
                for n in range(1, N + 1):
                    if (t + n) < numframes:
                        after = coeff[t + n]
                    else:
                        after = 0

                    if (t - n) >= 0:
                        before = coeff[t - n]
                    else:
                        before = 0

                    delta = delta + n*(after - before)

                new_coeffs[k + numcep][t] = delta / denom

    return new_coeffs

def mfcc_delta(signal, winlen, winstep, samplerate=16000, numcep=13, nfilt=26,
               NFFT=512, preemph=0.97, ceplifter=22, appendEnergy=True, N = 2,
               num_deltas=2):
    """Computes MFCC features from an audio signal and calculates it's deltas.

    @param signal: the audio signal from which to compute features. Should be an
    N*1 array
    @param winlen: the length of the analysis window in seconds. Default is
    0.02s (20 milliseconds)
    @param winstep: the step between successive windows in seconds. Default is
    0.01s (10 milliseconds)
    @param samplerate: the samplerate of the signal we are working with.
    @param numcep: the number of cepstrum to return, default 13.
    @param nfilt: the number of filters in the filterbank, default 26.
    @param NFFT: the FFT size. Default is 512.
    @param preemph: apply preemphasis filter with preemph as coefficient. 0 is
    no filter. Default is 0.97.
    @param ceplifter: apply a lifter to final cepstral coefficients. 0 is no
    lifter. Default is 22.
    @param appendEnergy: if this is true, the zeroth cepstral coefficient is
    replaced with the log of the total frame energy.
    @param N: complexity of delta (by default, 2).
    @param num_deltas: number of delta calculations.

    @returns: A numpy array of size (num_deltas*numcep x NUMFRAMES) (transposed)
    containing features (the MFCCs and it's deltas).
    """
    mfccs = mfcc(signal, winlen, winstep, samplerate, numcep, nfilt, NFFT, preemph,
                 ceplifter, appendEnergy)
    if num_deltas < 1:
        return mfccs
    return add_delta(mfccs, N, num_deltas)


# TEST
if __name__ == '__main__':
    import scipy.io.wavfile as wavf
    import matplotlib.pyplot as plt

    (samplerate, signal) = wavf.read('%smit/enroll_2/f08/phrase54_16k.wav' %
                                     CORPORA_DIR)

    winlen = 0.02
    winstep = 0.01
    preemph = 0.97
    num_deltas = 2
    nfilt = 26
    NFFT = 512
    freq = np.linspace(0, samplerate/2, math.floor(NFFT/2 + 1))

    fbank = filterbanks(samplerate, nfilt, NFFT)
    print('fbank', len(fbank), 'x', len(fbank[0]))
    print(fbank)
    fig = plt.figure()
    plt.grid(True)
    #figure 1
    for i in range(len(fbank)):
        plt.plot(np.array(list(range(1, math.floor(NFFT/2 + 1) + 1))), fbank[i])
    fig.suptitle('%d filters' % nfilt)
    plt.xlabel('from 0 to %d (filter length)' % math.floor(NFFT/2 + 1))

    signal_fb = filterbank_signal(signal, winlen, winstep, samplerate, nfilt,
                                  NFFT, preemph)

    print('signal_fb', len(signal_fb))
    print('signal_fb[0] (features)', len(signal_fb[0]), 'x', len(signal_fb[0][0]))
    print(signal_fb[0])
    fig = plt.figure()
    plt.grid(True)
    #figure 2
    for sigfb in signal_fb[0]:
        plt.plot(np.array(list(range(1, nfilt + 1))), sigfb)
    fig.suptitle('signal filterbanked (%d features)' % len(signal_fb[0]))
    plt.xlabel('filter')
    plt.ylabel('filter value')

    print('signal_fb[1] (energy)', len(signal_fb[1]))
    print(signal_fb[1])
    fig = plt.figure()
    plt.grid(True)
    #figure 3
    plt.plot(np.array(list(range(1, len(signal_fb[1]) + 1))), signal_fb[1])
    fig.suptitle('signal filterbanked (energy)')
    plt.xlabel('frame')
    plt.ylabel('energy')

    logsig = np.log(signal_fb[0])
    print('log signal_fb[0]', len(logsig), 'x', len(logsig[0]))
    print(logsig)
    fig = plt.figure()
    plt.grid(True)
    for lsig in logsig: #figure 4
        plt.plot(np.array(list(range(1, len(lsig) + 1))), lsig)
    fig.suptitle('log signal filterbanked (%d features)' % len(logsig))
    plt.xlabel('filter')
    plt.ylabel('filter value')

    mfccs = mfcc(signal, winlen, winstep, samplerate, nfilt=nfilt, NFFT=NFFT,
                 preemph=preemph)
    print('mfccs', len(mfccs), 'x', len(mfccs[0]))
    print(mfccs)
    fig = plt.figure()
    plt.grid(True)
    for melfeat in mfccs: #figure 5
        plt.plot(np.array(list(range(1, len(melfeat) + 1))), melfeat)
    fig.suptitle('%d mfccs' % len(mfccs))
    plt.xlabel('frame')
    plt.ylabel('feature value')

    mfccs_deltas = mfcc_delta(signal, winlen, winstep, samplerate, nfilt=nfilt,
                              NFFT=NFFT, preemph=preemph, num_deltas=num_deltas)
    print('mfccs_deltas', len(mfccs_deltas), 'x', len(mfccs_deltas[0]))
    print(mfccs_deltas)
    fig = plt.figure()
    plt.grid(True)
    for melfeat_deltas in mfccs_deltas: #figure 6
        plt.plot(np.array(list(range(1, len(melfeat_deltas) + 1))), melfeat_deltas)
    fig.suptitle('%d mfccs + %d deltas' % (len(mfccs), len(mfccs)*num_deltas))
    plt.xlabel('frame')
    plt.ylabel('feature value')

    plt.show()