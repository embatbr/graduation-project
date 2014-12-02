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

def filterbanks(samplerate=16000, nfilt=26, nfft=512):
    """Computes a Mel-filterbank. The filters are stored in the rows, the columns
    correspond to fft bins. The filters are returned as an array of size
    (nfilt x (nfft/2 + 1))

    @param samplerate: the samplerate of the signal we are working with. Affects
    mel spacing.
    @param nfilt: the number of filters in the filterbank, default 26.
    @param nfft: the FFT size. Default is 512.

    @returns: A numpy array of size (nfilt x (floor(nfft/2) + 1)) containing filterbank.
    Each row holds 1 filter.
    """
    lowfreq_mel = 0
    highfreq_mel = hz2mel(samplerate / 2)
    melpoints = np.linspace(lowfreq_mel, highfreq_mel, nfilt + 2)
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

def filterbank_signal(signal, winlen, winstep, samplerate=16000, nfilt=26,
                      nfft=512, preemph=0.97):
    """Computes Mel-filterbank energy features from an audio signal.

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

    @returns: 2 values. The first is a numpy array of size (NUMFRAMES x nfilt)
    containing features. Each row holds 1 feature vector. The second is the energy
    in each frame (total energy, unwindowed)
    """
    signal = sigproc.preemphasis(signal, preemph)
    frames = sigproc.frame_signal(signal, winlen*samplerate, winstep*samplerate)
    pspec = sigproc.powspec(frames, nfft)

    signal_fb = filterbanks(samplerate, nfilt, nfft)
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

def mfcc(signal, winlen, winstep, samplerate=16000, numcep=13, nfilt=26, nfft=512,
         preemph=0.97, ceplifter=22, appendEnergy=True):
    """Computes MFCC features from an audio signal.

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

    @returns: A numpy array of size (numcep x NUMFRAMES) (transposed) containing
    features. Each row holds 1 feature (with NUMFRAMES "timestamps") and each
    column holds 1 vector (with numcep features). Ex:

    |f_1_1 f_1_2 ... f_1_T|
    |f_2_1 f_2_2 ... f_2_T|
    |...                  |
    |f_c_1 f_c_2 ... f_c_T|

    where 'c' is the number of features and 'T' the number of frames.
    """
    (feat, energy) = filterbank_signal(signal, winlen, winstep, samplerate, nfilt,
                                       nfft, preemph)
    feat = np.log(feat)
    feat = dct(feat, type=2, axis=1, norm='ortho')[ : , : numcep]
    feat = lifter(feat, ceplifter)
    if appendEnergy:
        # replace first cepstral coefficient with log of frame energy
        feat[ : , 0] = np.log(energy)

    return feat.transpose()

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
    new_coeffs[: numcep, :] = mfccs[:,:]

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
               nfft=512, preemph=0.97, ceplifter=22, appendEnergy=True, N = 2,
               num_deltas=2):
    """Computes MFCC features from an audio signal and calculates it's deltas.

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
    @param mfccs: the original mfccs calculated by mfcc().
    @param N: complexity of delta (by default, 2).
    @param double: if True, calculates the Delta-Delta.

    @returns: A numpy array of size (num_deltas*numcep x NUMFRAMES) (transposed)
    containing features (the MFCCs and it's deltas).
    """
    mfccs = mfcc(signal, winlen, winstep, samplerate, numcep, nfilt, nfft, preemph,
                 ceplifter, appendEnergy)
    return add_delta(mfccs, N, num_deltas)


# TEST
if __name__ == '__main__':
    import scipy.io.wavfile as wavf
    import matplotlib.pyplot as plt

    winlen = 0.02
    winstep = 0.01
    preemph = 0.97
    num_deltas = 2

    fbank = filterbanks()
    print('fbank', len(fbank), 'x', len(fbank[0]))
    print(fbank)
    fig = plt.figure()
    plt.grid(True)
    for i in range(len(fbank)): #figure 1
        plt.plot(fbank[i], 'b')
    fig.suptitle('filterbanks')
    plt.xlabel('filter size')

    (samplerate, signal) = wavf.read('../bases/mit/corpuses/enroll_2/f08/phrase54_16k.wav')
    signal_fb = filterbank_signal(signal, winlen, winstep, samplerate, preemph=preemph)
    print('signal_fb', len(signal_fb))
    print('signal_fb[0] (features)', len(signal_fb[0]), 'x', len(signal_fb[0][0]))
    print(signal_fb[0])
    recovered = np.array(list())
    for sigfb in signal_fb[0]:
        recovered = np.concatenate((recovered, sigfb))
    fig = plt.figure()
    plt.grid(True)
    plt.plot(recovered) #figure 2
    fig.suptitle('signal filterbanked (features)')
    plt.xlabel('frequency (Hz)')
    print('signal_fb[1] (energy)', len(signal_fb[1]))
    print(signal_fb[1])
    fig = plt.figure()
    plt.grid(True)
    plt.plot(signal_fb[1]) #figure 3
    fig.suptitle('signal filterbanked (energy)')

    logsig = np.log(signal_fb[0])
    print('log signal_fb[0]', len(logsig), 'x', len(logsig[0]))
    print(logsig)
    recovered = np.array(list())
    for lsig in logsig:
        recovered = np.concatenate((recovered, lsig))
    fig = plt.figure()
    plt.grid(True)
    plt.plot(recovered) #figure 4
    fig.suptitle('log signal filterbanked (features)')
    plt.xlabel('frequency (Hz)')

    mfccs = mfcc(signal, winlen, winstep, samplerate, preemph=preemph)
    print('mfccs', len(mfccs), 'x', len(mfccs[0]))
    print(mfccs)
    recovered = np.array(list())
    for mfcc in mfccs:
        recovered = np.concatenate((recovered, mfcc))
    fig = plt.figure()
    plt.grid(True)
    plt.plot(recovered) #figure 5
    fig.suptitle('mfccs')
    plt.xlabel('time (samples)')

    mfccs_deltas = mfcc_delta(signal, winlen, winstep, samplerate, preemph=preemph,
                              num_deltas=num_deltas)
    print('mfccs_deltas', len(mfccs_deltas), 'x', len(mfccs_deltas[0]))
    print(mfccs_deltas)
    recovered = np.array(list())
    for mfcc_d in mfccs_deltas:
        recovered = np.concatenate((recovered, mfcc_d))
    fig = plt.figure()
    plt.grid(True)
    plt.plot(recovered) #figure 6
    fig.suptitle('mfccs + deltas until %dª order' % num_deltas)
    plt.xlabel('time (samples)')

    plt.show()