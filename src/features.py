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

def filterbank(samplerate=16000, nfilt=26, NFFT=512):
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
    highfreq_mel = hz2mel(samplerate/2)
    melpoints = np.linspace(lowfreq_mel, highfreq_mel, nfilt + 2)
    hzpoints = mel2hz(melpoints)
    bin = np.floor((NFFT + 1) * hzpoints / samplerate)  #'bin', from FFT bin = 'caixa' de FFT

    fbank = np.zeros((nfilt, math.floor(NFFT/2 + 1)))
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

def filtersignal(signal, winlen=0.02, winstep=0.01, samplerate=16000, nfilt=26,
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
    in each frame (total energy, unwindowed), so, a numpy array of energies with
    size NUMFRAMES.
    """
    presignal = sigproc.preemphasis(signal, preemph)
    framedsignal = sigproc.framesignal(presignal, winlen*samplerate, winstep*samplerate)
    powframedsignal = sigproc.powspec(framedsignal, NFFT)

    fbank = filterbank(samplerate, nfilt, NFFT)
    feats = np.dot(powframedsignal, fbank.T)       # feats[n] = np.dot(powframedsignal, fbank[n])
    energy = np.sum(powframedsignal, 1)            # this stores the total energy in each frame

    return (feats, energy)

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
         preemph=0.97, ceplifter=22, appendEnergy=True):
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
    (feats, energy) = filtersignal(signal, winlen, winstep, samplerate, nfilt,
                                   NFFT, preemph)
    logfeats = np.log10(feats)
    logfeats = dct(logfeats, type=2, axis=1, norm='ortho')[ : , : numcep]
    liflogfeats = lifter(logfeats, ceplifter)
    if appendEnergy:
        # replace first cepstral coefficient with log of frame energy
        liflogfeats[ : , 0] = np.log(energy)

    liflogfeats = liflogfeats.transpose()
    #CMS = Cepstral Mean Subtraction
    #liflogfeats = np.array([feat - np.mean(feat) for feat in liflogfeats])
    return liflogfeats

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


#TESTS
if __name__ == '__main__':
    import scipy.io.wavfile as wavf
    import os, os.path, shutil
    from useful import CORPORA_DIR, IMAGES_DIR, plotfile, multiplotfile


    IMAGES_FEATURES_DIR = '%sfeatures/' % IMAGES_DIR

    if os.path.exists(IMAGES_FEATURES_DIR):
            shutil.rmtree(IMAGES_FEATURES_DIR)
    os.mkdir(IMAGES_FEATURES_DIR)

    filecounter = 0
    filename = '%sfigure' % IMAGES_FEATURES_DIR

    #Reading signal from base and plotting (signal and preemphasized signal)
    voice = ('enroll_2', 'f08', 54)
    (enroll, speaker, speech) = voice
    speechname = '%smit/%s/%s/phrase%02d_16k.wav' % (CORPORA_DIR, enroll, speaker, speech)
    (samplerate, signal) = wavf.read(speechname)
    presignal = sigproc.preemphasis(signal)     #coeff = 0.97

    NFFT = 512
    numfftbins = math.floor(NFFT/2 + 1)    #fft bins == 'caixas' de FFT
    freq = np.linspace(0, samplerate/2, numfftbins)
    nfilt = 26

    #Mel frequence plotting
    melfreq = hz2mel(freq)
    ###figure000
    filecounter = plotfile(freq, melfreq, 'Mel scale', 'f (Hz)', 'mel[f]',
                           filename, filecounter, 'red')
    ###figure001
    filecounter = plotfile(melfreq, np.log10(melfreq), 'Log-mel scale', 'm (Mel)',
                           'log10[m]', filename, filecounter, 'red')

    #Filterbank
    fbank = filterbank(samplerate, nfilt, NFFT)
    numfilters = len(fbank)
    print('#filters = %d' % numfilters)
    ###figure002
    filecounter = multiplotfile(freq, fbank, '%d-filterbank, NFFT = %d' % (nfilt, NFFT),
                                'f (Hz)', 'filter[n][f]', filename, filecounter,
                                'green')

    #Pre emphasized signal's squared magnitude of spectrum after 21st filter (index 20)
    powpresig = sigproc.powspec(presignal, NFFT)
    filter_index = 20
    framedpowpresig = np.multiply(powpresig, fbank[filter_index])
    ###figure003
    filecounter = plotfile(freq, fbank[filter_index], 'Filter[%d]' % filter_index,
                           'f (Hz)', 'filter[%d]' % filter_index, filename,
                           filecounter, 'green')
    ###figure004
    filecounter = plotfile(freq, framedpowpresig, '|FFT * filter[%d]|²' % filter_index,
                           'f (Hz)', '|FFT * filter[%d]|²' % filter_index, filename,
                           filecounter, 'red', True)

    #|FFT|² of pre emphasized signal after filterbank
    powpresig = sigproc.powspec(presignal, NFFT)
    powpresigfull = np.zeros(len(powpresig))
    for f in fbank:
        fspec = np.multiply(powpresig, f)
        powpresigfull = np.maximum(powpresigfull, fspec)
    ###figure005
    filecounter = plotfile(freq, powpresigfull, '|FFT|² * %d-filterbank' % nfilt,
                           'f (Hz)', 'powspec[f]', filename, filecounter, 'red',
                           True)

    winlen = 0.02
    winstep = 0.01

    #Filterbanked signal
    fbankedsignal = filtersignal(signal, winlen, winstep, samplerate, nfilt, NFFT)
    (feats, energy) = fbankedsignal
    numframes = len(energy)
    frameindices = np.linspace(0, numframes, numframes, False)

    print(feats.shape, energy.shape)
    for (feat, n) in zip(feats.T, range(numframes)):
        filecounter = plotfile(frameindices, feat, 'Feature %d' % n, 'frame',
                               'feature[frame]', filename, filecounter, 'magenta')

#    featsfull = np.zeros(numframes)
#    for (feat, n) in zip(feats.T, range(numframes)):
#        featsfull = np.maximum(featsfull, feat)
#        logfeat = np.log10(feat)
#        plotfile(frameindices, feat, 'Feature %d' % n, 'frame', 'feature[frame]')
#            plotfile(frameindices, logfeat, xlabel='frames', ylabel='log(feature[frame])',
#                     suptitle='Log-feature %d' % n)

    #    if args == []:
    #        plotfile(frameindices, featsfull, xlabel='frames', ylabel='max(feature[frame])',
    #                 suptitle='Maximum feature value per frame')
    #
    #    #Energy per frame
    #    plotfile(frameindices, energy, xlabel='frames', ylabel='energy[frame]',
    #             suptitle='Energy per frame')
    #    plotfile(frameindices, np.log10(energy), xlabel='frames', ylabel='log(energy[frame])',
    #             suptitle='Log-energy per frame')
    #
    #elif option == 'mfcc':
    #    pass
