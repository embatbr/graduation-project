#!/usr/bin/python3.4

"""Module with code to exhibit data on the screen.
"""


import numpy as np
import pylab as pl
import scipy.io.wavfile as wavf
from scipy.fftpack import dct
import pickle
import time
from matplotlib.patches import Ellipse
from common import UBMS_DIR, GMMS_DIR, frange
import features


FONTSIZE = 8


def plot_gmm(gmm, featsvec, x_axis=0, y_axis=1, param_feats='b.', param_mix='r.'):
    """Plots a GMM and the vector of features used to train it. The plotting is
    in a 2D space.

    @param gmm: the GMM.
    @param featsvec: the vector of features.
    @param x_axis: the dimension plotted in the x axis.
    @param y_axis: the dimension plotted in the y axis.
    """
    # features
    if type(featsvec) is list:
        for (feats, param) in zip(featsvec, param_feats):
            pl.plot(feats[:, x_axis], feats[:, y_axis], param)
    else:
        pl.plot(featsvec[:, x_axis], featsvec[:, y_axis], param_feats)

    # mixture of gaussians
    M = len(gmm.meansvec)
    if type(param_mix) is list:
        pl.plot(gmm.meansvec[: M//2, x_axis], gmm.meansvec[: M//2, y_axis], param_mix[0])
        pl.plot(gmm.meansvec[M//2 :, x_axis], gmm.meansvec[M//2 :, y_axis], param_mix[1])
    else:
        pl.plot(gmm.meansvec[:, x_axis], gmm.meansvec[:, y_axis], param_mix)
    ax = pl.gca()

    m = 0
    for (means, variances) in zip(gmm.meansvec, gmm.variancesvec):
        if not (type(param_mix) is list):
            color = param_mix[0]
        elif m < M//2:
            color = param_mix[0][0]
        else:
            color = param_mix[1][0]
        m = m + 1

        ellipse = Ellipse(xy=(means[x_axis], means[y_axis]), width=variances[x_axis]**0.5,
                          height=variances[y_axis]**0.5, edgecolor=color,
                          linewidth=1.5, fill=False, zorder=2)
        ax.add_artist(ellipse)


def set_plot_params(ax, fontsize=FONTSIZE, grid=False, xticks=None, yticks=None):
    """Changes the size of ticks on axes x and y.

    @param ax: the object containing the axes x and y.
    @param fontsize: the size of the font. Default, 10.
    @param grid: determines if must show a grid. Default, True.
    @param xticks: the x ticks. Default, None.
    @param yticks: the y ticks. Default, None.
    """
    pl.grid(grid)
    if not (xticks is None):
        pl.xticks(xticks)
    if not (yticks is None):
        pl.yticks(yticks)
    [tick.label.set_fontsize(fontsize) for tick in ax.xaxis.get_major_ticks()]
    [tick.label.set_fontsize(fontsize) for tick in ax.yaxis.get_major_ticks()]


if __name__ == '__main__':
    import sys
    import bases
    import mixtures

    command = sys.argv[1]
    args = sys.argv[2:] if len(sys.argv) > 2 else list()

    numceps = 19
    t = time.time()

    if command == 'utterance':
        SIGNAL_PATH = '../bases/corpora/mit/enroll_1/f00/phrase02_16k.wav'
        (samplerate, signal) = wavf.read(SIGNAL_PATH)

        # plotting utterance "karen livescu"
        duration = np.linspace(0, len(signal) / samplerate, len(signal))
        ax = pl.subplot(3, 1, 1)
        set_plot_params(ax, grid=True)
        pl.plot(duration, signal, 'b')
        ax.set_xlim([duration[0], duration[-1]])

        FILE_PATH = '../docs/report/images/chapters/speaker-recognition-systems/speech_signal.png'
        pl.savefig(FILE_PATH, bbox_inches='tight')

    elif command == 'mfcc-images':
        mask = '1' * 7
        if len(args) > 0:
            mask = args[0]

        nfilt = 26
        preemph = 0.97
        NFFT = 512
        winlen = 0.02
        winstep = 0.01
        numceps = 6
        ceplifter = 22

        SIGNAL_PATH = '../bases/corpora/mit/enroll_1/f00/phrase02_16k.wav'
        (samplerate, signal) = wavf.read(SIGNAL_PATH)

        if mask[0] == '1':
            print('plotting hertz-mel relation')
            lowfreq_mel = 0
            highfreq_mel = features.hz2mel(samplerate/2)
            melpoints = np.linspace(lowfreq_mel, highfreq_mel, nfilt + 2) # equally spaced in mel scale
            hzpoints = features.mel2hz(melpoints)

            ax = pl.subplot(2, 1, 1)
            xticks = np.arange(0, samplerate/2 + 1, 1000)
            set_plot_params(ax, grid=True, xticks=xticks)
            pl.plot(hzpoints, melpoints, 'r')

            FILE_PATH = '../docs/report/images/chapters/feature-extraction/mel_scale.png'
            pl.savefig(FILE_PATH, bbox_inches='tight')
            pl.clf()

        if mask[1] == '1':
            print('plotting signal and preemphasized signal')

            # plotting signals
            duration = np.linspace(0, len(signal) / samplerate, len(signal))
            ax = pl.subplot(3, 2, 1)
            ax.set_title('signal', size=10)
            set_plot_params(ax, grid=True)
            pl.plot(duration, signal, 'b')
            ax = pl.subplot(3, 2, 2)
            ax.set_title('pre-emphasized signal', size=10)
            set_plot_params(ax, grid=True)
            emph_signal = features.preemphasis(signal, preemph)
            pl.plot(duration, emph_signal, 'b')

            # plotting spectra
            frequencies = np.linspace(0, samplerate//2, NFFT//2 + 1)
            xticks = np.arange(0, samplerate//2 + 1, 2000)
            ax = pl.subplot(3, 2, 3)
            pl.subplots_adjust(hspace=0.4)
            ax.set_title('signal\'s spectrum', size=10)
            set_plot_params(ax, grid=True, xticks=xticks)
            signal_magspec = features.magspec(signal)
            pl.fill_between(frequencies, signal_magspec, edgecolor='red', facecolor='red')
            ax = pl.subplot(3, 2, 4)
            pl.subplots_adjust(hspace=0.4)
            ax.set_title('pre-emphasized signal\'s spectrum', size=10)
            set_plot_params(ax, grid=True, xticks=xticks)
            emph_magspec = features.magspec(emph_signal)
            pl.fill_between(frequencies, emph_magspec, edgecolor='red', facecolor='red')

            FILE_PATH = '../docs/report/images/chapters/feature-extraction/preemphasis.png'
            pl.savefig(FILE_PATH, bbox_inches='tight')
            pl.clf()

        if mask[2] == '1':
            print('plotting framing')

            frames = features.framesignal(emph_signal, winlen*samplerate, winstep*samplerate)
            frame = frames[50]
            begin = int(50*winstep*samplerate)
            end = int((50*winstep + winlen)*samplerate)
            length = int(winlen*samplerate)
            duration = np.linspace(begin, end, length)
            xticks = np.arange(begin, end, 50)
            ax = pl.subplot(3, 1, 1)
            set_plot_params(ax, grid=True, xticks=xticks)
            pl.plot(duration, frame, 'g')

            FILE_PATH = '../docs/report/images/chapters/feature-extraction/framing.png'
            pl.savefig(FILE_PATH, bbox_inches='tight')
            pl.clf()

        if mask[3] == '1':
            print('plotting FFT')

            frequencies = np.linspace(0, samplerate//2, NFFT//2 + 1)
            xticks = np.arange(0, samplerate//2 + 1, 1000)
            position = 1
            for func in [features.magspec, features.powspec]:
                func_frames = func(frames)
                ax = pl.subplot(2, 1, position)
                if position == 2:
                    pl.subplots_adjust(hspace=0.2)
                position = position + 1
                set_plot_params(ax, grid=True, xticks=xticks)
                for magframe in func_frames:
                    pl.plot(frequencies, magframe)

            FILE_PATH = '../docs/report/images/chapters/feature-extraction/fft.png'
            pl.savefig(FILE_PATH, bbox_inches='tight')
            pl.clf()

        if mask[4] == '1':
            print('plotting filterbank')

            frequencies = np.linspace(0, samplerate//2, NFFT//2 + 1)
            fbank = features.filterbank(samplerate, nfilt, NFFT)
            xticks = np.arange(0, samplerate//2 + 1, 1000)
            ax = pl.subplot(3, 1, 1)
            set_plot_params(ax, xticks=xticks)
            for f in fbank:
                pl.plot(frequencies, f, 'y')

            FILE_PATH = '../docs/report/images/chapters/feature-extraction/filterbank.png'
            pl.savefig(FILE_PATH, bbox_inches='tight')
            pl.clf()

        if mask[5] == '1':
            print('plotting features_and_featuresdB')

            emph_signal = features.preemphasis(signal, preemph)
            frames = features.framesignal(emph_signal, winlen*samplerate, winstep*samplerate)
            powframes = features.powspec(frames, NFFT)
            fbank = features.filterbank(samplerate, nfilt, NFFT)
            featsvec = np.dot(powframes, fbank.T)
            xticks = np.arange(0, len(featsvec) + 1, 20)
            for position in [1, 2]:
                ax = pl.subplot(2, 2, position)
                set_plot_params(ax, grid=True, xticks=xticks)
                if position == 2:
                    featsvec = 20*np.log10(featsvec) #dB
                for feats in featsvec.T:
                    pl.plot(feats)

            FILE_PATH = '../docs/report/images/chapters/feature-extraction/features_and_featuresdB.png'
            pl.savefig(FILE_PATH, bbox_inches='tight')
            pl.clf()

        if mask[6] == '1':
            print('plotting mfcc')

            emph_signal = features.preemphasis(signal, preemph)
            frames = features.framesignal(emph_signal, winlen*samplerate, winstep*samplerate)
            powframes = features.powspec(frames, NFFT)
            fbank = features.filterbank(samplerate, nfilt, NFFT)
            featsvec = np.dot(powframes, fbank.T)
            featsvec = 20*np.log10(featsvec) #dB
            featsvec = dct(featsvec, type=2, axis=1, norm='ortho')[ : , : numceps] # TODO colocar n=26?
            featsvec = features.lifter(featsvec, ceplifter)

            xticks = np.arange(0, len(featsvec) + 1, 20)
            configs = [(False, False, 0, False, 'mfcc'),
                       (True, False, 0, False, 'mfcc_energy_appended'),
                       (True, True, 0, False, 'mfcc_energy_appended_cms'),
                       (True, True, 2, False, 'mfcc_energy_appended_cms_delta_order_2'),
                       (True, True, 2, True, 'mfcc_energy_appended_cms_delta_order_2_shifted')]
            for (append_energy, applyCMS, delta_order, frac, filename) in configs:
                ax = pl.subplot(3, 1, 1)
                set_plot_params(ax, grid=True, xticks=xticks)

                if append_energy:
                    energy = np.sum(powframes, axis=1) # stores the total energy of each frame
                    energy = 20*np.log10(energy) #dB
                    featsvec[ : , 0] = energy

                if applyCMS:
                    featsvec = featsvec - np.mean(featsvec, axis=0) # CMS reduces the effect of noise

                while delta_order > 0:
                    N = 2
                    featsvec = features.append_deltas(featsvec, numceps, N)
                    delta_order = delta_order - 1

                if frac:
                    min_featsvec = np.amin(featsvec, axis=0)
                    featsvec = featsvec + (1 - min_featsvec)

                pl.plot(featsvec)

                directory = '/chapters/%s' % ('gmm' if frac else 'feature-extraction')
                FILE_PATH = '../docs/report/images%s/%s.png' % (directory, filename)
                pl.savefig(FILE_PATH, bbox_inches='tight')
                pl.clf()

    elif command == 'em':
        speaker = args[0]
        M = int(args[1])
        delta_order = int(args[2])
        x_axis = int(args[3])
        y_axis = int(args[4])

        featsvec = bases.read_speaker(numceps, delta_order, 'enroll_1', speaker,
                                      downlim='01', uplim='59')

        gmm = mixtures.GMM(speaker, M, numceps, featsvec)
        ax = pl.subplot(2, 2, 1)
        set_plot_params(ax)
        plot_gmm(gmm, featsvec, x_axis, y_axis)
        gmm.train(featsvec)
        ax = pl.subplot(2, 2, 2)
        set_plot_params(ax)
        plot_gmm(gmm, featsvec, x_axis, y_axis)

        FILE_PATH = '../docs/report/images/chapters/gmm/em_algorithm.png'
        pl.savefig(FILE_PATH, bbox_inches='tight')

    elif command == 'frac-em':
        speaker = args[0]
        M = int(args[1])
        delta_order = int(args[2])
        x_axis = int(args[3])
        y_axis = int(args[4])

        rs = [0.95, 0.99, 1, 1.01, 1.05]
        featsvec = bases.read_speaker(numceps, delta_order, 'enroll_1', speaker,
                                      downlim='01', uplim='19')
        min_featsvec = np.amin(featsvec, axis=0)
        featsvec_shifted = featsvec + (1 - min_featsvec)

        for r in rs:
            print('r = %.02f' % r)
            frac_gmm = mixtures.GMM(speaker, M, numceps, featsvec, r=r)
            ax = pl.subplot(2, 2, 1)
            set_plot_params(ax)
            plot_gmm(frac_gmm, [featsvec, featsvec_shifted], x_axis, y_axis, ['b.', 'g.'])
            frac_gmm.train(featsvec)
            ax = pl.subplot(2, 2, 2)
            set_plot_params(ax)
            plot_gmm(frac_gmm, featsvec, x_axis, y_axis)

            print('Testing fractional likelihoods')
            featslist = bases.read_features_list(numceps, delta_order, 'enroll_2',
                                                 speaker, downlim='01', uplim='19')
            log_likes = list()
            for feats in featslist:
                log_likes.append(frac_gmm.log_likelihood(feats))
            print('max = %f, min = %f' % (max(log_likes), min(log_likes)))

            r_apx = str(r)[ : 4].replace('.', '')
            FILE_PATH = '../docs/report/images/chapters/gmm/em_algorithm_r%s.png' % r_apx
            pl.savefig(FILE_PATH, bbox_inches='tight')
            pl.clf()

    elif command == 'frac-em-extremes':
        speaker = args[0]
        M = int(args[1])
        delta_order = int(args[2])
        x_axis = int(args[3])
        y_axis = int(args[4])

        featsvec = bases.read_speaker(numceps, delta_order, 'enroll_1', speaker,
                                      downlim='01', uplim='19')

        (r_left, r_right) = (0.8, 1.2)

        frac_gmm = mixtures.GMM(speaker, M, numceps, featsvec, r=r_left)
        frac_gmm.train(featsvec)
        ax = pl.subplot(2, 2, 1)
        ax.set_title('r = %.1f' % r_left, fontsize=FONTSIZE)
        set_plot_params(ax)
        plot_gmm(frac_gmm, featsvec, x_axis, y_axis)

        frac_gmm = mixtures.GMM(speaker, M, numceps, featsvec, r=r_right)
        frac_gmm.train(featsvec)
        ax = pl.subplot(2, 2, 2)
        ax.set_title('r = %.1f' % r_right, fontsize=FONTSIZE)
        set_plot_params(ax)
        plot_gmm(frac_gmm, featsvec, x_axis, y_axis)

        FILE_PATH = '../docs/report/images/chapters/experiments/frac-em-extremes.png'
        pl.savefig(FILE_PATH, bbox_inches='tight')
        pl.clf()

    elif command == 'frac-em-r-down':
        speaker = args[0]
        M = int(args[1])
        delta_order = int(args[2])
        x_axis = int(args[3])
        y_axis = int(args[4])

        featsvec = bases.read_speaker(numceps, delta_order, 'enroll_1', speaker,
                                      downlim='01', uplim='19')

        min_featsvec = np.amin(featsvec, axis=0)
        featsvec_shifted = featsvec + (1 - min_featsvec)

        frac_gmm = mixtures.GMM(speaker, M, numceps, featsvec)
        frac_gmm.train(featsvec)
        ax = pl.subplot(2, 2, 1)
        ax.set_title('non-fractional', fontsize=FONTSIZE)
        set_plot_params(ax)
        plot_gmm(frac_gmm, featsvec, x_axis, y_axis)

        rs = [0.95, 0.8, 0.65]
        for (r, position) in zip(rs, range(2, 5)):
            print('r = %.2f' % r)
            featsvec_to_r = featsvec_shifted**r #- (1 - min_featsvec)

            frac_gmm = mixtures.GMM(speaker, M, numceps, featsvec, r=r)
            frac_gmm.train(featsvec)
            ax = pl.subplot(2, 2, position)
            ax.set_title('r = %.2f' % r, fontsize=FONTSIZE)
            set_plot_params(ax)
            plot_gmm(frac_gmm, [featsvec, featsvec_to_r], x_axis, y_axis,
                     param_feats=['b.', 'g.'])

        FILE_PATH = '../docs/presentation/images/frac-em-r-down.png'
        pl.savefig(FILE_PATH, bbox_inches='tight')
        pl.clf()

    elif command == 'frac-em-r-up':
        speaker = args[0]
        M = int(args[1])
        delta_order = int(args[2])
        x_axis = int(args[3])
        y_axis = int(args[4])

        featsvec = bases.read_speaker(numceps, delta_order, 'enroll_1', speaker,
                                      downlim='01', uplim='19')

        min_featsvec = np.amin(featsvec, axis=0)
        featsvec_shifted = featsvec + (1 - min_featsvec)

        frac_gmm = mixtures.GMM(speaker, M, numceps, featsvec)
        frac_gmm.train(featsvec)
        ax = pl.subplot(2, 2, 1)
        ax.set_title('non-fractional', fontsize=FONTSIZE)
        set_plot_params(ax)
        plot_gmm(frac_gmm, featsvec, x_axis, y_axis)

        rs = [1.05, 1.2, 1.35]
        for (r, position) in zip(rs, range(2, 5)):
            print('r = %.2f' % r)
            featsvec_to_r = featsvec_shifted**r #- (1 - min_featsvec)

            frac_gmm = mixtures.GMM(speaker, M, numceps, featsvec, r=r)
            frac_gmm.train(featsvec)
            ax = pl.subplot(2, 2, position)
            ax.set_title('r = %.2f' % r, fontsize=FONTSIZE)
            set_plot_params(ax)
            plot_gmm(frac_gmm, [featsvec, featsvec_to_r], x_axis, y_axis,
                     param_feats=['b.', 'g.'])

        FILE_PATH = '../docs/presentation/images/frac-em-r-up.png'
        pl.savefig(FILE_PATH, bbox_inches='tight')
        pl.clf()

    elif command == 'ubm':
        M = int(args[0])
        delta_order = int(args[1])
        x_axis = int(args[2])
        y_axis = int(args[3])

        featsvec_f = bases.read_background(numceps, delta_order, 'f', downlim='01',
                                           uplim='19')
        featsvec_m = bases.read_background(numceps, delta_order, 'm', downlim='01',
                                           uplim='19')

        # training
        D = numceps * (1 + delta_order)
        ubm_f = mixtures.GMM('f', M // 2, D, featsvec_f)
        ubm_f.train(featsvec_f)
        ubm_m = mixtures.GMM('m', M // 2, D, featsvec_m)
        ubm_m.train(featsvec_m)

        ax = pl.subplot(2, 2, 1)
        set_plot_params(ax)
        ax.set_title('female', size=10)
        plot_gmm(ubm_f, featsvec_f, x_axis, y_axis)
        ax = pl.subplot(2, 2, 2)
        set_plot_params(ax)
        ax.set_title('male', size=10)
        plot_gmm(ubm_m, featsvec_m, x_axis, y_axis, param_mix='y.')

        # combination
        ubm = ubm_f
        new_name = 'all_%d' % M
        ubm.absorb(ubm_m, new_name)

        featsvec = np.vstack((featsvec_f, featsvec_m))
        ax = pl.subplot(2, 2, 3)
        set_plot_params(ax)
        ax.set_title('female and male', size=10)
        plot_gmm(ubm, featsvec, x_axis, y_axis, param_mix=['r.', 'y.'])
        ax = pl.subplot(2, 2, 4)
        set_plot_params(ax)
        ax.set_title('combined UBM', size=10)
        plot_gmm(ubm, featsvec, x_axis, y_axis)

        FILE_PATH = '../docs/report/images/chapters/gmm/em_algorithm_ubm_%d.png' % M
        pl.savefig(FILE_PATH, bbox_inches='tight')

    elif command == 'adapt':
        speaker = args[0]
        M = int(args[1])
        delta_order = int(args[2])
        x_axis = int(args[3])
        y_axis = int(args[4])
        adaptations = args[5]

        featsvec_f = bases.read_background(numceps, delta_order, 'f', downlim='01',
                                           uplim='59')
        featsvec_m = bases.read_background(numceps, delta_order, 'm', downlim='01',
                                           uplim='59')
        featsvec = np.vstack((featsvec_f, featsvec_m))
        featsvec_speaker = bases.read_speaker(numceps, delta_order, 'enroll_1',
                                              speaker, downlim='01', uplim='59')

        UBMS_PATH = '%smit_%d_%d/' % (UBMS_DIR, numceps, delta_order)
        ubmfile = open('%sall_%d.gmm' % (UBMS_PATH, M), 'rb')
        ubm = pickle.load(ubmfile)
        ubmfile.close()

        GMMS_PATH = '%sadapted_%s/mit_%d_%d/' % (GMMS_DIR, adaptations, numceps, delta_order)
        gmmfile = open('%s%s_all_%d_%s.gmm' % (GMMS_PATH, speaker, M, adaptations), 'rb')
        gmm = pickle.load(gmmfile)
        gmmfile.close()

        ax = pl.subplot(2, 2, 1)
        set_plot_params(ax)
        plot_gmm(ubm, featsvec, x_axis, y_axis)
        ax = pl.subplot(2, 2, 2)
        set_plot_params(ax)
        plot_gmm(gmm, [featsvec, featsvec_speaker], x_axis, y_axis,
                 param_feats=['b.', 'g.'])

        FILE_PATH = '../docs/report/images/chapters/gmm/adapted_%s.png' % adaptations
        pl.savefig(FILE_PATH, bbox_inches='tight')

    t = time.time() - t
    print('Total time: %f seconds' % t)