#!/usr/bin/python3.4

"""Module with code to exhibit data on the screen.
"""


import numpy as np
import pylab as pl
import scipy.io.wavfile as wavf
import pickle
import time
from matplotlib.patches import Ellipse
from common import UBMS_DIR, GMMS_DIR, frange
import features


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



if __name__ == '__main__':
    import sys
    import bases
    import mixtures

    command = sys.argv[1]
    args = sys.argv[2:] if len(sys.argv) > 2 else None

    numceps = 19
    show = True
    t = time.time()

    if command == 'utterance':
        show = False

        SIGNAL_PATH = '../bases/corpora/mit/enroll_1/f00/phrase02_16k.wav'
        (samplerate, signal) = wavf.read(SIGNAL_PATH)

        # plotting utterance "karen livescu"
        duration = np.linspace(0, len(signal) / samplerate, len(signal))
        ax = pl.subplot(3, 1, 1)
        pl.grid(True)
        [tick.label.set_fontsize(10) for tick in ax.xaxis.get_major_ticks()]
        [tick.label.set_fontsize(10) for tick in ax.yaxis.get_major_ticks()]
        pl.plot(duration, signal, 'b')
        pl.savefig('../docs/paper/images/speech_signal.png', bbox_inches='tight')

    elif command == 'mfcc':
        show = False
        nfilt = 26
        preemph = 0.97
        NFFT = 512
        winlen = 0.02
        winstep = 0.01

        SIGNAL_PATH = '../bases/corpora/mit/enroll_1/f00/phrase02_16k.wav'
        (samplerate, signal) = wavf.read(SIGNAL_PATH)

        # hertz-mel relation
        lowfreq_mel = 0
        highfreq_mel = features.hz2mel(samplerate/2)
        melpoints = np.linspace(lowfreq_mel, highfreq_mel, nfilt + 2) # equally spaced in mel scale
        hzpoints = features.mel2hz(melpoints)

        # plotting hertz-mel relation
        ax = pl.subplot(2, 1, 1)
        pl.grid(True)
        ticks = np.arange(0, samplerate/2 + 1, 1000)
        pl.xticks(ticks)
        [tick.label.set_fontsize(10) for tick in ax.xaxis.get_major_ticks()]
        [tick.label.set_fontsize(10) for tick in ax.yaxis.get_major_ticks()]
        pl.plot(hzpoints, melpoints, 'r')
        pl.savefig('../docs/paper/images/mel_scale.png', bbox_inches='tight')
        pl.clf()

        # plotting signal and preemphasized signal
        duration = np.linspace(0, len(signal) / samplerate, len(signal))
        ax = pl.subplot(3, 2, 1)
        ax.set_title('signal', fontsize=10)
        pl.grid(True)
        [tick.label.set_fontsize(10) for tick in ax.xaxis.get_major_ticks()]
        [tick.label.set_fontsize(10) for tick in ax.yaxis.get_major_ticks()]
        pl.plot(duration, signal, 'b')
        ax = pl.subplot(3, 2, 2)
        ax.set_title('pre-emphasized signal', fontsize=10)
        pl.grid(True)
        [tick.label.set_fontsize(10) for tick in ax.xaxis.get_major_ticks()]
        [tick.label.set_fontsize(10) for tick in ax.yaxis.get_major_ticks()]
        emph_signal = features.preemphasis(signal, preemph)
        pl.plot(duration, emph_signal, 'b')

        # plotting spectrum
        frequencies = np.linspace(0, samplerate//2, NFFT//2 + 1)
        ticks = np.arange(0, samplerate//2 + 1, 2000)
        ax = pl.subplot(3, 2, 3)
        pl.subplots_adjust(hspace=0.4)
        ax.set_title('signal\'s spectrum', fontsize=10)
        pl.grid(True)
        pl.xticks(ticks)
        [tick.label.set_fontsize(10) for tick in ax.xaxis.get_major_ticks()]
        [tick.label.set_fontsize(10) for tick in ax.yaxis.get_major_ticks()]
        magspec = features.magspec(signal)
        pl.fill_between(frequencies, magspec, edgecolor='red', facecolor='red')
        ax = pl.subplot(3, 2, 4)
        pl.subplots_adjust(hspace=0.4)
        ax.set_title('pre-emphasized signal\'s spectrum', fontsize=10)
        pl.grid(True)
        pl.xticks(ticks)
        [tick.label.set_fontsize(10) for tick in ax.xaxis.get_major_ticks()]
        [tick.label.set_fontsize(10) for tick in ax.yaxis.get_major_ticks()]
        emph_magspec = features.magspec(emph_signal)
        pl.fill_between(frequencies, emph_magspec, edgecolor='red', facecolor='red')
        pl.savefig('../docs/paper/images/preemphasis.png', bbox_inches='tight')
        pl.clf()

        # plotting framing
        frames = features.framesignal(emph_signal, winlen*samplerate, winstep*samplerate)
        frames = frames[50]
        begin = int(50*winstep*samplerate)
        end = int((50*winstep + winlen)*samplerate)
        length = int(winlen*samplerate)
        duration = np.linspace(begin, end, length)
        ticks = np.arange(begin, end, 50)
        ax = pl.subplot(3, 1, 1)
        pl.grid(True)
        pl.xticks(ticks)
        [tick.label.set_fontsize(10) for tick in ax.xaxis.get_major_ticks()]
        [tick.label.set_fontsize(10) for tick in ax.yaxis.get_major_ticks()]
        pl.plot(duration, frames, 'g')
        pl.savefig('../docs/paper/images/framing.png', bbox_inches='tight')

        # TODO gerar figura 'fft' do capítulo 3

    elif command == 'em':
        speaker = args[0]
        M = int(args[1])
        delta_order = int(args[2])
        x_axis = int(args[3])
        y_axis = int(args[4])
        if len(args) > 5:
            show = False if args[5].lower() == 'false' else show

        featsvec = bases.read_speaker(numceps, delta_order, 'enroll_1', speaker,
                                      downlim='01', uplim='59')

        gmm = mixtures.GMM(speaker, M, numceps, featsvec)
        pl.subplot(2, 2, 1)
        plot_gmm(gmm, featsvec, x_axis, y_axis)
        gmm.train(featsvec)
        pl.subplot(2, 2, 2)
        plot_gmm(gmm, featsvec, x_axis, y_axis)

        pl.savefig('../docs/paper/images/em_algorithm.png', bbox_inches='tight')

    elif command == 'frac-em':
        speaker = args[0]
        M = int(args[1])
        delta_order = int(args[2])
        x_axis = int(args[3])
        y_axis = int(args[4])

        show = False
        rs = frange(0.95, 1.06, 0.01)
        featsvec = bases.read_speaker(numceps, delta_order, 'enroll_1', speaker,
                                      downlim='01', uplim='19')

        untrained_gmm = mixtures.GMM(speaker, M, numceps, featsvec)
        trained_gmm = untrained_gmm.clone(featsvec)
        trained_gmm.train(featsvec)

        for r in rs:
            print('\nr = %.02f' % r)
            pl.subplot(2, 2, 1)
            plot_gmm(untrained_gmm, featsvec, x_axis, y_axis)
            pl.subplot(2, 2, 2)
            plot_gmm(trained_gmm, featsvec, x_axis, y_axis)

            print('Fractional')
            frac_gmm = mixtures.GMM(speaker, M, numceps, featsvec, r=r)
            pl.subplot(2, 2, 3)
            plot_gmm(frac_gmm, featsvec, x_axis, y_axis)
            frac_gmm.train(featsvec)
            pl.subplot(2, 2, 4)
            plot_gmm(frac_gmm, featsvec, x_axis, y_axis)

            print('Fractional likelihoods')
            featslist = bases.read_features_list(numceps, delta_order, 'enroll_2',
                                                 speaker, downlim='01', uplim='19')
            log_likes = list()
            for feats in featslist:
                log_likes.append(frac_gmm.log_likelihood(feats))
            print('max = %f, min = %f' % (max(log_likes), min(log_likes)))

            FILE_PATH = '../docs/paper/images/em_algorithm_r%.2f.png' % r
            pl.savefig(FILE_PATH, bbox_inches='tight')
            pl.clf()

    elif command == 'ubm':
        M = int(args[0])
        delta_order = int(args[1])
        x_axis = int(args[2])
        y_axis = int(args[3])
        if len(args) > 4:
            show = False if args[4].lower() == 'false' else show

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
        ax.set_title('female', fontsize=10)
        plot_gmm(ubm_f, featsvec_f, x_axis, y_axis)
        ax = pl.subplot(2, 2, 2)
        ax.set_title('male', fontsize=10)
        plot_gmm(ubm_m, featsvec_m, x_axis, y_axis, param_mix='y.')

        # combination
        ubm = ubm_f
        new_name = 'all_%d' % M
        ubm.absorb(ubm_m, new_name)

        featsvec = np.vstack((featsvec_f, featsvec_m))
        ax = pl.subplot(2, 2, 3)
        ax.set_title('female and male', fontsize=10)
        plot_gmm(ubm, featsvec, x_axis, y_axis, param_mix=['r.', 'y.'])
        ax = pl.subplot(2, 2, 4)
        ax.set_title('combined UBM', fontsize=10)
        plot_gmm(ubm, featsvec, x_axis, y_axis)

        FILE_PATH = '../docs/paper/images/em_algorithm_ubm_%d.png' % M
        pl.savefig(FILE_PATH, bbox_inches='tight')

    elif command == 'adapt':
        speaker = args[0]
        M = int(args[1])
        delta_order = int(args[2])
        x_axis = int(args[3])
        y_axis = int(args[4])
        adaptations = args[5]
        if len(args) > 6:
            show = False if args[6].lower() == 'false' else show

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

        pl.subplot(2, 2, 1)
        plot_gmm(ubm, featsvec, x_axis, y_axis)
        pl.subplot(2, 2, 2)
        plot_gmm(gmm, [featsvec, featsvec_speaker], x_axis, y_axis,
                 param_feats=['b.', 'g.'])

        FILE_PATH = '../docs/paper/images/adapted_%s.png' % adaptations
        pl.savefig(FILE_PATH, bbox_inches='tight')

    t = time.time() - t
    print('Total time: %f seconds' % t)

    if show:
        pl.show()