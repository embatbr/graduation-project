#!/usr/bin/python3.4

"""Module with code to exhibit data on the screen.
"""


import numpy as np
import pylab as pl
import pickle
from matplotlib.patches import Ellipse
from common import UBMS_DIR, GMMS_DIR, frange


def plot_gmm(gmm, featsvec, x_axis=0, y_axis=1, snd_featsvec=None):
    """Plots a GMM and the vector of features used to train it. The plotting is
    in a 2D space.

    @param gmm: the GMM.
    @param featsvec: the vector of features.
    @param x_axis: the dimension plotted in the x axis.
    @param y_axis: the dimension plotted in the y axis.
    """
    if not featsvec is None:
        pl.plot(featsvec[:, x_axis], featsvec[:, y_axis], 'b.')
    if not snd_featsvec is None:
        pl.plot(snd_featsvec[:, x_axis], snd_featsvec[:, y_axis], 'g.')
    pl.plot(gmm.meansvec[:, x_axis], gmm.meansvec[:, y_axis], 'r.')

    ax = pl.gca()
    for (means, variances) in zip(gmm.meansvec, gmm.variancesvec):
        ellipse = Ellipse(xy=(means[x_axis], means[y_axis]), width=variances[x_axis]**0.5,
                          height=variances[y_axis]**0.5, edgecolor='r', linewidth=1.5,
                          fill=False, zorder=2)
        ax.add_artist(ellipse)


if __name__ == '__main__':
    import sys
    import bases
    import mixtures

    command = sys.argv[1]
    args = sys.argv[2:]

    numceps = 19
    speaker = args[0]
    M = int(args[1])
    delta_order = int(args[2])
    x_axis = int(args[3])
    y_axis = int(args[4])

    if command == 'em':
        featsvec = bases.read_speaker(numceps, delta_order, 'enroll_1', speaker)

        gmm = mixtures.GMM(speaker, M, numceps, featsvec)
        pl.subplot(2, 2, 1)
        plot_gmm(gmm, featsvec, x_axis, y_axis)
        gmm.train(featsvec)
        pl.subplot(2, 2, 2)
        plot_gmm(gmm, featsvec, x_axis, y_axis)

        pl.savefig('../docs/paper/images/em_algorithm.png', bbox_inches='tight')
        pl.show()

    if command == 'frac-em':
        rs = frange(0.95, 1.06, 0.01)
        featsvec = bases.read_speaker(numceps, delta_order, 'enroll_1', speaker)

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
            featslist = bases.read_features_list(numceps, delta_order, 'enroll_2', speaker)
            log_likes = list()
            for feats in featslist:
                log_likes.append(frac_gmm.log_likelihood(feats))
            print('max = %f, min = %f' % (max(log_likes), min(log_likes)))

            FILE_PATH = '../docs/paper/images/em_algorithm_r%.2f.png' % r
            pl.savefig(FILE_PATH, bbox_inches='tight')
            pl.clf()

    if command == 'ubm':
        r = None
        if len(args) > 5:
            r = float(args[5])

        featsvec_f = bases.read_background(numceps, delta_order, 'f')
        featsvec_m = bases.read_background(numceps, delta_order, 'm')
        featsvec = np.vstack((featsvec_f, featsvec_m))

        # training
        D = numceps * (1 + delta_order)
        ubm_f = mixtures.GMM('f', M // 2, D, featsvec_f, r=r)
        ubm_f.train(featsvec)
        ubm_m = mixtures.GMM('m', M // 2, D, featsvec_m, r=r)
        ubm_m.train(featsvec)

        pl.subplot(1, 3, 1)
        plot_gmm(ubm_f, featsvec_f, x_axis, y_axis)
        pl.subplot(1, 3, 2)
        plot_gmm(ubm_m, featsvec_m, x_axis, y_axis)

        # combination
        ubm = ubm_f
        r_apx = '' if r is None else '_%.02f' % r
        new_name = 'all_%d%s' % (M, r_apx)
        ubm.absorb(ubm_m, new_name)

        pl.subplot(1, 3, 3)
        plot_gmm(ubm, featsvec, x_axis, y_axis)

        FILE_PATH = '../docs/paper/images/ubm_%d_%s.png' % (M, r_apx)
        pl.savefig(FILE_PATH, bbox_inches='tight')
        pl.show()

    if command == 'adapt':
        adaptations = args[5]

        featsvec_f = bases.read_background(numceps, delta_order, 'f')
        featsvec_m = bases.read_background(numceps, delta_order, 'm')
        featsvec = np.vstack((featsvec_f, featsvec_m))
        featsvec_speaker = bases.read_speaker(numceps, delta_order, 'enroll_1', speaker)

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
        plot_gmm(gmm, featsvec, x_axis, y_axis, featsvec_speaker)

        FILE_PATH = '../docs/paper/images/adapted_%s_%s.png' % (speaker, adaptations)
        pl.savefig(FILE_PATH, bbox_inches='tight')
        pl.show()