"""Module with code to exhibit data on the screen.
"""


import numpy as np
import pylab as pl
import pickle
from matplotlib.patches import Ellipse
from common import UBMS_DIR, GMMS_DIR


def plot_gmm(gmm, featsvec, x_axis=0, y_axis=1, snd_featsvec=None):
    """Plots a GMM and the vector of features used to train it. The plotting is
    in a 2D space.

    @param gmm: the GMM.
    @param featsvec: the vector of features.
    @param x_axis: the dimension plotted in the x axis.
    @param y_axis: the dimension plotted in the y axis.
    """
    if not featsvec is None:
        pl.plot(featsvec[:, x_axis], featsvec[:, y_axis], 'bo')
    if not snd_featsvec is None:
        pl.plot(snd_featsvec[:, x_axis], snd_featsvec[:, y_axis], 'go')
    pl.plot(gmm.meansvec[:, x_axis], gmm.meansvec[:, y_axis], 'ro')

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
        gmm.train(featsvec, use_kmeans=True, use_EM=True)
        pl.subplot(2, 2, 2)
        plot_gmm(gmm, featsvec, x_axis, y_axis)

        pl.savefig('../docs/paper/images/em_algorithm.png', bbox_inches='tight')
        pl.show()

    if command == 'frac-em':
        r = float(args[5])

        featsvec = bases.read_speaker(numceps, delta_order, 'enroll_1', speaker)

        gmm = mixtures.GMM(speaker, M, numceps, featsvec)
        pl.subplot(2, 2, 1)
        plot_gmm(gmm, featsvec, x_axis, y_axis)
        gmm.train(featsvec, use_kmeans=True, use_EM=True)
        pl.subplot(2, 2, 2)
        plot_gmm(gmm, featsvec, x_axis, y_axis)

        min_featsvec = np.amin(featsvec, axis=0)
        featsvec_draw = featsvec + (1 - min_featsvec)

        gmm = mixtures.GMM(speaker, M, numceps, featsvec)
        pl.subplot(2, 2, 3)
        plot_gmm(gmm, featsvec, x_axis, y_axis)
        gmm.train(featsvec, r=r, use_kmeans=True, use_EM=True)
        pl.subplot(2, 2, 4)
        plot_gmm(gmm, featsvec, x_axis, y_axis)

        featslist = bases.read_features_list(numceps, delta_order, 'enroll_2', speaker)
        log_likes = list()
        for feats in featslist:
            log_likes.append(gmm.log_likelihood(feats))
        print(log_likes)

        pl.savefig('../docs/paper/images/frac-em_algorithm.png', bbox_inches='tight')
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

        pl.savefig('../docs/paper/images/adapted_%s.png' % adaptations, bbox_inches='tight')
        pl.show()