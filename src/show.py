"""Module with code to exhibit data on the screen.
"""


import numpy as np
import pylab as pl
import pickle
from common import UBMS_DIR, GMMS_DIR
from matplotlib.patches import Ellipse


def plot_gmm(gmm, featsvec, x_axis=0, y_axis=1):
    """Plots a GMM and the vector of features used to train it. The plotting is
    in a 2D space.

    @param gmm: the GMM.
    @param featsvec: the vector of features.
    @param x_axis: the dimension plotted in the x axis.
    @param y_axis: the dimension plotted in the y axis.
    """
    pl.plot(featsvec[:, x_axis], featsvec[:, y_axis], 'bo')
    pl.plot(gmm.meansvec[:, x_axis], gmm.meansvec[:, y_axis], 'ro')

    ax = pl.gca()
    for (means, variances) in zip(gmm.meansvec, gmm.variancesvec):
        ellipse = Ellipse(xy=(means[x_axis], means[y_axis]), width=variances[x_axis]**0.5,
                          height=variances[y_axis]**0.5, edgecolor='r', linewidth=1,
                          fill=False, zorder=2)
        ax.add_artist(ellipse)


if __name__ == '__main__':
    import sys
    import bases
    import mixtures

    args = sys.argv[1:]
    numceps = 19
    M = int(args[0])
    delta_order = int(args[1])
    x_axis = int(args[2])
    y_axis = int(args[3])
    environment = args[4]

    configurations = {'office' : ('01', '19'), 'hallway' : ('21', '39'),
                      'intersection' : ('41', '59'), 'all' : ('01', '59')}
    dowlim = configurations[environment][0]
    uplim = configurations[environment][1]

    featsvec = bases.read_background(numceps, delta_order, None, dowlim, uplim)
    featsvec_f = bases.read_speaker(numceps, delta_order, 'enroll_1', 'f04', dowlim, uplim)
    featsvec_m = bases.read_speaker(numceps, delta_order, 'enroll_1', 'm10', dowlim, uplim)

    UBMS_PATH = '%smit_%d_%d/' % (UBMS_DIR, numceps, delta_order)
    ubm_file = open('%soffice_%d.ubm' % (UBMS_PATH, M), 'rb')
    ubm = pickle.load(ubm_file)

    # female
    GMMS_PATH = '%smit_%d_%d/' % (GMMS_DIR, numceps, delta_order)
    gmm_file = open('%sf04_office_%d.gmm' % (GMMS_PATH, M), 'rb')
    gmm = pickle.load(gmm_file)
    pl.subplot(2, 2, 1)
    plot_gmm(ubm, featsvec, x_axis, y_axis)
    pl.subplot(2, 2, 2)
    pl.plot(featsvec[:, x_axis], featsvec[:, y_axis], 'go')
    plot_gmm(gmm, featsvec_f, x_axis, y_axis)

    # male
    GMMS_PATH = '%smit_%d_%d/' % (GMMS_DIR, numceps, delta_order)
    gmm_file = open('%sm10_office_%d.gmm' % (GMMS_PATH, M), 'rb')
    gmm = pickle.load(gmm_file)
    pl.subplot(2, 2, 3)
    plot_gmm(ubm, featsvec, x_axis, y_axis)
    pl.subplot(2, 2, 4)
    pl.plot(featsvec[:, x_axis], featsvec[:, y_axis], 'go')
    plot_gmm(gmm, featsvec_m, x_axis, y_axis)

    #ubm = mixtures.GMM('test', M, numceps)
    #ubm.train(featsvec, use_kmeans=True, use_EM=False)
    #pl.figure()
    #pl.subplot(1, 2, 1)
    #plot_gmm(ubm, featsvec, x_axis, y_axis)
    #ubm.train(featsvec, use_kmeans=False, use_EM=True)
    #pl.subplot(1, 2, 2)
    #plot_gmm(ubm, featsvec, x_axis, y_axis)

    pl.show()