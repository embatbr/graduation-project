"""Module with code to exhibit data on the screen.
"""


import numpy as np
import pylab as pl
import pickle
from common import GMMS_DIR
from matplotlib.patches import Ellipse


def plot_gmm(gmm, featsvec, x_axis=0, y_axis=1):
    """Plots a GMM and the vector of features used to train it. The plotting is
    in a 2D space.

    @param gmm: the GMM.
    @param featsvec: the vector of features.
    @param x_axis: the dimension plotted in the x axis.
    @param y_axis: the dimension plotted in the y axis.
    """
    pl.plot(featsvec[:, x_axis], featsvec[:, y_axis], 'b.')
    pl.plot(gmm.meansvec[:, x_axis], gmm.meansvec[:, y_axis], 'r.')

    ax = pl.gca()
    for (means, variances) in zip(gmm.meansvec, gmm.variancesvec):
        for (i, color) in zip(range(1, 4), ['r', 'y', 'm']):
            ellipse = Ellipse(xy=(means[x_axis], means[y_axis]), width=i*variances[x_axis]**0.5,
                              height=i*variances[y_axis]**0.5, edgecolor=color,
                              linewidth=1, fill=False, zorder=2)
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

    featsvec_f = bases.read_background(numceps, delta_order, 'f', '01', '19')
    featsvec_m = bases.read_background(numceps, delta_order, 'm', '01', '19')
    featsvec = np.vstack((featsvec_f, featsvec_m))

    PATH = '%smit_%d_%d/' % (GMMS_DIR, numceps, delta_order)
    ubm_file = open('%soffice_%d.ubm' % (PATH, M), 'rb')
    ubm = pickle.load(ubm_file)
    plot_gmm(ubm, featsvec, x_axis, y_axis)

    #ubm = mixtures.GMM('test', M, numceps)
    #ubm.train(featsvec, use_kmeans=True, use_EM=False)
    #pl.figure()
    #pl.subplot(1, 2, 1)
    #plot_gmm(ubm, featsvec, x_axis, y_axis)
    #ubm.train(featsvec, use_kmeans=False, use_EM=True)
    #pl.subplot(1, 2, 2)
    #plot_gmm(ubm, featsvec, x_axis, y_axis)

    pl.show()