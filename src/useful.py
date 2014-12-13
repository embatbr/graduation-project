"""This modules just contains global definitions and functions
"""


import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import math


BASES_DIR = '../bases/'
CORPORA_DIR = '%scorpora/' % BASES_DIR
FEATURES_DIR = '%sfeatures/' % BASES_DIR
GMMS_DIR = '%sgmms/' % BASES_DIR
TEST_IMAGES_DIR = '../docs/paper/images/test/'


def plot(x, y, suptitle='', xlabel='', ylabel='', color='blue', fill=False,
         linestyle='-'):
    """Plots the numpy array y related to numpy array x.

    @param x: a numpy array.
    @param y: a numpy array of the same size of @param x.
    @param suptitle: the title of the figure.
    @param xlabel: the label of the x axis.
    @param ylabel: the label of the y axis.
    @param color: the color of line (and area filled).
    @param fill: to fill the area beneath the curve.
    @param linestyle: the style of the line (dashed, dotted, etc.).
    """
    fig = plt.gcf()
    fig.suptitle(suptitle)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)

    if fill:
        plt.fill_between(x, y, color=color)
    plt.plot(x, y, color=color, linestyle=linestyle)

def plotfigure(x, y, suptitle='', xlabel='', ylabel='', filename=None, filecounter=0,
               color='blue', fill=False, xlim=True):
    """Creates a Matplotlib figure and plots the @param y related to @param x.

    @param x: a numpy array.
    @param y: a numpy array of the same size of @param x.
    @param suptitle: the title of the figure.
    @param xlabel: the label of the x axis.
    @param ylabel: the label of the y axis.
    @param filename: name of file to plot.
    @param filecounter: composes the final filename.
    @param color: the color of line (and area filled).
    @param fill: to fill the area beneath the curve.
    @param xlim: if True, limits the x-axis.
    """
    plt.clf()
    if xlim:
        plt.xlim(x[0], x[-1])
    if y.ndim == 1:
        plot(x, y, suptitle, xlabel, ylabel, color, fill)
    else:
        for yval in y:
            plot(x, yval, suptitle, xlabel, ylabel, color, fill)

    if not filename is None:
        plt.savefig('%s%05d.png' % (filename, filecounter))
        return (filecounter + 1)

    return filecounter

def plotpoints(x, y, suptitle='', xlabel='', ylabel='', filename=None, filecounter=0,
               color='blue'):
    """Creates a Matplotlib figure and plots the @param y related to @param x as
    points.

    @param x: a numpy array.
    @param y: a numpy array of the same size of @param x.
    @param suptitle: the title of the figure.
    @param xlabel: the label of the x axis.
    @param ylabel: the label of the y axis.
    @param filename: name of file to plot.
    @param filecounter: composes the final filename.
    @param color: the color of line (and area filled).
    """
    plt.clf()
    plot(x, y, suptitle, xlabel, ylabel, color, False, ':')

    if not filename is None:
        plt.savefig('%s%05d.png' % (filename, filecounter))
        return (filecounter + 1)

    return filecounter

def plotgaussian(x, y, mean, variance, suptitle='', xlabel='', ylabel='',
                 filename=None, filecounter=0):
    """Creates a Matplotlib figure and plots the @param y related to @param x as
    points. It also draws the real and desired gaussians.

    @param x: a numpy array.
    @param y: a numpy array of the same size of @param x.
    @param suptitle: the title of the figure.
    @param xlabel: the label of the x axis.
    @param ylabel: the label of the y axis.
    @param filename: name of file to plot.
    @param filecounter: composes the final filename.
    """
    plt.clf()
    plot(x, y, suptitle, xlabel, ylabel, 'blue', False, ':')

    #The gaussian given by 'x'
    x = np.sort(x)
    rgauss = stats.norm.pdf(x, np.mean(x), np.std(x))
    rgauss = (np.amax(y) - np.amin(y)) * rgauss
    plt.plot(x, rgauss, color='red')

    #The gaussian given by 'mean' and 'variance'
    x = np.sort(x)
    dgauss = stats.norm.pdf(x, mean, variance**0.5)
    dgauss = (np.amax(y) - np.amin(y)) * dgauss
    plt.plot(x, dgauss, color='magenta')

    if not filename is None:
        plt.savefig('%s%05d.png' % (filename, filecounter))
        return (filecounter + 1)

    return filecounter

def plotgmm(x, gmm, featnum, suptitle='', xlabel='', ylabel='', filename=None,
            filecounter=0):
    """Creates a Matplotlib figure and plots the GMM's gaussians weighted.

    @param gmm: the GMM, a list of tuples (weight, means, covmatrix).
    @param suptitle: the title of the figure.
    @param xlabel: the label of the x axis.
    @param ylabel: the label of the y axis.
    @param filename: name of file to plot.
    @param filecounter: composes the final filename.
    """
    plt.clf()
    plt.grid(True)

    gaussfull = 0
    for (weight, means, covmatrix) in gmm:
        mean = means[featnum]
        variance = covmatrix[featnum][featnum]

        x = np.sort(x)
        gauss = stats.norm.pdf(x, mean, variance**0.5)
        gauss = weight*gauss
        gaussfull = gaussfull + gauss
        plot(x, gauss, suptitle, xlabel, ylabel, 'red', linestyle='--')

    plot(x, gaussfull, suptitle, xlabel, ylabel, 'blue')

    if not filename is None:
        plt.savefig('%s%05d.png' % (filename, filecounter))
        return (filecounter + 1)

    return filecounter