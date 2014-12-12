"""This modules just contains global definitions and functions
"""


import numpy as np
import matplotlib.pyplot as plt
import math


BASES_DIR = '../bases/'
CORPORA_DIR = '%scorpora/' % BASES_DIR
FEATURES_DIR = '%sfeatures/' % BASES_DIR
TEST_IMAGES_DIR = '../docs/paper/images/test/'


def plot(x, y, suptitle='', xlabel='', ylabel='', color='blue', fill=False,
         linestyle='-'):
    fig = plt.gcf()
    fig.suptitle(suptitle)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)

    if fill:
        plt.fill_between(x, y, color=color)
    plt.plot(x, y, color=color, linestyle=linestyle)

def plotfile(x, y, suptitle='', xlabel='', ylabel='', filename=None, filecounter=0,
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