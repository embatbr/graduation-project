"""This modules just contains global definitions and functions
"""


import numpy as np
import matplotlib.pyplot as plt


BASES_DIR = '../bases/'
CORPORA_DIR = '%scorpora/' % BASES_DIR
FEATURES_DIR = '%sfeatures/' % BASES_DIR
IMAGES_DIR = '../docs/paper/images/'
IMAGES_SIGPROC_DIR = '%ssigproc' % IMAGES_DIR


def testplot(x, y, suptitle='', xlabel='', ylabel='', filename=None, fill=False,
             color='blue'):
    """Creates a Matplotlib figure and plots the @param y related to @param x.

    @param x: a numpy array.
    @param y: a numpy array of the same size of @param x.
    @param suptitle: the title of the figure.
    @param xlabel: the label of the x axis.
    @param ylabel: the label of the y axis.
    @param xlim: the limits of the x axis.
    @param ylim: the limits of the y axis.
    """
    plt.clf()
    fig = plt.gcf()
    fig.suptitle(suptitle)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    #plt.xlim([x[0], x[-1]])
    plt.grid(True)
    if fill:
        plt.fill_between(x, y, color=color)
    plt.plot(x, y, color=color)
    if not filename is None:
        plt.savefig('%s%s.png' % (IMAGES_DIR, filename))