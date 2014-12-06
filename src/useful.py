"""This modules just contains global definitions and functions
"""


import numpy as np
import matplotlib.pyplot as plt


BASES_DIR = '../bases/'
CORPORA_DIR = '%scorpora/' % BASES_DIR
FEATURES_DIR = '%sfeatures/' % BASES_DIR


def testplot(x, y, suptitle='', xlabel='', ylabel=''):
    """Creates a Matplotlib figure and plots the @param y related to @param x.

    @param x: a numpy array.
    @param y: a numpy array.
    @param suptitle: the title of the figure.
    @param xlabel: the label of the x axis.
    @param ylabel: the label of the y axis.
    @param xlim: the limits of the x axis.
    @param ylim: the limits of the y axis.
    """
    fig = plt.figure()
    fig.suptitle(suptitle)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.xlim([x[0], len(x)])
    plt.ylim([np.amin(y), np.amax(y)])

    plt.grid(True)
    plt.plot(x, y)