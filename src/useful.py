"""This modules just contains global definitions and functions
"""


import numpy as np
import matplotlib.pyplot as plt


BASES_DIR = '../bases/'
CORPORA_DIR = '%scorpora/' % BASES_DIR
FEATURES_DIR = '%sfeatures/' % BASES_DIR


def testplot(x, y, newfig=True, suptitle='', xlabel='', ylabel='', fill=False):
    """Creates a Matplotlib figure and plots the @param y related to @param x.

    @param x: a numpy array.
    @param y: a numpy array of the same size of @param x.
    @param suptitle: the title of the figure.
    @param xlabel: the label of the x axis.
    @param ylabel: the label of the y axis.
    @param xlim: the limits of the x axis.
    @param ylim: the limits of the y axis.
    """
    if newfig:
        fig = plt.figure()
        fig.suptitle(suptitle)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim([x[0], x[-1]])
    plt.grid(True)
    if fill:
        plt.fill_between(x, y)
    else:
        plt.plot(x, y)