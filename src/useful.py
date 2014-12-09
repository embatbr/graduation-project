"""This modules just contains global definitions and functions
"""


import numpy as np
import matplotlib.pyplot as plt


BASES_DIR = '../bases/'
CORPORA_DIR = '%scorpora/' % BASES_DIR
FEATURES_DIR = '%sfeatures/' % BASES_DIR
IMAGES_DIR = '../docs/paper/images/'


def testplot(x, y, suptitle='', xlabel='', ylabel='', filename=None, filecounter=0,
             color='blue', fill=False):
    """Creates a Matplotlib figure and plots the @param y related to @param x.

    @param x: a numpy array.
    @param y: a numpy array of the same size of @param x.
    @param suptitle: the title of the figure.
    @param xlabel: the label of the x axis.
    @param ylabel: the label of the y axis.
    @param ylim: the limits of the y axis.
    """
    plt.clf()
    fig = plt.gcf()
    fig.suptitle(suptitle)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)

    if fill:
        plt.fill_between(x, y, color=color)
    plt.plot(x, y, color=color)

    if not filename is None:
        plt.savefig('%s%03d.png' % (filename, filecounter))

    return (filecounter + 1)

def testmultiplot(x, y, suptitle='', xlabel='', ylabel='', filename=None, color='blue'):
    """Creates a Matplotlib figure and plots the @param y related to @param x.

    @param x: a numpy array.
    @param y: a numpy array of the same size of @param x.
    @param suptitle: the title of the figure.
    @param xlabel: the label of the x axis.
    @param ylabel: the label of the y axis.
    @param ylim: the limits of the y axis.
    """
    plt.clf()
    fig = plt.gcf()
    fig.suptitle(suptitle)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)

    for i in range(len(y)):
        plt.plot(x, y[i], color=color)

    if not filename is None:
        plt.savefig('%s.png' % filename)