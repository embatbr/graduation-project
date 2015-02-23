"""Module for functions and variables used in other modules.
"""


import numpy as np


BASES_DIR = '../bases/'
CORPORA_DIR = '%scorpora/' % BASES_DIR
FEATURES_DIR = '%sfeatures/' % BASES_DIR
GMMS_DIR = '%sgmms/' % BASES_DIR

FLOAT_MIN = np.finfo(np.float64).min
ZERO = 1e-323
EPS = np.finfo(np.float64).eps