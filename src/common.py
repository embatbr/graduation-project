"""Module for functions and variables used in other modules.
"""


import numpy as np


BASES_DIR = '../bases/'
CORPORA_DIR = '%scorpora/' % BASES_DIR
FEATURES_DIR = '%sfeatures/' % BASES_DIR
GMMS_DIR = '%sgmms/' % BASES_DIR

EXPERIMENTS_DIR = '../experiments/'
EXP_VERIFICATION_DIR = '%sverification/' % EXPERIMENTS_DIR

FLOAT_MIN = np.finfo(np.float64).min # -1.7976931348623157e+308
FLOAT_MAX = np.finfo(np.float64).max # +1.7976931348623157e+308
ZERO = 1e-323
EPS = np.finfo(np.float64).eps # 2.2204460492503131e-16
MIN_VARIANCE = 1e-2