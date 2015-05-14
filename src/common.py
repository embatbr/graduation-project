"""Module for functions and variables used in other modules.
"""


import numpy as np
import math


BASES_DIR = '../bases/'
CORPORA_DIR = '%scorpora/' % BASES_DIR
FEATURES_DIR = '%sfeatures/' % BASES_DIR
GMMS_DIR = '%sgmms/' % BASES_DIR
SPEAKERS_DIR = '%sspeakers/' % GMMS_DIR
UBMS_DIR = '%subms/' % GMMS_DIR
FRAC_GMMS_DIR = '%sfrac-gmms/' % BASES_DIR
FRAC_UBMS_DIR = '%subms/' % FRAC_GMMS_DIR

EXPERIMENTS_DIR = '../experiments/'
VERIFY_DIR = '%sverify/' % EXPERIMENTS_DIR

CHECK_DIR = '../check/'

FLOAT_MIN = np.finfo(np.float64).min # -1.7976931348623157e+308
FLOAT_MAX = np.finfo(np.float64).max # +1.7976931348623157e+308
ZERO = 1e-323
EPS = np.finfo(np.float64).eps # 2.2204460492503131e-16
MIN_VARIANCE = 1e-2


def isequal(A, B):
    return math.fabs(A - B) >= 10*EPS

def calculate_eer(false_detection, false_rejection):
    false_detection = np.array(false_detection)
    false_rejection = np.array(false_rejection)
    diff = false_detection - false_rejection
    EER_index = np.argmin(np.fabs(diff))

    EER = (false_detection[EER_index] + false_rejection[EER_index]) / 2
    return (EER, EER_index)