"""Module for functions and variables used in other modules.
"""


import numpy as np
import math
import sys


BASES_DIR = '../bases/'
CORPORA_DIR = '%scorpora/' % BASES_DIR
FEATURES_DIR = '%sfeatures/' % BASES_DIR
GMMS_DIR = '%sgmms/' % BASES_DIR
SPEAKERS_DIR = '%sspeakers/' % GMMS_DIR
UBMS_DIR = '%subms/' % GMMS_DIR
FRAC_GMMS_DIR = '%sfrac-gmms/' % BASES_DIR
FRAC_SPEAKERS_DIR = '%sspeakers/' % FRAC_GMMS_DIR
FRAC_UBMS_DIR = '%subms/' % FRAC_GMMS_DIR

DOCS_DIR = '../docs/'
REPORT_DIR = '%sreport/' % DOCS_DIR
CHAPTERS_DIR = '%schapters/' % REPORT_DIR
TABLES_DIR = '%stables/' % CHAPTERS_DIR

EXPERIMENTS_DIR = '../experiments/'
VERIFY_DIR = '%sverify/' % EXPERIMENTS_DIR
IDENTIFY_DIR = '%sidentify/' % EXPERIMENTS_DIR

CHECK_DIR = '../check/'

EM_THRESHOLD = 1E-3

NUM_SPEAKERS = 48
NUM_UTTERANCES_PER_ENROLLED_SPEAKER = 54
NUM_ENROLLED_UTTERANCES = NUM_SPEAKERS * NUM_UTTERANCES_PER_ENROLLED_SPEAKER

FLOAT_MIN = np.finfo(np.float64).min # -1.7976931348623157e+308
FLOAT_MAX = np.finfo(np.float64).max # +1.7976931348623157e+308
INT_MAX = sys.maxsize
ZERO = 1e-323
EPS = np.finfo(np.float64).eps # 2.2204460492503131e-16
MIN_VARIANCE = 1e-2


def frange(start, stop, step):
    ret = list()

    while start < stop:
        ret.append(start)
        start = start + step

    return ret

def isequal(A, B):
    return math.fabs(A - B) >= 10*EPS

def calculate_eer(false_detection, false_rejection):
    false_detection = np.array(false_detection)
    false_rejection = np.array(false_rejection)
    diff = false_detection - false_rejection
    EER_index = np.argmin(np.fabs(diff))

    EER = (false_detection[EER_index] + false_rejection[EER_index]) / 2
    return (EER, EER_index)