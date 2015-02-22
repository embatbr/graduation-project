#!/usr/bin/python3.4

"""Module to execute all important things from project.
"""


import sys
import os, os.path
import shutil
from common import FEATURES_DIR
import corpus


commands = sys.argv[1:]

delta_orders = [0, 1, 2]

#Extract the MFCCs from base MIT and put in the correct format.
if 'extract-features' in commands:
    if os.path.exists(FEATURES_DIR):
        shutil.rmtree(FEATURES_DIR)
    os.mkdir(FEATURES_DIR)

    print('FEATURE EXTRACTION')

    winlen = 0.02
    winstep = 0.01
    numcep = 13
    for delta_order in delta_orders:
        corpus.extract_mit(winlen, winstep, numcep, delta_order)