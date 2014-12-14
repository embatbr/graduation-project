"""Module to execute all important things from project.
"""


import sys
import os, os.path
import shutil
from useful import FEATURES_DIR, GMMS_DIR, IDENTIFY_DIR
import corpus, mixtures
import numpy as np
import pickle


command = sys.argv[1]
params = sys.argv[2:]

#Extract the MFCCs from base MIT and put in the correct format.
if command == 'extract-features':
    if os.path.exists(FEATURES_DIR):
        shutil.rmtree(FEATURES_DIR)

    print('FEATURE EXTRACTION')

    winlen = float(params[0])       #default 0.02
    winstep = float(params[1])      #default 0.01
    numcep = float(params[2])       #default 13
    numdeltas = float(params[3])    #0, 1 or 2
    corpus.mit_features(winlen, winstep, numcep, numdeltas)

#Train each speaker's GMM given the numbers of cepstral coefficients (numcep),
#deltas (numdeltas) and mixtures (M).
elif command == 'train-gmms':
    if os.path.exists(GMMS_DIR):
        shutil.rmtree(GMMS_DIR)
    os.mkdir(GMMS_DIR)

    (numcep, numdeltas, M) = (int(params[0]), int(params[1]), int(params[2]))
    print('SPEAKER GMMs TRAINING\nGMM BASE: mit_%d_%d\nM = %d' % (numcep, numdeltas, M))

    gmmspath = '%smit_%d_%d/' % (GMMS_DIR, numcep, numdeltas)
    os.mkdir(gmmspath)
    speakers = os.listdir('%smit_%d_%d/enroll_1/' % (FEATURES_DIR, numcep,
                                                     numdeltas))
    speakers.sort()
    numfeats = numcep*(numdeltas + 1)

    for speaker in speakers:
        mfccs = corpus.read_speaker_features(numcep, numdeltas, speaker)
        print('SPEAKER: %s\nmfccs: %s' % (speaker, mfccs.shape))
        oldgmm = mixtures.create_gmm(M, numfeats)
        newgmm = mixtures.train_gmm(oldgmm, mfccs)
        gmmfile = open('%s%s_%d.gmm' % (gmmspath, speaker, M), 'wb')
        pickle.dump(newgmm, gmmfile)

#Identify each speaker's speech given the numbers of cepstral coefficients (numcep),
#deltas (numdeltas) and mixtures (M). The test is discriminate gender.
elif command == 'identify-speakers':
    if os.path.exists(IDENTIFY_DIR):
        shutil.rmtree(IDENTIFY_DIR)
    os.mkdir(IDENTIFY_DIR)

    (numcep, numdeltas, M) = (int(params[0]), int(params[1]), int(params[2]))
    print('IDENTIFYING SPEAKER GMMS\nmit_%d_%d\nM = %d' % (numcep, numdeltas, M))

    exppath = '%smit_%d_%d_%d.exp' % (IDENTIFY_DIR, numcep, numdeltas, M)
    expfile = open(exppath, 'w')
    expfile.write('NUMCEP %d\nNUMDELTAS %d\n' % (numcep, numdeltas))
    numfeats = numcep*(numdeltas + 1)

    gmmspath = '%smit_%d_%d/' % (GMMS_DIR, numcep, numdeltas)
    gmmfilenames = os.listdir(gmmspath)
    gmmfilenames.sort()
    gmmlist = list()
    for gmmfilename in gmmfilenames:
        gmmfile = open('%smit_%d_%d/%s' % (GMMS_DIR, numcep, numdeltas,
                                              gmmfilename), 'rb')
        gmm = pickle.load(gmmfile)
        gmmlist.append(gmm)

    for dataset in ['enroll_2', 'imposter']:
        print('DATASET %s' % dataset)
        expfile.write('DATASET %s\n' % dataset)
        datasetpath = '%smit_%d_%d/%s/' % (FEATURES_DIR, numcep, numdeltas,
                                           dataset)
        speakers = os.listdir(datasetpath)
        speakers.sort()
        for speaker in speakers:
            print('SPEAKER %s' % speaker)
            expfile.write('SPEAKER %s\n' % speaker)

            #Gambi
            if speaker[0] == 'f':
                start = 0
                end = 22
            else:
                start = 22
                end = 48

            speakerpath = '%s%s/' % (datasetpath, speaker)
            mfccsfilenames = os.listdir(speakerpath)
            mfccsfilenames.sort()

            corrects = 0
            wrongs = 0
            for mfccsfilename in mfccsfilenames:
                uttnum = int(mfccsfilename[0:2])
                mfccs = corpus.read_features(numcep, numdeltas, dataset,
                                             speaker, uttnum)
                likelihoods = np.array([mixtures.loglikelihood_gmm(gmm, mfccs)
                                        for gmm in gmmlist[start : end]])
                indexspeaker = np.argmax(likelihoods)
                classified = speakers[indexspeaker]

                if classified == speaker:
                    corrects = corrects + 1
                else:
                    wrongs = wrongs + 1

            print('CORRECTS %d\nWRONGS %d' % (corrects, wrongs))
            expfile.write('CORRECTS %d\nWRONGS %d\n' % (corrects, wrongs))