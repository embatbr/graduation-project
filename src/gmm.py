"""This module defines the GMM and contains functions to training and evaluation.
"""


import numpy as np
import math

import corpus


def gaussian(features, means=np.array([0]), covariances=np.array([[1]])):
    """A multivariate (D features) gaussian pdf.

    @param features: Dx1 vector of features
    @param means: Dx1 vector of means (one for each feature).
    @param covariances: DxD matrix of covariances (usually a diagonal matrix).

    @returns: the value (float) of the gaussian pdf for the given parameters.
    """
    D = len(features)
    determinant = np.linalg.det(covariances)
    cte = 1 / ((2*math.pi)**(D/2) * determinant**0.5)
    inverse = np.linalg.inv(covariances)

    #(x - mu)' * inverse * (x - mu)
    #due to the structure of array, the A.T is made at the end
    A = np.array([(features - means).tolist()])
    power = np.dot(A, inverse)
    power = np.dot(power, A.T)
    power = -0.5*power[0, 0]

    return (cte * math.exp(power))


#TESTS
if __name__ == '__main__':
    import scipy.io.wavfile as wavf
    import os, os.path, shutil
    from useful import CORPORA_DIR, TEST_IMAGES_DIR, plotpoints
    import math
    import corpus


    if not os.path.exists(TEST_IMAGES_DIR):
        os.mkdir(TEST_IMAGES_DIR)

    IMAGES_GMM_DIR = '%sgmm/' % TEST_IMAGES_DIR

    if os.path.exists(IMAGES_GMM_DIR):
            shutil.rmtree(IMAGES_GMM_DIR)
    os.mkdir(IMAGES_GMM_DIR)

    filecounter = 0
    filename = '%sfigure' % IMAGES_GMM_DIR

    numcep = 13
    numdeltas = 0
    voice = ('enroll_1', 'f08', 54)
    (dataset, speaker, speech) = voice

    #Reading MFCCs from features base
    mfccs = corpus.read_speaker_features(numcep, numdeltas, speaker)
    means = np.array([np.mean(feat) for feat in mfccs])
    stds = np.array([np.std(feat)**2 for feat in mfccs])
    covmatrix = np.diag(stds)
    print(mfccs.shape, len(means), covmatrix.shape)

    numframes = len(mfccs.T)
    print('#frames = %d' % numframes)
    pdfs = np.array([gaussian(features, means, covmatrix) for features in mfccs.T])
    for n in range(numcep):
        (mu, sigma) = (means[n], covmatrix[n][n])
        filecounter = plotpoints(mfccs[n], pdfs, 'MFCCs[%d] %s\nN(%f, %f)' %
                                 (n, voice, mu, sigma), 'MFCCs[%d]' % n,
                                 'gaussian', filename, filecounter)

    #Multivariate plotting
    print('Multivariate plotting')
    for i in range(numcep):
        (mu_i, sigma_i) = (means[i], covmatrix[i][i])
        for j in range(i + 1, numcep):
            (mu_j, sigma_j) = (means[j], covmatrix[j][j])
            filecounter = plotpoints(mfccs[i], mfccs[j],
                                     'MFCCs[%d] x MFCCs[%d] %s\nN([%f, %f], [%f, %f])' %
                                     (i, j, voice, mu_i, mu_j, sigma_i, sigma_j),
                                     'mfccs[%d]' % i, 'mfccs[%d]' % j, filename,
                                     filecounter, 'green')