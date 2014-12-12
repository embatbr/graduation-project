"""This module defines the GMM and contains functions to training and evaluation.
"""


import numpy as np
import math
import random

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

def create_gmm(M, D):
    """Creates a GMM with M mixtures and fed by a D-dimensional feature vector.

    @param M: number of mixtures.
    @param D: size of features (used to the means and covariances).

    @returns: an untrained GMM.
    """
    weights = np.array([random.uniform(M/2, M) for _ in range(M)])
    weights = weights / np.sum(weights)

    gmm = list()
    for weight in weights:
        means = np.array([random.uniform(-M/2, M/2) for _ in range(D)])
        variances = np.array([random.uniform(1, D/2) for _ in range(D)])
        covmatrix = np.diag(variances)  #nodal-diagonal Covariance Matrix
        mixture = (weight, means, covmatrix)
        gmm.append(mixture)

    return gmm

def feed_gmm(gmm, features):
    """Feeds the GMM with the given features.

    @param gmm: the GMM used (a list of tuples (weight, means, covariances)).
    @param features: a Dx1 vector of features.

    @returns: the logarithm of the weighted sum of gaussians for gmm.
    """
    prob = 0
    for mixture in gmm:
        (weight, means, covariances) = mixture
        prob = prob + weight*gaussian(features, means, covariances)

    return prob

def eval_gmm(gmm, mfccs):
    """Feeds the GMM with a sequence of feature vectors.

    @param gmm: the GMM used (a list of tuples (weight, means, covariances)).
    @param features: a D x NUMFRAMES matrix of features.

    @returns: the average sum of logarithm of the weighted sum of gaussians for
    gmm for each feature vector.
    """
    numframes = len(mfccs.T)
    probs = np.array([feed_gmm(gmm, features) for features in mfccs.T])
    return (np.sum(np.log10(probs)) / numframes)


#TESTS
if __name__ == '__main__':
    import scipy.io.wavfile as wavf
    import os, os.path, shutil
    from useful import CORPORA_DIR, TEST_IMAGES_DIR, plotpoints, plotgaussian
    from useful import plotgmm, plotfigure
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
    numfeats = numcep*(numdeltas + 1)
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
        (mean, variance) = (means[n], covmatrix[n][n])
        filecounter = plotgaussian(mfccs[n], pdfs, mean, variance, 'MFCCs[%d] %s\nN(%f, %f)' %
                                   (n, voice, mean, variance), 'MFCCs[%d]' % n,
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

    Ms = [2**n for n in range(5, 12)]
    for M in Ms:
        print('Plotting GMM, M = %d' % M)
        gmm = create_gmm(M, numfeats)
        x = np.linspace(-(M + numfeats)/2, (M + numfeats)/2, 1000)
        for featnum in range(numfeats):
            filecounter = plotgmm(x, gmm, featnum, 'M = %d, GMM[%d]' % (M, featnum),
                                  'x', 'pdf', filename, filecounter)

    #Feeding GMM
    print('Feeding GMM')
    print(mfccs.shape)
    M = 32
    gmm = create_gmm(M, numfeats)
    probs = list()
    for features in mfccs.T:
        prob = feed_gmm(gmm, features)
        probs.append(prob)

    probs = np.array(probs)
    logprobs = np.log10(probs)
    numframes = len(mfccs.T)
    log_likelihood = np.sum(logprobs) / numframes
    print('log p(X|lambda) = %f' % log_likelihood)
    frameindices = np.linspace(0, numframes, numframes, False)
    filecounter = plotfigure(frameindices, logprobs, 'Log of probability per frame',
                             'frame', 'log', filename, filecounter)

    #Evaluating GMM
    print('Evaluating GMM')
    log_likelihood = eval_gmm(gmm, mfccs)
    print('log p(X|lambda) = %f' % log_likelihood)