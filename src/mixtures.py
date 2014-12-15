"""This module defines the GMM and contains functions to training and evaluation.
"""


import numpy as np
import math
import random


PI = math.pi
twoPI = 2*PI


def gaussian(features, means=np.array([0]), variances=np.array([1])):
    """A D-variate gaussian pdf.

    @param features: Dx1 vector of features
    @param means: Dx1 vector of means (one for each feature).
    @param variances: Dx1 vector of variances (diagonal from covariance matrix).

    @returns: the value (float) of the gaussian pdf for the given parameters.
    """
    D = len(features)
    determinant = np.prod(variances)
    cte = (twoPI**D * determinant)**0.5
    cte = 1 / cte

    #(x - mu)' * inverse * (x - mu)
    A = features - means
    inverse = 1 / variances
    power = np.multiply(A, inverse)
    power = np.dot(power, A)
    power = -0.5*power

    return (cte * math.exp(power))

def create_gmm(M, D, mfccs):
    """Creates a GMM with M mixtures and fed by a D-dimensional feature vector.

    @param M: number of mixtures.
    @param D: size of features (used to the means and variances).

    @returns: an untrained GMM.
    """
    #Definining ranges for means and variances
    a = np.amin(mfccs) - np.std(mfccs)
    b = np.amax(mfccs) + np.std(mfccs)
    c = np.std(mfccs)**2 - 1
    d = c + 2

    weights = np.array([random.uniform(0.1, 0.9) for _ in range(M)])
    weights = weights / np.sum(weights)
    gmm = list()
    for weight in weights:
        means = np.array([random.uniform(a, b) for _ in range(D)])
        variances = np.array([random.uniform(c, d) for _ in range(D)])
        gmm.append((weight, means, variances))

    return gmm

def eval_gmm(gmm, features):
    """Feeds the GMM with the given features.

    @param gmm: the GMM used (a list of tuples (weight, means, variances)).
    @param features: a Dx1 vector of features.

    @returns: the logarithm of the weighted sum of gaussians for gmm.
    """
    prob = 0
    for mixture in gmm:
        (weight, means, variances) = mixture
        prob = prob + weight*gaussian(features, means, variances)

    return prob

def loglikelihood_gmm(gmm, mfccs):
    """Feeds the GMM with a sequence of feature vectors.

    @param gmm: the GMM used (a list of tuples (weight, means, variances)).
    @param mfccs: a D x NUMFRAMES matrix of features.

    @returns: the average sum of logarithm of the weighted sum of gaussians for
    gmm for each feature vector, aka, the log-likelihood.
    """
    numframes = len(mfccs.T)
    #probs = np.array([eval_gmm(gmm, features) for features in mfccs.T])
    logprobs = 0
    i = 0
    for features in mfccs.T:
        if (i % 1000) == 0:
            print('log-likelihood', i)
        i += 1
        prob = eval_gmm(gmm, features)
        logprobs = logprobs + math.log10(prob)
    return (logprobs / numframes)
    #return (np.sum(np.log10(probs)) / numframes)

def prob_posterior_i(gmm_i, features, gmm):
    """Calculates the a posteriori probability for a tuple (weight_i, means_i, variances_i)
    for the given GMM and a D x 1 vector of features.

    @param gmm_i: i is the index from 0 to (M - 1).
    @param features: a D x 1 feature vector.
    @param gmm: the current GMM.

    @returns: the a posteriori probability for the tuple (weight_i, means_i, variances_i).
    """
    (weight_i, means_i, variances_i) = gmm_i
    prior = gaussian(features, means_i, variances_i)
    evidence = eval_gmm(gmm, features)
    return ((weight_i*prior) / evidence)

def prob_posterior_array(gmm_i, mfccs, gmm):
    """Calculates the a posteriori probability for all features throught time.

    @param gmm_i: i is the index from 0 to (M - 1).
    @param mfccs: a D x NUMFRAMES feature matrix.
    @param gmm: the current GMM.

    @returns: the array of a posteriori probability for 'mfccs'.
    """
    probs = list()
    i = 0
    for features in mfccs.T:
        if (i % 1000) == 0:
            print('posteriori', i)
        i+=1
        prob = prob_posterior_i(gmm_i, features, gmm)
        probs.append(prob)

    return np.array(probs)

def train_gmm(gmm, mfccs, threshold=0.01):
    """Train the given GMM with the sequence of given feature vectors.

    @param gmm: the GMM used (a list of tuples (weight, means, variances)).
    @param mfccs: a D x NUMFRAMES matrix of features.
    @param threshold: the difference between old and new probabilities must be
    lower than (or equal to) this parameter.

    @returns: the average sum of logarithm of the weighted sum of gaussians for
    gmm for each feature vector.
    """
    T = len(mfccs.T)
    old_gmm = gmm
    iteration = 1

    while True:
        print('[%d]\nCREATING new_gmm' % iteration)
        iteration += 1
        m = 1

        new_gmm = list()
        for old_gmm_i in old_gmm:   #old_gmm_i == (weight_i, means_i, covmatrix_i)
            print('mixture #%d' % m)
            m += 1
            #Expectation
            posteriors = prob_posterior_array(old_gmm_i, mfccs, old_gmm)
            summed = np.sum(posteriors)

            new_weight_i = summed / T

            new_means_i = np.dot(posteriors, mfccs.T)
            new_means_i = new_means_i / summed

            new_variances_i = np.dot(posteriors, (mfccs.T)**2)
            new_variances_i = (new_variances_i / summed) - new_means_i**2
            #a linha acima faz new_variances_i ficar com valores ZERO, ferrando
            #o determinante na função "gaussian()". Culpa do "- new_means_i**2"
            if 0 in new_variances_i:
                print(m-1, 'GAMBI')
                return

            #Maximization
            new_gmm_i = (new_weight_i, new_means_i, new_variances_i)
            new_gmm.append(new_gmm_i)

        print('CALCULATING oldprob')
        oldprob = loglikelihood_gmm(old_gmm, mfccs)
        print('%f\nCALCULATING newprob' % oldprob)
        newprob = loglikelihood_gmm(new_gmm, mfccs)
        reduction = (oldprob - newprob) / oldprob
        print('%f\nnewprob > oldprob ? %s\nreduction = %f' % (newprob, newprob > oldprob,
                                                              reduction))
        if reduction <= threshold:
            print('RETURNING new_gmm')
            return new_gmm

        old_gmm = new_gmm


#TESTS
if __name__ == '__main__':
    import scipy.io.wavfile as wavf
    import os, os.path, shutil
    from useful import CORPORA_DIR, TESTS_DIR, plotpoints, plotgaussian
    from useful import plotgmm, plotfigure
    import math
    import corpus


    if not os.path.exists(TESTS_DIR):
        os.mkdir(TESTS_DIR)

    IMAGES_MIXTURES_DIR = '%smixtures/' % TESTS_DIR

    if os.path.exists(IMAGES_MIXTURES_DIR):
            shutil.rmtree(IMAGES_MIXTURES_DIR)
    os.mkdir(IMAGES_MIXTURES_DIR)

    filecounter = 0
    filename = '%sfigure' % IMAGES_MIXTURES_DIR

    numcep = 13
    numdeltas = 0
    numfeats = numcep*(numdeltas + 1)
    speaker = 'f08'
    voice = ('enroll_1', speaker)

    #Reading MFCCs from features base
    mfccs = corpus.read_speaker_features(numcep, numdeltas, speaker)
    means = np.array([np.mean(feat) for feat in mfccs])
    variances = np.array([np.std(feat)**2 for feat in mfccs])
    print(mfccs.shape, len(means), len(variances))

    a = np.amin(mfccs) - 3*np.std(mfccs)
    b = np.amax(mfccs) + 3*np.std(mfccs)
    x = np.linspace(a, b, 1000)

    print('Gaussians')
    numframes = len(mfccs.T)
    print('#frames = %d' % numframes)
    pdfs = np.array([gaussian(features, means, variances) for features in mfccs.T])
    for n in range(numcep):
        (mean, variance) = (means[n], variances[n])
        filecounter = plotgaussian(mfccs[n], pdfs, mean, variance, 'MFCCs[%d] %s\nN(%f, %f)' %
                                   (n, voice, mean, variance), 'MFCCs[%d]' % n,
                                   'gaussian', filename, filecounter)

#    #Multivariate plotting
#    print('Multivariate plotting')
#    for i in range(numcep):
#        (mu_i, sigma2_i) = (means[i], variances[i])
#        for j in range(i + 1, numcep):
#            (mu_j, sigma2_j) = (means[j], variances[j])
#            filecounter = plotpoints(mfccs[i], mfccs[j],
#                                     'MFCCs[%d] x MFCCs[%d] %s\nN([%f, %f], [%f, %f])' %
#                                     (i, j, voice, mu_i, mu_j, sigma2_i, sigma2_j),
#                                     'mfccs[%d]' % i, 'mfccs[%d]' % j, filename,
#                                     filecounter, 'green')
#
#    Ms = [2**n for n in range(3, 9)]
#    print('Plotting GMMs with %d <= M <= %d' % (Ms[0], Ms[-1]))
#    for M in Ms:
#        print('M = %d' % M)
#        gmm = create_gmm(M, numfeats, mfccs)
#        for featnum in range(numfeats):
#            filecounter = plotgmm(x, gmm, featnum, 'M = %d, GMM[%d]' % (M, featnum),
#                                  'x', 'pdf', filename, filecounter)
#
#    #Evaluating GMM
#    print('Evaluating GMM')
#    print(mfccs.shape)
#    M = 32
#    gmm = create_gmm(M, numfeats, mfccs)
#    probs = list()
#    for features in mfccs.T:
#        prob = eval_gmm(gmm, features)
#        probs.append(prob)
#
#    probs = np.array(probs)
#    logprobs = np.log10(probs)
#    log_likelihood = np.sum(logprobs) / numframes
#    print('log p(X|lambda) = %f' % log_likelihood)
#    frameindices = np.linspace(0, numframes, numframes, False)
#    filecounter = plotfigure(frameindices, logprobs, 'Log of probability per frame',
#                             'frame', 'log', filename, filecounter)
#
#    #log-likelihood of GMM
#    print('log-likelihood of GMM')
#    log_likelihood = loglikelihood_gmm(gmm, mfccs)
#    print('log p(X|lambda) = %f' % log_likelihood)
#
#    #Training GMM
#    print('Section: Training')
#    speaker = 'f08'
#    mfccs = corpus.read_features(numcep, numdeltas, 'enroll_1', speaker, 54)
#    print(mfccs.shape)
#    a = np.amin(mfccs) - 3*np.std(mfccs)
#    b = np.amax(mfccs) + 3*np.std(mfccs)
#    x = np.linspace(a, b, 1000)
#
#    M = 32
#    print('creating GMM (M = %d)...' % M)
#    gmm = create_gmm(M, numfeats, mfccs)
#    print('training GMM...')
#    new_gmm = train_gmm(gmm, mfccs)
#    print('GMM trained!')
#    for featnum in range(numfeats):
#        filecounter = plotgmm(x, gmm, featnum, 'M = %d, GMM[%d], Untrained\n%s' %
#                              (M, featnum, speaker), 'x', 'pdf', filename, filecounter)
#        filecounter = plotgmm(x, new_gmm, featnum, 'M = %d, GMM[%d], Trained\n%s' %
#                              (M, featnum, speaker), 'x', 'pdf', filename, filecounter)