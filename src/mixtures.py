"""This module defines the GMM and contains functions to training and evaluation.
"""


import numpy as np
import math
import random


PI = math.pi


class GMM(object):
    """Represents a GMM with number of mixtures M and a D-variate gaussian.
    """

    def __init__(self, M, D):
        """Creates a GMM.

        @param M: number of mixtures (integer).
        @param D: number of features (integer).
        """
        self.num_mixtures = M
        self.num_features = D

        std = (0.5*(M/D)**0.5, (M/D)**0.5)

        self.weights = np.array([1/M for _ in range(M)])
        self.means = list()
        self.variances = list()
        for m in range(M):
            means = [random.uniform(-D, D) for _ in range(D)]
            self.means.append(means)
            variances = [random.uniform(std[0], std[1]) for _ in range(D)]
            self.variances.append(variances)

        self.means = np.array(self.means)
        self.variances = np.array(self.variances)

    def get_mixture(self, m):
        """@returns: the m-th mixture of the GMM (with 0 <= m < M).
        """
        return (self.weights[m], self.means[m], self.variances[m])

    #99.8% of reduction in time to compute, compared with the previous function
    def eval(self, features):
        """Feeds the GMM with the given features.

        @param features: a Dx1 vector of features.

        @returns: the weighted sum of gaussians for gmm.
        """
        D = self.num_features

        #Constant
        determinant = np.prod(self.variances, axis=1)
        cte = (2*PI**D * determinant)**0.5

        #Exponent
        A = features - self.means
        power = np.divide(A, self.variances)
        power = np.multiply(power, A)
        power = np.sum(power, axis=1)
        power = -0.5*power

        #Probability
        prob = (np.exp(power) / cte)
        prob = np.dot(self.weights, prob)

        return prob

    def log_likelihood(self, mfccs):
        """Feeds the GMM with a sequence of feature vectors.

        @param mfccs: a NUMFRAMES x D matrix of features (features over time).

        @returns: the average sum of logarithm of the weighted sum of gaussians
        for gmm for each feature vector, aka, the log-likelihood.
        """
        numframes = len(mfccs)
        logprobs = 0

        #TODO transformar o loop em uma operação totalmente vetorial
        for features in mfccs:
            prob = self.eval(features)
            logprobs = logprobs + math.log10(prob)

        return (logprobs / numframes)

    def train_gmm(self, mfccs, threshold=0.01):
        """Train the given GMM with the sequence of given feature vectors.

        @param gmm: the GMM used (a list of tuples (weight, means, variances)).
        @param mfccs: a D x NUMFRAMES matrix of features.
        @param threshold: the difference between old and new probabilities must be
        lower than (or equal to) this parameter.

        @returns: the average sum of logarithm of the weighted sum of gaussians for
        gmm for each feature vector.
        """
        pass


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
    evidence = eval_gmm(features, gmm)
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
    import time


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
    mfccs = corpus.read_features(numcep, numdeltas, 'enroll_1', speaker, 54, False)
    #mfccs = corpus.read_speaker_features(numcep, numdeltas, speaker, False)
    #mfccs = corpus.read_background_features(numcep, numdeltas, 'm', False)
    print('mfccs:', mfccs.shape)
    x = np.linspace(-numfeats, numfeats, 1000)

    M = 32
    print('Creating GMM (M = %d)...' % M)
    gmm = GMM(M, numfeats)
    print('GMM created!')
    for featnum in range(numfeats):
        filecounter = plotgmm(x, gmm, featnum, 'M = %d, GMM[%d]' % (M, featnum),
                              'x', 'pdf', filename, filecounter)

    #Evaluating GMM
    print('Evaluating GMM...')
    t = time.time()
    probs = list()
    for features in mfccs:
        prob = gmm.eval(features)
        probs.append(prob)
    t = time.time() - t
    print('GMM evaluated in', t, 'seconds')
    probs = np.array(probs)
    probs = np.log10(probs)

    numframes = len(probs)
    print('numframes = %d' % numframes)
    frameindices = np.linspace(0, numframes, numframes, False)
    filecounter = plotfigure(frameindices, probs, 'Log of probability per frame',
                             'frame', 'log', filename, filecounter)

    #log-likelihood of GMM
    print('log-likelihood of GMM')
    t = time.time()
    log_likelihood = gmm.log_likelihood(mfccs)
    t = time.time() - t
    print('Log likelihood calculated in', t, 'seconds')
    print('log p(X|lambda) = %f' % log_likelihood)

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