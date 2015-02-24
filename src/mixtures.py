"""This module defines the GMM and contains functions for training and evaluation.
"""


import numpy as np
from math import pi as PI
from common import EPS


DEBUG = True


# K-means algorithm to generate the means

def partition(featsvec, numsets):
    """Divides the features in a given number of sets.

    @param featsvec: features extracted from speech signals.
    @param numsets: number of sets to divide the features.

    @returns: means and clusters.
    """
    numframes = len(featsvec)
    step = int(numframes / numsets)
    clusters = list()
    means = list()
    variances = list()

    start = end = 0
    while numsets > 0:
        start = end
        end = end + step
        if numsets == 1:
            end = numframes

        cluster = featsvec[start : end]
        clusters.append(cluster)
        mean = np.mean(cluster, axis=0)
        means.append(mean)
        variance = np.std(cluster, axis=0)
        variances.append(variance)

        numsets = numsets - 1

    means = np.array(means)
    variances = np.array(variances)
    return (clusters, means, variances)


class GMM(object):
    """Represents a GMM with number of mixtures M and a D-variate gaussian.
    """
    def __init__(self, nummixtures, featsvec):
        """Creates a GMM.

        @param nummixtures: number of mixtures (integer).
        @param featsvec: features used by the GMM.
        """
        self.nummixtures = nummixtures
        self.numfeats = len(featsvec[0])
        self.weights = np.tile(1 / self.nummixtures, self.nummixtures)
        (_, self.meansvec, self.variancesvec) = partition(featsvec, self.nummixtures)

    #99.8% of reduction in time to compute, compared with the version with loop
    def eval(self, feats, sumProbs=True):
        """Feeds the GMM with the given features. It performs a normal pdf for a
        number M of mixtures.

        @param feats: a Dx1 vector of features.
        @param sumProbs: tells if the weighted mixtures must be summed. Default
        is True.

        @returns: the weighted gaussians for gmm, summed or not.
        """
        D = self.numfeats

        #Constant
        determinant = np.prod(self.variancesvec, axis=1)
        cte_denom = ((2*PI)**D * determinant)**0.5

        #Exponent
        x_minus_mu = feats - self.meansvec
        power = x_minus_mu / self.variancesvec
        power = power * x_minus_mu
        power = np.sum(power, axis=1)
        power = -0.5*power

        #Probability
        prob = (np.exp(power) / cte_denom)
        ret = self.weights * prob
        ret = np.where(ret == 0, EPS, ret)

        if sumProbs:
            return np.sum(ret, axis=0)
        return ret

    def log_likelihood(self, featsvec):
        """Feeds the GMM with a sequence of feature vectors.

        @param featsvec: a NUMFRAMES x D matrix of features (features over time).

        @returns: the average sum of logarithm of the weighted sum of gaussians
        for gmm for each feature vector, aka, the log-likelihood.
        """
        probs = np.array([self.eval(feats) for feats in featsvec])
        probs = np.where(probs == 0, EPS, probs)
        logprobs = np.log(probs)
        logprobs = np.sum(logprobs, axis=0)

        numframes = len(featsvec) # numframes is the number T of x_t elements
        return (logprobs / numframes)

    def train(self, featsvec, threshold=1E-5):
        """Train the given GMM with the sequence of given feature vectors. Uses
        the EM algorithm.

        @param gmm: the GMM used (a list of tuples (weight, means, variances)).
        @param featsvec: a D x NUMFRAMES matrix of features.
        @param threshold: the difference between old and new probabilities must be
        lower than (or equal to) this parameter in %. Default 0.01 (1%).

        @returns: the average sum of logarithm of the weighted sum of gaussians
        for gmm for each feature vector.
        """
        #The new GMM; optimization to avoid create new arrays every iteration
        T = len(featsvec)
        posteriors = np.zeros((T, self.nummixtures))
        old_weights = np.zeros(self.weights.shape)
        old_meansvec = np.zeros(self.meansvec.shape)
        old_variancesvec = np.zeros(self.variancesvec.shape)

        #self.weights = np.where(self.weights == 0, EPS, self.weights)
        #self.meansvec = np.where(self.meansvec == 0, EPS, self.meansvec)
        #self.variancesvec = np.where(self.variancesvec == 0, EPS, self.variancesvec)

        if DEBUG: print('CALCULATING old_logprob')
        old_logprob = self.log_likelihood(featsvec)
        if DEBUG: print(old_logprob)

        if DEBUG: iteration = 1 #DEBUG
        while True:
            if DEBUG:
                print('iter = %d' % iteration)
                iteration += 1

            for t in range(T):
                feats = featsvec[t]
                gaussians = self.eval(feats, sumProbs=False) # one for each one of M mixtures
                weighted_gaussians = self.weights * gaussians
                likelihood_in_t = np.sum(weighted_gaussians, axis=0)
                #likelihood_in_t = EPS if likelihood_in_t == 0 else likelihood_in_t
                posteriors[t] = weighted_gaussians / likelihood_in_t
                posteriors[t] = np.where(posteriors[t] == 0, EPS, posteriors[t])

            #Summation from t=1 until t=T
            summed_posteriors_in_t = np.sum(posteriors, axis=0)

            for i in range(self.nummixtures):
                old_weights[i] = self.weights[i]
                #Updating i-th weight
                self.weights[i] = summed_posteriors_in_t[i] / T
                self.weights[i] = np.where(self.weights[i] == 0, EPS, self.weights[i])

                old_meansvec[i] = self.meansvec[i]
                #Updating i-th meansvec
                self.meansvec[i] = np.dot(posteriors[:, i], featsvec)
                self.meansvec[i] = self.meansvec[i] / summed_posteriors_in_t[i]
                self.meansvec[i] = np.where(self.meansvec[i] == 0, EPS, self.meansvec[i])

                old_variancesvec[i] = self.variancesvec[i]
                #Updating i-th variancesvec
                self.variancesvec[i] = np.dot(posteriors[:, i], featsvec**2)
                self.variancesvec[i] = self.variancesvec[i] / summed_posteriors_in_t[i]
                self.variancesvec[i] = self.variancesvec[i] - self.meansvec[i]**2
                self.variancesvec[i] = np.where(self.variancesvec[i] == 0, EPS, self.variancesvec[i])

            if DEBUG: print('CALCULATING new_logprob')
            new_logprob = self.log_likelihood(featsvec)
            reduction = (old_logprob - new_logprob) / old_logprob
            if DEBUG: print(new_logprob)
            if DEBUG: print('reduction = %e' % reduction)

            if (reduction >= 0) and (reduction <= threshold):
                if DEBUG: print('CORRETO')
                break
            # This 'if' should never be triggered
            #if new_logprob <= old_logprob: #log of probabilities are negative; |new| >= |old|
            #    self.weights[i] = old_weights[i]
            #    self.meansvec[i] = old_meansvec[i]
            #    self.variancesvec[i] = old_variancesvec[i]
            #    if DEBUG: print('ERRADO')
            #    break

            old_logprob = new_logprob


if __name__ == '__main__':
    import scipy.io.wavfile as wavf
    import os, os.path, shutil
    import time
    import pylab as pl
    import bases

    from common import CORPORA_DIR, FEATURES_DIR


    winlen = 0.02
    winstep = 0.01
    numcep = 13
    M = 8
    featsindices = np.linspace(1, 13, 13)

    featsvec = bases.read_mit_speaker_features(numcep, 0, 'enroll_1', 'f00')
    (clusters, means, variances) = partition(featsvec, M)
    print('featsvec.shape =', featsvec.shape)
    print('num sets: %d' % len(clusters))

    # partitionating training data and calculating means and variances
    pl.subplot(311)
    for i in range(len(clusters)):
        print('set %d: %s' % (i, clusters[i].shape))
        pl.plot(clusters[i][:, 0], clusters[i][:, 1], '.')
    pl.subplot(312)
    for i in range(len(means)):
        print('set %d: %s' % (i, means[i].shape))
        pl.plot(means[i][0], means[i][1], '.')
    print('means.shape =', means.shape)
    pl.subplot(313)
    pl.plot(featsindices, variances.T)
    pl.xticks(featsindices)
    pl.xlim(featsindices[0], featsindices[-1])
    print('variances.shape =', means.shape)

    M = 8
    print('Creating GMM (M = %d)...' % M)
    t = time.time()
    gmm = GMM(M, featsvec)
    t = time.time() - t
    print('GMM created in %f seconds' % t)
    featsvec = bases.read_mit_features(numcep, 0, 'enroll_1', 'f00', 2)
    t = time.time()
    log_likelihood = gmm.log_likelihood(featsvec)
    t = time.time() - t
    print('GMM log-likelihood = %f in %f seconds' % (log_likelihood, t))
    t = time.time()
    gmm.train(featsvec)
    t = time.time() - t
    print('GMM trained in %f seconds' % t)
    log_likelihood = gmm.log_likelihood(featsvec)
    print('log-likelihood =', log_likelihood)

    pl.show()