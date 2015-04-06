"""This module defines the GMM and contains functions for training and evaluation.
"""


import numpy as np
import random
from math import pi as PI
from common import ZERO, MIN_VARIANCE


KMEANS_MIN_AMAX_REDUCTION = 1E-2
EM_THRESHOLD = 1E-3


def partition(featsvec, M):
    np.random.shuffle(featsvec)
    T = len(featsvec)
    step = int(T / M)

    clusters = list()
    means = list()

    for i in range(M):
        if i == (M - 1):
            cluster = featsvec[i*step : ]
        else:
            cluster = featsvec[i*step : (i + 1)*step]
        clusters.append(cluster)

        mean = np.mean(cluster, axis=0)
        means.append(mean)

    means = np.array(means)
    return means


def kmeans(featsvec, M):
    old_means = partition(featsvec, M)
    old_amax = None

    while True:
        clusters = [list() for _ in range(M)]
        for feats in featsvec:
            distance = np.linalg.norm(feats - old_means, axis=1)**2
            index = np.argmin(distance)
            clusters[index].append(feats)

        means = list()
        for cluster in clusters:
            mean = np.mean(cluster, axis=0)
            means.append(mean)
        means = np.array(means)

        diff = np.fabs(old_means - means)
        amax = np.amax(diff)
        if not old_amax is None:
            reduction = np.fabs(amax / old_amax)
            if reduction <= KMEANS_MIN_AMAX_REDUCTION:
                break

        old_means = means
        old_amax = amax

    T = len(featsvec)
    weights = list()
    variances = list()
    for cluster in clusters:
        weights.append(len(cluster) / T)
        variance = np.std(cluster, axis=0)**2
        variances.append(variance)
    weights = np.array(weights)
    variances = np.array(variances)

    return (weights, means, variances)


class GMM(object):
    """Represents a GMM with number of mixtures M and a D-variate gaussian.
    """
    def __init__(self, name, M, featsvec):
        """Creates a GMM.

        @param name: name of the GMM.
        @param M: number of mixtures (integer).
        @param featsvec: features used by the GMM.
        """
        #TODO usar k-means para evitar ficar preso em máximo local?
        self.name = name
        self.M = M
        self.D = featsvec.shape[1]
        self.weights = self.meansvec = self.variancesvec = None

    def __repr__(self):
        """String representation of a GMM object.

        @returns: a string containing the atributes.
        """
        ret = 'M = %d\nD = %d' % (self.M, self.D)
        ret = '%s\nweights:\n' % ret
        for m in range(self.M):
            ret = '%s%d: %f\n' % (ret, m, self.weights[m])
        ret = '%smeans:\n' % ret
        for m in range(self.M):
            ret = '%s%d: %s\n' % (ret, m, self.meansvec[m])
        ret = '%svariances:\n' % ret
        for m in range(self.M):
            ret = '%s%d: %s\n' % (ret, m, self.variancesvec[m])

        return ret

    def eval(self, feats):
        """Feeds the GMM with the given features. It performs a normal pdf for a
        number M of mixtures (one for each m from M).

        @param feats: a Dx1 vector of features.
        @param sumProbs: tells if the weighted mixtures must be summed. Default
        is True.

        @returns: a tuple: the weighted gaussians for gmm, summed and as array.
        """
        #Denominator of constant
        determinant = np.prod(self.variancesvec, axis=1)
        cte = ((2*PI)**(self.D/2)) * (determinant**(1/2))

        #Exponent
        feats_minus_meansvec = feats - self.meansvec
        exponent = feats_minus_meansvec / self.variancesvec
        exponent = exponent * feats_minus_meansvec
        exponent = np.sum(exponent, axis=1)
        exponent = -(1/2)*exponent

        #Probabilities
        probs = np.exp(exponent) / cte
        w_probs = (self.weights * probs)
        #if np.any(w_probs == 0):
        #    print(len(probs[w_probs == 0]))
        w_probs = np.where(w_probs == 0, ZERO, w_probs)
        likelihood = np.sum(w_probs, axis=0)

        return (likelihood, w_probs) #sum in mixtures axis

    def log_likelihood(self, featsvec):
        """Feeds the GMM with a sequence of feature vectors.

        @param featsvec: a NUMFRAMES x D matrix of features (features over time).

        @returns: the average sum of logarithm of the weighted sum of gaussians
        for gmm for each feature vector, aka, the log-likelihood.
        """
        probs = np.array([self.eval(feats)[0] for feats in featsvec])
        logprobs = np.log10(probs)
        return np.mean(logprobs, axis=0) # sum logprobs and divide by number of samples (T)

    def train(self, featsvec, threshold=EM_THRESHOLD):
        """Trains the given GMM with the sequence of given feature vectors. Uses
        the EM algorithm.

        @param featsvec: a NUMFRAMES x D matrix of features.
        @param threshold: the difference between old and new probabilities must be
        lower than (or equal to) this parameter in %. Default 0.01 (1%).
        """
        print('kmeans')
        (self.weights, self.meansvec, self.variancesvec) = kmeans(featsvec, self.M)
        T = len(featsvec)
        posteriors = np.zeros((T, self.M))
        old_log_like = self.log_likelihood(featsvec)

        print('EM')
        while True:
            # E-Step
            for t in range(T):
                (likelihood_in_t, w_gaussians) = self.eval(featsvec[t]) # one for each one of M mixtures
                posteriors[t] = w_gaussians / likelihood_in_t

            #Summation of posteriors from 1 to T
            sum_posteriors = np.sum(posteriors, axis=0)

            # M-Step
            for i in range(self.M):
                #Updating i-th weight
                self.weights[i] = sum_posteriors[i] / T

                #Updating i-th meansvec
                self.meansvec[i] = np.dot(posteriors[:, i], featsvec)
                self.meansvec[i] = self.meansvec[i] / sum_posteriors[i]

                #Updating i-th variancesvec
                self.variancesvec[i] = np.dot(posteriors[:, i], featsvec**2)
                self.variancesvec[i] = self.variancesvec[i] / sum_posteriors[i]
                self.variancesvec[i] = self.variancesvec[i] - self.meansvec[i]**2
                self.variancesvec[i] = np.where(self.variancesvec[i] < MIN_VARIANCE,
                                                MIN_VARIANCE, self.variancesvec[i])

            new_log_like = self.log_likelihood(featsvec)
            diff = new_log_like - old_log_like
            if diff <= threshold:
                break

            old_log_like = new_log_like