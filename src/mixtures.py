"""This module defines the GMM and contains functions for training and evaluation.
"""


import numpy as np
import random
from math import pi as PI
from common import FLOAT_MAX, ZERO, MIN_VARIANCE


EM_THRESHOLD = 1E-3


class EmptyClusterError(Exception):
    """Error triggered when a cluster in kmeans is empty
    """
    def __init__(self):
        self.msg = 'Empty cluster generated during k-means'

    def __str__(self):
        return self.msg


def kmeans(featsvec, M):
    """Clusters a vector of features until total separation.

    @param featsvec: the vector of features.
    @param M: the number of clusters.
    """
    indices = list(range(len(featsvec)))
    chosen = random.sample(indices, M)
    old_means = featsvec[chosen, :]
    max_diff = FLOAT_MAX

    iteration = 0
    while max_diff != 0.0:
        clusters = [list() for _ in range(M)]
        for feats in featsvec:
            distance = np.linalg.norm(feats - old_means, axis=1)**2
            index = np.argmin(distance)
            clusters[index].append(feats)

        means = list()
        for cluster in clusters:
            if len(cluster) == 0:
                raise EmptyClusterError()
            mean = np.mean(cluster, axis=0)
            means.append(mean)
        means = np.array(means)

        max_diff = np.amax(np.fabs(old_means - means))
        old_means = means
        iteration += 1

    print('Number of iterations: %d' % iteration)

    #turn every cluster into a numpy array
    clusters = [np.array(cluster) for cluster in clusters]

    T = len(featsvec)
    weights = list()
    variances = list()
    for cluster in clusters:
        weights.append(len(cluster) / T)
        variance = np.std(cluster, axis=0)**2
        variances.append(variance)
    weights = np.array(weights)
    variances = np.array(variances)
    variances = np.where(variances < MIN_VARIANCE, MIN_VARIANCE, variances)

    return (weights, means, variances)


class GMM(object):
    """Represents a GMM with number of mixtures M and a D-variate gaussian.
    """
    def __init__(self, name, M, D):
        """Creates a GMM.

        @param name: name of the GMM.
        @param M: number of mixtures (integer).
        """
        self.name = name
        self.M = M
        self.D = D

        self.weights = np.tile(1 / M, M)
        self.meansvec = np.zeros((M, D))
        self.variancesvec = np.ones((M, D))

    def merge(self, gmm, name=None):
        """
        Merges the GMM object with another GMM object.

        @param gmm: the other GMM object to be merged.
        @param name: the new name of the merged object. Default, None.
        """
        if not name is None:
            self.name = name
        self.M = self.M + gmm.M
        self.weights = np.hstack((self.weights, gmm.weights))
        self.weights = self.weights / np.sum(self.weights, axis=0)
        self.meansvec = np.vstack((self.meansvec, gmm.meansvec))
        self.variancesvec = np.vstack((self.variancesvec, gmm.variancesvec))

    def clone(self, name=None):
        clonename = self.name if name is None else name
        clone = GMM(clonename, self.M, self.D)

        clone.weights = np.copy(self.weights)
        clone.meansvec = np.copy(self.meansvec)
        clone.variancesvec = np.copy(self.variancesvec)

        return clone

    def posterior(self, feats):
        """Feeds the GMM with the given features. It performs a normal pdf for a
        number M of mixtures (one for each m from M).

        @param feats: a Dx1 vector of features.

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
        w_probs = np.where(w_probs == 0, ZERO, w_probs)
        likelihood = np.sum(w_probs, axis=0)

        return (likelihood, w_probs) #sum in mixtures axis

    def log_likelihood(self, featsvec):
        """Feeds the GMM with a sequence of feature vectors.

        @param featsvec: a NUMFRAMES x D matrix of features (features over time).
        @param normalize: determines if the log-likelihood is divided by T. By
        default, True.

        @returns: the average (if normalize is True) sum of logarithm of the weighted
        sum of gaussians for the GMM for each feature vector, aka, the log-likelihood
        in base 10.
        """
        probs = np.array([self.posterior(feats)[0] for feats in featsvec])
        logprobs = np.log10(probs)
        return np.mean(logprobs, axis=0) # sums logprobs and divides by number of samples (T)

    def train(self, featsvec, threshold=EM_THRESHOLD, use_kmeans=True, use_EM=True):
        """Trains the given GMM with the sequence of given feature vectors. Uses
        the EM algorithm.

        @param featsvec: a NUMFRAMES x D matrix of features.
        @param threshold: the difference between old and new log-likelihoods must
        be lower than (or equal to) this parameter. Default EM_THRESHOLD.
        @param use_kmeans: determines if the k-means algorithm is used. Default, True.
        @param use_EM: determines if the EM algorithm is used. Default, True.
        """
        if use_kmeans:
            print('kmeans')
            (self.weights, self.meansvec, self.variancesvec) = kmeans(featsvec, self.M)

        if use_EM:
            print('EM')
            T = len(featsvec)
            posteriors = np.zeros((T, self.M))
            old_log_like = self.log_likelihood(featsvec)
            print('log_like = %f' % old_log_like)

            iteration = 0
            diff = FLOAT_MAX
            while diff > threshold:
                # E-Step
                for t in range(T):
                    (posterior_in_t, w_gaussians) = self.posterior(featsvec[t]) # one for each one of M mixtures
                    posteriors[t] = w_gaussians / posterior_in_t

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

                self.weights = self.weights / np.sum(self.weights, axis=0)


                new_log_like = self.log_likelihood(featsvec)
                diff = new_log_like - old_log_like
                old_log_like = new_log_like
                iteration += 1

            print('After %d iterations\nlog_like = %f' % (iteration, new_log_like))

    def adapt_gmm(self, featsvec, adaptations='wmv', top_C=None, relevance_factor=16):
        """
        Adapts an UBM to a GMM for a specific speaker, given the speaker's features
        vector.

        @param featsvec: a NUMFRAMES x D matrix of features.
        @param relevance_factor: the relevance factor for adaptations of weights,
        means and variances. Default, 16.
        @param adaptations: determines which parameters will be adapted. To adapt
        weights, use 'w', means, 'm', and variances, 'v'. Default 'wmv'.
        """
        T = len(featsvec)
        posteriors = np.zeros((T, self.M))

        # E-Step
        for t in range(T):
            (posterior_in_t, w_gaussians) = self.posterior(featsvec[t]) # one for each one of M mixtures
            posteriors[t] = w_gaussians / posterior_in_t

        #Summation of posteriors from 1 to T.
        #Is the 'n_i' from the adaptation algorithm
        sum_posteriors = np.sum(posteriors, axis=0)

        for i in range(self.M):
            alpha_i = sum_posteriors[i] / (sum_posteriors[i] + relevance_factor)
            oldmeans = self.meansvec[i]

            if 'w' in adaptations:
                self.weights[i] = (alpha_i*sum_posteriors[i]) / T + (1 - alpha_i)*self.weights[i]

            if 'm' in adaptations:
                meansvec_map = np.dot(posteriors[:, i], featsvec)
                meansvec_map = meansvec_map / sum_posteriors[i]
                self.meansvec[i] = alpha_i*meansvec_map + (1 - alpha_i)*self.meansvec[i]

            if 'v' in adaptations:
                variancesvec_map = np.dot(posteriors[:, i], featsvec**2)
                variancesvec_map = variancesvec_map / sum_posteriors[i]
                self.variancesvec[i] = alpha_i*variancesvec_map + (1 - alpha_i)*\
                                       (self.variancesvec[i] + oldmeans**2) -\
                                       self.meansvec[i]**2
                self.variancesvec[i] = np.where(self.variancesvec[i] < MIN_VARIANCE,
                                                MIN_VARIANCE, self.variancesvec[i])

        if 'w' in adaptations:
            self.weights = self.weights / np.sum(self.weights, axis=0)

        if not top_C is None:
            probs = np.array([self.posterior(feats)[1] for feats in featsvec])
            logprobs = np.log10(probs)
            logprobs_mean = np.mean(logprobs, axis=0)
            sorted_indices = np.argsort(logprobs_mean)[-top_C : ]

            self.weights = self.weights[sorted_indices]
            self.weights = self.weights / np.sum(self.weights, axis=0)
            self.meansvec = self.meansvec[sorted_indices]
            self.variancesvec = self.variancesvec[sorted_indices]