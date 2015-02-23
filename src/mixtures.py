"""This module defines the GMM and contains functions for training and evaluation.
"""


import numpy as np
from math import pi as PI
from common import ZERO


DEBUG = False


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

def kmeans(featsvec, means, numsets):
    """Divides the elements into 'numsets' clusters.

    @param featsvec: features extracted from speech signals.
    @param means: the mean for each cluster.
    @param numsets: number of sets to divide the features.

    @returns: a 'numsets' number of sets clustered.
    """
    while(True):
        clusters = [list() for _ in range(numsets)]
        for feat in featsvec:
            distances = np.linalg.norm(feat - means, axis=1)
            index = np.argmin(distances)
            clusters[index].append(feat)

        clusters = [np.array(cluster) for cluster in clusters]

        oldmeans = means
        means = list()
        for cluster in clusters:
            mean = np.mean(cluster, axis=0)
            means.append(mean)

        means = np.array(means)
        distances = np.linalg.norm(means - oldmeans, axis=1)

        #if len(distances[np.where(distances > 0.01)]) == 0:
        if len(distances[distances > 0.01]) == 0:
            variances = [np.std(cluster, axis=0) for cluster in clusters]
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

        (_, means, variances) = partition(featsvec, self.nummixtures)
        #(_, means, variances) = kmeans(featsvec, means, self.nummixtures) # too slow

        self.weights = np.tile(1 / self.nummixtures, self.nummixtures)
        self.meansvec = means
        self.variancesvec = variances

    #99.8% of reduction in time to compute, compared with the version with loop
    def eval(self, feats, sumProbs=True):
        """Feeds the GMM with the given features.

        @param feats: a Dx1 vector of features.
        @param func: the function applied in the return. By default is 'numpy.dot',
        to calculate the evaluation of the GMM fed by a features vector.

        @returns: the weighted sum of gaussians for gmm.
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
        ret = np.where(ret < ZERO, ZERO, ret)

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
        probs = np.where(probs < ZERO, ZERO, probs)
        logprobs = np.log(probs)
        logprobs = np.sum(logprobs, axis=0)

        numframes = len(featsvec) # numframes is the number T of x_t elements
        return (logprobs / numframes)

    def train(self, featsvec, threshold=0.01):
        """Train the given GMM with the sequence of given feature vectors.

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

        iteration = 1 #DEBUG
        run = True
        while run:
            if DEBUG:
                print('iter = %d' % iteration)
                iteration += 1

            if DEBUG: print('CALCULATING oldprob')
            oldprob = self.log_likelihood(featsvec)

            for t in range(T):
                feats = featsvec[t]
                post = self.eval(feats, sumProbs=False)
                evaluated = np.sum(post, axis=0)
                evaluated = ZERO if evaluated < ZERO else evaluated
                post = (self.weights * post) / evaluated
                posteriors[t] = np.array(post)
                posteriors[t] = np.where(posteriors[t] < ZERO, ZERO, posteriors[t])

            #Summation from t=1 until t=T
            summed_posteriors = np.sum(posteriors, axis=0)

            for i in range(self.nummixtures):
                #Updating i-th weight
                self.weights[i] = summed_posteriors[i] / T
                self.weights[i] = np.where(self.weights[i] < ZERO, ZERO, self.weights[i])

                #Updating i-th meansvec
                self.meansvec[i] = np.dot(posteriors[:, i], featsvec)
                self.meansvec[i] = self.meansvec[i] / summed_posteriors[i]
                self.meansvec[i] = np.where(self.meansvec[i] < ZERO, ZERO, self.meansvec[i])

                #Updating i-th variancesvec
                self.variancesvec[i] = np.dot(posteriors[:, i], featsvec**2)
                self.variancesvec[i] = self.variancesvec[i] / summed_posteriors[i]
                self.variancesvec[i] = self.variancesvec[i] - self.meansvec[i]**2
                self.variancesvec[i] = np.where(self.variancesvec[i] < ZERO, ZERO, self.variancesvec[i])

            if DEBUG: print('%f\nCALCULATING newprob' % oldprob)
            newprob = self.log_likelihood(featsvec)
            reduction = (oldprob - newprob) / oldprob
            if DEBUG: print('%f\nMONOTONIC ? %s\nreduction = %f' % (newprob, newprob >= oldprob,
                                                                    reduction))

            if reduction <= threshold:
                run = False


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

    # pre k-means
    pl.subplot(321)
    for i in range(len(clusters)):
        print('set %d: %s' % (i, clusters[i].shape))
        pl.plot(clusters[i][:, 0], clusters[i][:, 1], '.')
    pl.subplot(323)
    for i in range(len(means)):
        print('set %d: %s' % (i, means[i].shape))
        pl.plot(means[i][0], means[i][1], '.')
    print('means.shape =', means.shape)
    pl.subplot(325)
    pl.plot(featsindices, variances.T)
    pl.xticks(featsindices)
    pl.xlim(featsindices[0], featsindices[-1])
    print('variances.shape =', means.shape)

    (clusters, means, variances) = kmeans(featsvec, means, M)
    # pos k-means
    pl.subplot(322)
    for i in range(len(clusters)):
        print('set %d: %s' % (i, clusters[i].shape))
        pl.plot(clusters[i][:, 0], clusters[i][:, 1], '.')
    pl.subplot(324)
    for i in range(len(means)):
        print('set %d: %s' % (i, means[i].shape))
        pl.plot(means[i][0], means[i][1], '.')
    print('means.shape =', means.shape)
    pl.subplot(326)
    pl.plot(featsindices, variances.T)
    pl.xticks(featsindices)
    pl.xlim(featsindices[0], featsindices[-1])
    print('variances.shape =', variances.shape)

    M = 32
    print('Creating GMM (M = %d)...' % M)
    t = time.time()
    gmm = GMM(M, featsvec)
    t = time.time() - t
    print('GMM created in %f seconds' % t)
    t = time.time()
    log_likelihood = gmm.log_likelihood(featsvec)
    t = time.time() - t
    print('GMM log-likelihood = %f in %f seconds' % (log_likelihood, t))

    pl.show()