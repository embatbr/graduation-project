"""This module defines the GMM and contains functions for training and evaluation.
"""


import numpy as np
import random
from math import pi as PI
from common import ZERO, MIN_VARIANCE


class GMM(object):
    """Represents a GMM with number of mixtures M and a D-variate gaussian.
    """
    def __init__(self, M, featsvec):
        #TODO achar o ponto central de 'featsvec', o desvio padrão e criar as médias
        #de modo aleatório, dentro de uma super-elipse (elipse D-dimensional) cujo
        #raio na dimensão d seja o desvio padrão nesta dimensão.
        #TODO as variâncias serão o quadrado do desvio padrão calculado acima.
        #TODO adicionar um atributo 'gender', com valores em {'f', 'm'}.
        """Creates a GMM.

        @param M: number of mixtures (integer).
        @param featsvec: features used by the GMM.
        """
        self.M = M
        self.D = featsvec.shape[1]
        self.weights = np.tile(1 / M, M)
        amax = np.amax(featsvec, axis=0)
        amin = np.amin(featsvec, axis=0)
        self.meansvec = np.array([random.uniform(amin, amax) for _ in range(M)])
        variances = np.std(featsvec, axis=0)**2
        self.variancesvec = np.ones((M, self.D))

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

    def train(self, featsvec, threshold=1E-2):
        """Trains the given GMM with the sequence of given feature vectors. Uses
        the EM algorithm.

        @param featsvec: a NUMFRAMES x D matrix of features.
        @param threshold: the difference between old and new probabilities must be
        lower than (or equal to) this parameter in %. Default 0.01 (1%).
        """
        T = len(featsvec)
        posteriors = np.zeros((T, self.M))
        old_log_like = self.log_likelihood(featsvec)

        run = True
        while run:
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
                #BUG: means is receiving 'nan'
                self.meansvec[i] = np.dot(posteriors[:, i], featsvec)
                self.meansvec[i] = self.meansvec[i] / sum_posteriors[i]

                #Updating i-th variancesvec
                self.variancesvec[i] = np.dot(posteriors[:, i], featsvec**2)
                self.variancesvec[i] = self.variancesvec[i] / sum_posteriors[i]
                self.variancesvec[i] = self.variancesvec[i] - self.meansvec[i]**2
                self.variancesvec[i] = np.where(self.variancesvec[i] < MIN_VARIANCE,
                                                MIN_VARIANCE, self.variancesvec[i])

            new_log_like = self.log_likelihood(featsvec)
            reduction = (old_log_like - new_log_like) / old_log_like
            if reduction <= threshold:
                run = False

            old_log_like = new_log_like


if __name__ == '__main__':
    import scipy.io.wavfile as wavf
    import os, os.path, shutil
    import time
    import pylab as pl
    import bases

    from common import CORPORA_DIR, FEATURES_DIR


    winlen = 0.02
    winstep = 0.01
    numcep = 6
    delta_order = 0
    Ms = [32, 64, 128]

    featsvec = bases.read_mit_speaker_features(numcep, delta_order, 'enroll_1', 'f02')
    test_featsvec = bases.read_mit_features(numcep, delta_order, 'enroll_1', 'f02', 1)
    print('featsvec.shape:', featsvec.shape)
    print('test_featsvec.shape:', test_featsvec.shape)

    T = len(featsvec)
    def normal(x, mean, variance):
        cte_denom = (2 * np.pi * variance)**0.5
        power = -0.5 * ((x - mean)**2) / variance
        return np.exp(power) / cte_denom

    (amin, amax) = (np.amin(featsvec, axis=0), np.amax(featsvec, axis=0))
    (mean, variance) = (np.mean(featsvec, axis=0), np.std(featsvec, axis=0)**2)
    X = [np.linspace(amin[d], amax[d], T) for d in range(numcep)]
    #[x.sort(axis=0) for x in X]

    # Checking prob distribution of features
    for d in range(numcep):
        y = normal(X[d], mean[d], variance[d])
        position = 231 + d
        pl.subplot(position)
        pl.plot(X[d], y)

    for M in Ms:
        #creation
        t = time.time()
        gmm = GMM(M, featsvec)
        t = time.time() - t
        print('GMM (M = %d) created in %f seconds' % (M, t))
        untrained_log_likelihood = gmm.log_likelihood(test_featsvec)
        print('untrained GMM: log-likelihood = %f' % untrained_log_likelihood)
        fig = pl.figure()
        fig.suptitle('M = %d, untrained' % M)
        for d in range(numcep):
            position = 231 + d
            pl.subplot(position)
            pl.xlabel('feature %d' % (d + 1))
            mixture = list()
            for i in range(M):
                mean = gmm.meansvec[i, 1]
                variance = gmm.variancesvec[i, 1]
                y = normal(X[0], mean, variance)
                mixture.append(gmm.weights[i] * y)
                pl.plot(X[0], y, 'b--')
            mixture = np.array(sum(mixture))
            pl.plot(X[0], mixture, 'r')

        #training
        t = time.time()
        gmm.train(featsvec)
        t = time.time() - t
        print('EM training in %f seconds' % t)
        trained_log_likelihood = gmm.log_likelihood(test_featsvec)
        print('trained GMM: log-likelihood =', trained_log_likelihood)
        fig = pl.figure()
        fig.suptitle('M = %d, trained' % M)
        for d in range(numcep):
            position = 231 + d
            pl.subplot(position)
            pl.xlabel('feature %d' % (d + 1))
            mixture = list()
            for i in range(M):
                mean = gmm.meansvec[i, 1]
                variance = gmm.variancesvec[i, 1]
                y = normal(X[0], mean, variance)
                mixture.append(gmm.weights[i] * y)
                pl.plot(X[0], y, 'b--')
            mixture = np.array(sum(mixture))
            pl.plot(X[0], mixture, 'r')

        increase = 1 - (trained_log_likelihood / untrained_log_likelihood)
        print('increase = %2.2f%%' % (increase*100))

    pl.show()