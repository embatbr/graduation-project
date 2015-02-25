"""This module defines the GMM and contains functions for training and evaluation.
"""


import numpy as np
from math import pi as PI
from common import ZERO


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
        self.weights = np.tile(1 / nummixtures, nummixtures)
        self.meansvec = np.random.random((nummixtures, self.numfeats))*10 # 0 t0 10
        self.variancesvec = (np.random.random((nummixtures, self.numfeats)) + 1)*100 # 100 to 200

    def eval(self, feats):
        """Feeds the GMM with the given features. It performs a normal pdf for a
        number M of mixtures (one for each m from M).

        @param feats: a Dx1 vector of features.
        @param sumProbs: tells if the weighted mixtures must be summed. Default
        is True.

        @returns: a tuple: the weighted gaussians for gmm, summed and as array.
        """
        #Denominator of constant
        D = self.numfeats
        determinant = np.prod(self.variancesvec, axis=1)
        cte = ((2*PI)**(D/2)) * (determinant**(1/2))

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
        posteriors = np.zeros((T, self.nummixtures))
        old_log_like = None

        run = True
        while run:
            # E-Step
            for t in range(T):
                (likelihood_in_t, w_gaussians) = self.eval(featsvec[t]) # one for each one of M mixtures
                posteriors[t] = w_gaussians / likelihood_in_t

            #Summation of posteriors from t=1 until t=T
            sum_posteriors = np.sum(posteriors, axis=0)

            # M-Step
            for i in range(self.nummixtures):
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

            new_log_like = self.log_likelihood(featsvec)
            if not old_log_like is None:
                reduction = (old_log_like - new_log_like) / old_log_like

                if reduction < 0: # If enters here, it's wrong
                    print('WRONG!')
                    wrong = open('wrong.log', 'w')
                    wrong.write('MLE by EM algorithm is not monotonically increasing.\n')
                    wrong.write('old_log_like = %f.\n' % old_log_like)
                    wrong.write('new_log_like = %f.\n' % new_log_like)
                    wrong.write('reduction = %f.\n' % reduction)
                    wrong.write('sum_posteriors = %f.\n' % sum_posteriors)
                    wrong.write('self.weights = %f.\n' % self.weights)
                    wrong.write('self.meansvec = %f.\n' % self.meansvec)
                    wrong.write('self.variancesvec = %f.\n' % self.variancesvec)
                    break
                elif reduction <= threshold:
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
    numcep = 13
    delta_order = 0
    Ms = [2**n for n in range(3, 11)]

    featsvec = bases.read_mit_speaker_features(numcep, delta_order, 'enroll_1', 'f02')
    test_featsvec = bases.read_mit_features(numcep, delta_order, 'enroll_1', 'f02', 1)

    for M in Ms:
        #creation
        t = time.time()
        gmm = GMM(M, featsvec)
        t = time.time() - t
        print('GMM (M = %d) created in %f seconds' % (M, t))
        untrained_log_likelihood = gmm.log_likelihood(test_featsvec)
        print('untrained GMM: log-likelihood = %f' % untrained_log_likelihood)
        fig = pl.figure()
        fig.suptitle('M = %d' % M)
        pl.subplot(221)
        pl.plot(featsvec[:, 0], featsvec[:, 1], '.')
        pl.subplot(223)
        for means in gmm.meansvec:
            pl.plot(means[0], means[1], 'o')

        #training
        t = time.time()
        gmm.train(featsvec)
        t = time.time() - t
        print('EM training in %f seconds' % t)
        trained_log_likelihood = gmm.log_likelihood(test_featsvec)
        print('trained GMM: log-likelihood =', trained_log_likelihood)
        pl.subplot(222)
        pl.plot(featsvec[:, 0], featsvec[:, 1], '.')
        pl.subplot(224)
        for means in gmm.meansvec:
            pl.plot(means[0], means[1], 'o')

        increase = 1 - (trained_log_likelihood / untrained_log_likelihood)
        print('increase = %2.2f%%' % (increase*100))

    pl.show()