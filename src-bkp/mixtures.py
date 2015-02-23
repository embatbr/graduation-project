"""This module defines the GMM and contains functions to training and evaluation.
"""


import numpy as np
import math
import random
import pickle


PI = math.pi


class GMM(object):
    """Represents a GMM with number of mixtures M and a D-variate gaussian.
    """

    def __init__(self, num_mixtures, mfccs):
        """Creates a GMM.

        @param num_mixtures: number of mixtures (integer).
        @param mfccs: the features over time used to train the GMM. This parameter
        allows the GMM to be more accurate representation of the features.
        """
        self.num_mixtures = num_mixtures
        self.num_featuresvec = len(mfccs)
        M = num_mixtures
        D = self.num_featuresvec

        self.weights = np.tile(1/M, self.num_mixtures)
        self.meansvec = np.random.uniform(np.amin(mfccs), np.amax(mfccs), (M, D))
        self.variancesvec = np.random.uniform(1, np.std(mfccs)**2, (M, D))

    def get_mixture(self, m):
        """@returns: the m-th mixture of the GMM (with 0 <= m < M).
        """
        return (self.weights[m], self.meansvec[m], self.variancesvec[m])

    #99.8% of reduction in time to compute, compared with the version with loop
    def eval(self, features, func=np.dot):
        """Feeds the GMM with the given features.

        @param features: a Dx1 vector of features.
        @param func: the function applied in the return. By default is 'numpy.dot',
        to calculate the evaluation of the GMM fed by a features vector.

        @returns: the weighted sum of gaussians for gmm.
        """
        D = self.num_featuresvec

        #Constant
        determinant = np.prod(self.variancesvec, axis=1)
        cte = (2*PI**D * determinant)**0.5

        #Exponent
        A = features - self.meansvec
        power = np.divide(A, self.variancesvec)
        power = np.multiply(power, A)
        power = np.sum(power, axis=1)
        power = -0.5*power

        #Probability
        prob = (np.exp(power) / cte)
        return func(self.weights, prob)

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

    #TODO colocar parametro 'debug', que diz se é para printar no console
    def train(self, mfccs, threshold=0.01):
        """Train the given GMM with the sequence of given feature vectors.

        @param gmm: the GMM used (a list of tuples (weight, means, variances)).
        @param mfccs: a D x NUMFRAMES matrix of features.
        @param threshold: the difference between old and new probabilities must be
        lower than (or equal to) this parameter.

        @returns: the average sum of logarithm of the weighted sum of gaussians for
        gmm for each feature vector.
        """
        #The new GMM; optimization to avoid create new arrays every iteration
        new_weights = np.zeros(self.weights.shape)
        new_meansvec = np.zeros(self.meansvec.shape)
        new_variancesvec = np.zeros(self.variancesvec.shape)

        T = len(mfccs)
        one_Tth = 1/T
        #posteriors = np.zeros((self.num_mixtures, T, self.num_mixtures))
        posteriors = np.zeros((T, self.num_mixtures))

        iteration = 1 #DEBUG
        run = True
        while run:
            print('GMM.train(), iter = %d' % iteration)
            iteration += 1

            for t in range(T):
                features = mfccs[t]
                evaluated = self.eval(features)
                post = self.eval(features, np.multiply)
                post = np.multiply(self.weights, post) / evaluated
                posteriors[t] = np.array(post, copy=False)

            #Summation from t=1 until t=T
            summed_posteriors = np.sum(posteriors, axis=0)

            #TODO verificar se este 'for' pode ser substituído por uma operação
            #matricial
            for i in range(self.num_mixtures):
                #Updating i-th weight
                #new_weights[i] = self.weights[i]
                new_weights[i] = one_Tth * summed_posteriors[i]

                #Updating i-th meansvec
                #new_meansvec[i] =  self.meansvec[i]
                new_meansvec[i] = np.dot(posteriors[:, i], mfccs)
                new_meansvec[i] = new_meansvec[i] / summed_posteriors[i]

                #Atualizãçao abaixo está dando pau...
                new_variancesvec[i] = np.dot(posteriors[:, i], mfccs**2)
                #note: faltou dividir por 'summed_posteriors[i]'
                new_variancesvec[i] = new_variancesvec[i] - new_meansvec[i]**2

            #Testing convergence
            print('CALCULATING oldprob')
            oldprob = self.log_likelihood(mfccs)
            #Updating weights, meansvec and variancesvec
            self.weights = new_weights
            self.meansvec = new_meansvec
            self.variancesvec = new_variancesvec
            print('%f\nCALCULATING newprob' % oldprob)
            newprob = self.log_likelihood(mfccs)
            reduction = (oldprob - newprob) / oldprob
            print('%f\nMONOTONIC ? %s\nreduction = %f' % (newprob, newprob >= oldprob,
                                                          reduction))

            if reduction <= threshold:
                run = False


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
    std = np.std(mfccs)
    x = np.linspace(np.amin(mfccs) - 3*std, np.amax(mfccs) + 3*std, 1000)

    M = 32
    print('Creating GMM (M = %d)...' % M)
    t = time.time()
    gmm = GMM(M, mfccs.T)
    t = time.time() - t
    print('GMM created in %f seconds' % t)
    for featnum in range(numfeats):
        filecounter = plotgmm(x, gmm, featnum, 'Untrained, M = %d, GMM[%d]' %
                              (M, featnum), 'x', 'pdf', filename, filecounter)
    print('Untrained GMM plotted')

    #log-likelihood of GMM
    print('log-likelihood of GMM')
    t = time.time()
    log_likelihood = gmm.log_likelihood(mfccs)
    t = time.time() - t
    print('Log likelihood calculated in %f seconds' % t)
    print('log p(X|lambda) = %f' % log_likelihood)

    #GMM training
    print('training GMM...')
    t = time.time()
    gmm.train(mfccs)
    t = time.time() - t
    print('GMM trained in %f seconds' % t)
    for featnum in range(numfeats):
        filecounter = plotgmm(x, gmm, featnum, 'Trained, M = %d, GMM[%d]' %
                              (M, featnum), 'x', 'pdf', filename, filecounter)
    print('Trained GMM plotted')

    print('Saving GMM into file')
    gmmfile = open('%sgmm' % IMAGES_MIXTURES_DIR, 'wb')
    pickle.dump(gmm, gmmfile)
    gmmfile.close()
    gmmfile = open('%sgmm' % IMAGES_MIXTURES_DIR, 'rb')
    gmm = pickle.load(gmmfile)
    for featnum in range(numfeats):
        filecounter = plotgmm(x, gmm, featnum, 'Loaded from file, M = %d, GMM[%d]' %
                              (M, featnum), 'x', 'pdf', filename, filecounter)
    print('Loaded GMM plotted')