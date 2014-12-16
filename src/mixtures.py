"""This module defines the GMM and contains functions to training and evaluation.
"""


import numpy as np
import math
import random


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
        self.num_features = len(mfccs)

        self.weights = list()
        self.meansvec = list()
        self.variancesvec = list()

        mean = np.mean(mfccs, axis=1)
        variance = np.std(mfccs, axis=1)**2
        D = self.num_features
        for m in range(self.num_mixtures):
            self.weights.append(math.fabs(random.choice(mean)) * random.choice(variance))
            self.meansvec.append([random.choice(mean) * random.uniform(-10, 10) for _ in range(D)])
            self.variancesvec.append([random.choice(variance) for _ in range(D)])

        self.weights = np.array(self.weights)
        self.weights = self.weights / np.sum(self.weights)
        self.meansvec = np.array(self.meansvec)
        self.variancesvec = np.array(self.variancesvec)

    def get_mixture(self, m):
        """@returns: the m-th mixture of the GMM (with 0 <= m < M).
        """
        return (self.weights[m], self.meansvec[m], self.variancesvec[m])

    #99.8% of reduction in time to compute, compared with the version with loop
    def eval(self, features, func=np.dot):
        """Feeds the GMM with the given features.

        @param features: a Dx1 vector of features.

        @returns: the weighted sum of gaussians for gmm.
        """
        D = self.num_features

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

            #TODO verificar se este for pode ser substituído por uma operação
            #matricial
            for i in range(self.num_mixtures):
                weight = self.weights[i]
                means = self.meansvec[i]
                variances = self.variancesvec[i]

                for t in range(T):
                    features = mfccs[t]
                    evaluated = self.eval(features)
                    posteriors[t] = self.eval(features, np.multiply) / evaluated

                #Summation from t=1 until t=T
                summed_posteriors = weight * np.sum(posteriors, axis=0)

                #Updating i-th weight
                new_weights[i] = one_Tth * summed_posteriors[i]

                #Updating i-th meansvec
                new_meansvec[i] = self.meansvec[i] #np.dot(posteriors[:, i], mfccs) / summed_posteriors[i]

                #For now, meansvec and variancesvec are the same
                new_variancesvec[i] = self.variancesvec[i]

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
            print('%f\nnewprob > oldprob ? %s\nreduction = %f' % (newprob, newprob > oldprob,
                                                                  reduction))
            if reduction <= threshold:
                run = False


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
    x = np.linspace(-200, 200, 1000)

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

#    #Evaluating GMM
#    print('Evaluating GMM with a %d-dimensional feature vector...' % numfeats)
#    t = time.time()
#    features = mfccs[0]
#    prob = gmm.eval(features)
#    t = time.time() - t
#    print('GMM evaluated in %f seconds' % t)
#
#    #log-likelihood of GMM
#    print('log-likelihood of GMM')
#    t = time.time()
#    log_likelihood = gmm.log_likelihood(mfccs)
#    t = time.time() - t
#    print('Log likelihood calculated in %f seconds' % t)
#    print('log p(X|lambda) = %f' % log_likelihood)

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