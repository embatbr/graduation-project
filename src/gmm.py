"""This module defines the GMM and contains functions to training and evaluation.
"""


import numpy as np
import math
import random


def gaussian(features, means=np.array([0]), covmatrix=np.array([[1]])):
    """A multivariate (D features) gaussian pdf.

    @param features: Dx1 vector of features
    @param means: Dx1 vector of means (one for each feature).
    @param covariances: DxD matrix of covariances (usually a diagonal matrix).

    @returns: the value (float) of the gaussian pdf for the given parameters.
    """
    D = len(features)
    determinant = np.linalg.det(covmatrix)
    cte = 1 / ((2*math.pi)**(D/2) * determinant**0.5)
    inverse = np.linalg.inv(covmatrix)

    #(x - mu)' * inverse * (x - mu)
    #due to the structure of array, the A.T is made at the end
    A = np.array([(features - means).tolist()])
    power = np.dot(A, inverse)
    power = np.dot(power, A.T)
    power = -0.5*power[0, 0]

    return (cte * math.exp(power))

def create_gmm(M, D):
    """Creates a GMM with M mixtures and fed by a D-dimensional feature vector.

    @param M: number of mixtures.
    @param D: size of features (used to the means and covariances).

    @returns: an untrained GMM.
    """
    weights = np.array([random.uniform(M/2, M) for _ in range(M)])
    weights = weights / np.sum(weights)

    gmm = list()
    for weight in weights:
        means = np.array([random.uniform(-M/2, M/2) for _ in range(D)])
        variances = np.array([random.uniform(1, D/2) for _ in range(D)])
        covmatrix = np.diag(variances)  #nodal-diagonal Covariance Matrix
        mixture = (weight, means, covmatrix)
        gmm.append(mixture)

    return gmm

def eval_gmm(gmm, features):
    """Feeds the GMM with the given features.

    @param gmm: the GMM used (a list of tuples (weight, means, covariances)).
    @param features: a Dx1 vector of features.

    @returns: the logarithm of the weighted sum of gaussians for gmm.
    """
    prob = 0
    for mixture in gmm:
        (weight, means, covmatrix) = mixture
        prob = prob + weight*gaussian(features, means, covmatrix)

    return prob

def loglikelihood_gmm(gmm, mfccs):
    """Feeds the GMM with a sequence of feature vectors.

    @param gmm: the GMM used (a list of tuples (weight, means, covariances)).
    @param mfccs: a D x NUMFRAMES matrix of features.

    @returns: the average sum of logarithm of the weighted sum of gaussians for
    gmm for each feature vector, aka, the log-likelihood.
    """
    numframes = len(mfccs.T)
    probs = np.array([eval_gmm(gmm, features) for features in mfccs.T])
    return (np.sum(np.log10(probs)) / numframes)

def posterior(gmm_i, features, gmm):
    """Calculates the a posteriori probability for a tuple (weight_i, means_i, covmatrix_i)
    for the given GMM and a D x 1 vector of features.

    @param i: the index from 0 to (M - 1).
    @param features: a D x 1 feature vector.
    @param gmm: the current GMM.

    @returns: the a posteriori probability for the tuple (weight_i, means_i, covmatrix_i).
    """
    (weight_i, means_i, covmatrix_i) = gmm_i
    weighted_prior = weight_i*gaussian(features, means_i, covmatrix_i)
    evidence = sum([weight*gaussian(features, means, covmatrix) for (weight, means, covmatrix) in gmm])

    return (weighted_prior / evidence)

def posterior_array(gmm_i, mfccs, gmm):
    return np.array([posterior(gmm_i, features, gmm) for features in mfccs.T])

def train_gmm(gmm, mfccs, threshold=0.01):
    """Train the given GMM with the sequence of given feature vectors.

    @param gmm: the GMM used (a list of tuples (weight, means, covariances)).
    @param mfccs: a D x NUMFRAMES matrix of features.
    @param threshold: the difference between old and new probabilities must be
    lower than this parameter.

    @returns: the average sum of logarithm of the weighted sum of gaussians for
    gmm for each feature vector.
    """
    T = len(mfccs.T)
    old_gmm = gmm
    iteration = 1

    while True:
        print('[%d]\nCREATING new_gmm' % iteration)
        iteration += 1
        new_gmm = list()
        for old_gmm_i in old_gmm:   #old_gmm_i == (weight_i, means_i, covmatrix_i)
            (weight_i, means_i, covmatrix_i) = old_gmm_i
            posteriors = posterior_array(old_gmm_i, mfccs, old_gmm)
            summed_posterior = np.sum(posteriors)

            new_weight_i = summed_posterior / T
            new_means_i = np.dot(posteriors, mfccs.T) / summed_posterior
            #TODO atualizar a matriz de covariancias
            new_covmatrix_i = covmatrix_i #TODO usar new_covmatrix = numpy.diag(new_variances)

            newmixture_i = (new_weight_i, new_means_i, new_covmatrix_i)
            new_gmm.append(newmixture_i)

        print('CALCULATING oldprob')
        oldprob = loglikelihood_gmm(old_gmm, mfccs)
        print('%f\nCALCULATING newprob' % oldprob)
        newprob = loglikelihood_gmm(new_gmm, mfccs)
        diff_perc = (oldprob - newprob) / oldprob
        print('%f\nnewprob > oldprob ? %s\ndiff_perc = %f' % (newprob, newprob > oldprob,
                                                              diff_perc))
        if diff_perc < threshold:
            print('RETURNING new_gmm')
            return new_gmm

        old_gmm = new_gmm
        print()


#TESTS
if __name__ == '__main__':
    import scipy.io.wavfile as wavf
    import os, os.path, shutil
    from useful import CORPORA_DIR, TEST_IMAGES_DIR, plotpoints, plotgaussian
    from useful import plotgmm, plotfigure
    import math
    import corpus


    if not os.path.exists(TEST_IMAGES_DIR):
        os.mkdir(TEST_IMAGES_DIR)

    IMAGES_GMM_DIR = '%sgmm/' % TEST_IMAGES_DIR

    if os.path.exists(IMAGES_GMM_DIR):
            shutil.rmtree(IMAGES_GMM_DIR)
    os.mkdir(IMAGES_GMM_DIR)

    filecounter = 0
    filename = '%sfigure' % IMAGES_GMM_DIR

    numcep = 13
    numdeltas = 0
    numfeats = numcep*(numdeltas + 1)
    speaker = 'f08'
    voice = ('enroll_1', speaker)

    #Reading MFCCs from features base
    mfccs = corpus.read_speaker_features(numcep, numdeltas, speaker)
    means = np.array([np.mean(feat) for feat in mfccs])
    stds = np.array([np.std(feat)**2 for feat in mfccs])
    covmatrix = np.diag(stds)
    print(mfccs.shape, len(means), covmatrix.shape)

    numframes = len(mfccs.T)
    print('#frames = %d' % numframes)
    pdfs = np.array([gaussian(features, means, covmatrix) for features in mfccs.T])
    for n in range(numcep):
        (mean, variance) = (means[n], covmatrix[n][n])
        filecounter = plotgaussian(mfccs[n], pdfs, mean, variance, 'MFCCs[%d] %s\nN(%f, %f)' %
                                   (n, voice, mean, variance), 'MFCCs[%d]' % n,
                                   'gaussian', filename, filecounter)

    #Multivariate plotting
    print('Multivariate plotting')
    for i in range(numcep):
        (mu_i, sigma_i) = (means[i], covmatrix[i][i])
        for j in range(i + 1, numcep):
            (mu_j, sigma_j) = (means[j], covmatrix[j][j])
            filecounter = plotpoints(mfccs[i], mfccs[j],
                                     'MFCCs[%d] x MFCCs[%d] %s\nN([%f, %f], [%f, %f])' %
                                     (i, j, voice, mu_i, mu_j, sigma_i, sigma_j),
                                     'mfccs[%d]' % i, 'mfccs[%d]' % j, filename,
                                     filecounter, 'green')

    Ms = [2**n for n in range(5, 12)]
    for M in Ms:
        print('Plotting GMM, M = %d' % M)
        gmm = create_gmm(M, numfeats)
        x = np.linspace(-(M + numfeats)/2, (M + numfeats)/2, 1000)
        for featnum in range(numfeats):
            filecounter = plotgmm(x, gmm, featnum, 'M = %d, GMM[%d]' % (M, featnum),
                                  'x', 'pdf', filename, filecounter)

    #Feeding GMM
    print('Feeding GMM')
    print(mfccs.shape)
    M = 32
    gmm = create_gmm(M, numfeats)
    probs = list()
    for features in mfccs.T:
        prob = eval_gmm(gmm, features)
        probs.append(prob)

    probs = np.array(probs)
    logprobs = np.log10(probs)
    numframes = len(mfccs.T)
    log_likelihood = np.sum(logprobs) / numframes
    print('log p(X|lambda) = %f' % log_likelihood)
    frameindices = np.linspace(0, numframes, numframes, False)
    filecounter = plotfigure(frameindices, logprobs, 'Log of probability per frame',
                             'frame', 'log', filename, filecounter)

    #Evaluating GMM
    print('Evaluating GMM')
    log_likelihood = loglikelihood_gmm(gmm, mfccs)
    print('log p(X|lambda) = %f' % log_likelihood)

    #Training GMM
    print('Section: Training')
    voice = ('enroll_1', 'f08', 54)
    (dataset, speaker, speech) = voice
    mfccs = corpus.read_features(numcep, numdeltas, dataset, speaker, speech)
    #mfccs = corpus.read_speaker_features(numcep, numdeltas, speaker)
    print(mfccs.shape)

    M = 32
    gmm = create_gmm(M, numfeats)
    x = np.linspace(-(M + numfeats)/2, (M + numfeats)/2, 1000)
    print('training GMM...')
    new_gmm = train_gmm(gmm, mfccs)
    print('GMM trained!')
    for featnum in range(numfeats):
        filecounter = plotgmm(x, gmm, featnum, 'M = %d, GMM[%d], Untrained\n%s' %
                              (M, featnum, voice), 'x', 'pdf', filename, filecounter)
        filecounter = plotgmm(x, new_gmm, featnum, 'M = %d, GMM[%d], Trained\n%s' %
                              (M, featnum, voice), 'x', 'pdf', filename, filecounter)

    mfccs = corpus.read_speaker_features(numcep, numdeltas, speaker)
    M = 512
    gmm = create_gmm(M, numfeats)
    x = np.linspace(-(M + numfeats)/2, (M + numfeats)/2, 1000)
    print('training GMM...')
    new_gmm = train_gmm(gmm, mfccs)
    print('GMM trained!')
    for featnum in range(numfeats):
        filecounter = plotgmm(x, gmm, featnum, 'M = %d, GMM[%d], Untrained\n%s' %
                              (M, featnum, voice), 'x', 'pdf', filename, filecounter)
        filecounter = plotgmm(x, new_gmm, featnum, 'M = %d, GMM[%d], Trained\n%s' %
                              (M, featnum, voice), 'x', 'pdf', filename, filecounter)