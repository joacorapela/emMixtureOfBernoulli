
import sys
import numpy as np
sys.path.append("..")
import mixtureOfBernoulli
import emMixtureOfBernoulli


def test_computeProbXnGivenPk(tol=1e-6):
    xn = np.array([0, 1, 1, 0, 0])
    pk = np.array([0.1, 0.8, 0.5, 0.4, 0.7])
    # prob = 0.9 * 0.8 * 0.5 * 0.6 * 0.3 = 0.06480000000000001
    logPk = np.log(pk)

    mob = mixtureOfBernoulli.MixtureOfBernoulli()
    logPXGivenPk = mob._computeLogProbXnGivenPk(xn=xn, logPk=logPk)
    pxGivenPk = mob._computeProbXnGivenPk(xn=xn, pk=pk)

    expLogPxGivenPK = np.exp(logPXGivenPk)
    assert np.abs(expLogPxGivenPK-pxGivenPk) < tol


def test_computeMixtureProbs(tol=1e-6, K=3,
                             images_filename ="../../../data/binarydigits.txt"):

    images = np.loadtxt(images_filename)
    N = images.shape[0]
    D = images.shape[1]
    P = np.random.uniform(low=0.25, high=0.75, size=(K, D))
    logP = np.log(P)

    mob = mixtureOfBernoulli.MixtureOfBernoulli()
    logMixtureProbs = mob.computeMixtureProbsWithLogProbs(x=images, logP=logP)
    mixtureProbs = mob.computeMixtureProbsWithNonLogProbs(x=images, P=P)

    expLogMixtureProbs = np.exp(logMixtureProbs)
    for n in range(N):
        for k in range(K):
            assert np.abs(expLogMixtureProbs[n, k]-mixtureProbs[n, k]) < tol


def test_computeLogLikelihood(tol=1e-6, K=3,
                              images_filename ="../../../data/binarydigits.txt"):

    images = np.loadtxt(images_filename)
    D = images.shape[1]
    P = np.random.uniform(low=0.25, high=0.75, size=(K, D))
    Pi = np.random.uniform(size=K)
    Pi = Pi/np.sum(Pi)
    logP = np.log(P)
    logPi = np.log(Pi)

    mob = mixtureOfBernoulli.MixtureOfBernoulli()
    ll_log = mob.computeLogLikelihoodWithLogProbs(x=images, logPi=logPi, logP=logP)
    ll_nonLog = mob.computeLogLikelihoodWithNonLogProbs(x=images, Pi=Pi, P=P)

    assert np.abs(ll_log-ll_nonLog) < tol


def test_e_step(tol=1e-6, K=3,
                images_filename ="../../../data/binarydigits.txt"):

    images = np.loadtxt(images_filename)
    N = images.shape[0]
    D = images.shape[1]
    P = np.random.uniform(low=0.25, high=0.75, size=(K, D))
    Pi = np.random.uniform(size=K)
    Pi = Pi/np.sum(Pi)
    logP = np.log(P)
    logPi = np.log(Pi)

    emobLog = emMixtureOfBernoulli.EMmixtureOfBernoulliLogProb(x=images)
    R_log = emobLog._e_step(Pi=logPi, P=logP)

    emobNonLog = emMixtureOfBernoulli.EMmixtureOfBernoulliNonLogProb(x=images)
    R_nonLog = emobNonLog._e_step(Pi=Pi, P=P)

    for n in range(N):
        for k in range(K):
            assert np.abs(R_log[n, k]-R_nonLog[n, k]) < tol


if __name__ == "__main__":
    test_computeProbXnGivenPk()
    test_computeMixtureProbs()
    test_computeLogLikelihood()
    test_e_step()
