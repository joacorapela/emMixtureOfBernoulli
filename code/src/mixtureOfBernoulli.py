import numpy as np
import scipy.special


class MixtureOfBernoulli:

    def _computeLogProbXnGivenPk(self, xn, logPk):
        log1mPk = np.log(1-np.exp(logPk))
        prob = np.sum(logPk[np.nonzero(xn)]) + \
            np.sum(log1mPk[np.nonzero(1-xn)])
        return prob

    def computeMixtureProbsWithLogProbs(self, x, logP):
        # P \in K x D
        N = x.shape[0]
        K = logP.shape[0]
        assert x.shape[1] == logP.shape[1]
        logP_NK = np.empty((N, K), dtype=np.double)

        for n in range(N):
            for k in range(K):
                logP_NK[n, k] = self._computeLogProbXnGivenPk(xn=x[n, :],
                                                             logPk=logP[k, :])

        return logP_NK

    def computeLogLikelihoodWithLogProbs(self, x, logPi, logP):
        # logPi \in K
        # logP  \in K x D
        logP_NK = self.computeMixtureProbsWithLogProbs(x=x, logP=logP)
        aux = scipy.special.logsumexp(np.expand_dims(logPi, 0) +
                                      logP_NK, axis=1)
        ll = np.sum(aux)
        return ll

    def _computeProbXnGivenPk(self, xn, pk):
        prob = np.prod(pk[np.nonzero(xn)])*np.prod(1-pk[np.nonzero(1-xn)])
        return prob

    def computeMixtureProbsWithNonLogProbs(self, x, P):
        # P \in K x D
        N = x.shape[0]
        K = P.shape[0]
        assert x.shape[1] == P.shape[1]
        P_NK = np.empty((N, K), dtype=np.double)

        for n in range(N):
            for k in range(K):
                P_NK[n, k] = self._computeProbXnGivenPk(xn=x[n, :],
                                                        pk=P[k, :])

        return P_NK

    def computeLogLikelihoodWithNonLogProbs(self, x, Pi, P):
        # Pi \in K
        # P \in K x D
        P_NK = self.computeMixtureProbsWithNonLogProbs(x=x, P=P)
        ll = np.sum(np.log(P_NK @ Pi))
        return ll

    def sample(self, Pi, P, tol=1e-6):
        # first choose the component to sample
        cum_Pi = np.cumsum(Pi)
        assert(np.abs(cum_Pi[-1]-1.0) < tol)
        a_rand = np.random.random_sample(size=1)
        diff = cum_Pi - a_rand
        k = np.nonzero(diff > 0)[0][0]

        # now sample Bernoulli's from the chosen component
        pk = P[k, :]
        D = len(pk)
        sample = np.random.binomial(n=1, p=pk)
        return sample
