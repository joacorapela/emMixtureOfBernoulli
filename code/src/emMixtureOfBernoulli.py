import sys
import os
import abc
import numpy as np
import mixtureOfBernoulli

sys.path.append(os.path.expanduser(
    "~/dev/research/programs/repos/python/joacorapela_common/src/"))
import joacorapela_common.utils.numerical_methods


class EMmixtureOfBernoulli(abc.ABC):

    def __init__(self, x):
        # x \in N x D
        self._x = x

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, x):
        # x \in N x D
        self._x = x

    @abc.abstractmethod
    def optimize(self, Pi0, P0, tol=1e-5, max_iter=1000):
        # Pi0 \in K
        # P0  \in K x D
        Pi = Pi0
        P = P0
        lls = []
        ll = self._computeLogLikelihood(Pi=Pi, P=P)
        lls.append(ll)
        i = 0
        print(f"LL({i})={ll}")
        converged = False
        while i < max_iter and not converged:
            R = self._e_step(Pi=Pi, P=P)
            Pi, P = self._m_step(R=R)
            i += 1
            prev_ll = ll
            ll = self._computeLogLikelihood(Pi=Pi, P=P)
            lls.append(ll)
            if np.abs(prev_ll-ll) < tol:
                converged = True
            print(f"LL({i})={ll}, LL increase={ll-prev_ll}")
        return Pi, P, R, lls

    @abc.abstractmethod
    def _computeLogLikelihood(Pi, P):
        pass

    @abc.abstractmethod
    def _e_step(self, Pi, P):
        pass

    def _m_step(self, R):
        # R \in N x K
        N_K = np.sum(R, 0)          # \in K
        X_bar_scaled = R.T @ self.x  # \in K x D
        X_bar = X_bar_scaled / np.expand_dims(N_K, 1)  # \in K x D
        Pi = N_K/np.sum(N_K)
        P = X_bar
        return Pi, P


class EMmixtureOfBernoulliNonLogProb(EMmixtureOfBernoulli):

    def optimize(self, Pi0, P0, tol=1e-5, max_iter=1000):
        return super().optimize(Pi0=Pi0, P0=P0, tol=tol, max_iter=max_iter)

    def _computeLogLikelihood(self, Pi, P):
        mob = mixtureOfBernoulli.MixtureOfBernoulli()
        ll = mob.computeLogLikelihoodWithNonLogProbs(x=self.x, Pi=Pi, P=P)
        return ll

    def _e_step(self, Pi, P):
        # Pi \in 1 x K
        # P \in K x D
        mob = mixtureOfBernoulli.MixtureOfBernoulli()
        P_NK = mob.computeMixtureProbsWithNonLogProbs(x=self.x, P=P)
        num = P_NK * np.expand_dims(Pi, 0)  # brodcasting, num \in N x K
        den = np.sum(num, 1)  # den \in N
        R = num / np.expand_dims(den, 1)  # R \in N x K
        return R


class EMmixtureOfBernoulliLogProb(EMmixtureOfBernoulli):

    def optimize(self, Pi0, P0, tol=1e-5, max_iter=1000):
        return super().optimize(Pi0=np.log(Pi0), P0=np.log(P0), tol=tol,
                                max_iter=max_iter)

    def _computeLogLikelihood(self, Pi, P):
        logPi = Pi
        logP = P
        mob = mixtureOfBernoulli.MixtureOfBernoulli()
        ll = mob.computeLogLikelihoodWithLogProbs(x=self.x, logPi=logPi,
                                                  logP=logP)
        return ll

    def _e_step(self, Pi, P):
        # Pi \in 1 x K
        # P \in K x D
        logPi = Pi
        logP = P
        mob = mixtureOfBernoulli.MixtureOfBernoulli()
        logP_NK = mob.computeMixtureProbsWithLogProbs(x=self.x, logP=logP)
        Y = np.expand_dims(logPi, 0) + logP_NK
        R = np.empty(shape=Y.shape)
        N = R.shape[0]
        for n in range(N):
            R[n, :] = joacorapela_common.utils.numerical_methods.exp_normalize(
                Y[n, :])
        return R

    def _m_step(self, R, epsilon=1e-9):
        Pi, P = super()._m_step(R=R)
        logPi = np.log(Pi)
        P[P < epsilon] = epsilon
        P[P > 1-epsilon] = 1-epsilon
        logP = np.log(P)
        return logPi, logP
