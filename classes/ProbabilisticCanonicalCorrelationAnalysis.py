"""Probabilistic CCA.
This is an EM algorithm for probabilistic CCA based on the graphical model in 

Bach and Jordan 2006. A Probabilistic Interpretation of Canonical Correlation
Analysis. https://www.di.ens.fr/~fbach/probacca.pdf

While it converges exactly onto the SVD solution of CCA, it gives the added
benefit of log likelihood values for different numbers of canonical directions
on test data, which allows us to replace bootstrapping tests to determine the
optimal number of canonical directions.
"""

# Author: Emilio Salazar <emilio.salazarcardozo@gmail.com>

import warnings
from math import sqrt, log
import numpy as np
from scipy import linalg
import scipy as sp


from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array, check_random_state
from sklearn.utils.extmath import fast_logdet, randomized_svd, squared_norm
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import ConvergenceWarning


class ProbabilisticCanonicalCorrelationAnalysis(BaseEstimator, TransformerMixin):
    """Probabilistic Canonical Correlation Analysis (pCCA)

    A simple linear generative model with Gaussian latent variables.

    The observations from two different sigansl are assumed to be caused by a
    linear transformation of a single set of lower dimensional latent factors
    that are shared across the areas, and added Gaussian noise.
    Without loss of generality the factors are distributed according to a
    Gaussian with zero mean and unit covariance. The noise for each signal is
    also zero mean but has a full rank covariance matrix. The objective of
    probabilistic CCA is to find a low rank representation of the
    cross-covariance between the two signals.

    If we would restrict the model further, by assuming that the Gaussian
    noise is diagonal for each signal we would obtain a factor analysis model
    whose objective is to find a low rank representation of the entire
    covariance matrix across both signals.

    pCCA performs a maximum likelihood estimate of the so-called
    `loading` matrix, the transformation of the latent variables to the
    observed ones from each signal, using expectation-maximization (EM).

    Note that for a signals X_1 and X_2 (both dim_1 or dim_2 x num_observation
    matrixes), the n paired L_1 and L_2 canonical directions found by pCCA will
    be equivalent first n paired directions for CCA, which are equivalent to
      L_1 = (Sig_11)^{-1/2} @ u_1
    and  
      L_2 = (Sig_22)^{-1/2} @ u_2,
    where u_1 and u_2 are the first n left and right singular vectors of
      (Sig_11)^{-1/2} @ Sig_12 @ (Sig_22)^{-1/2}
    (see Bach and Jordan 2006 for a more thorough treatment). However, pCCA
    gives the added benefit of providing a log likelihood of the fit for the
    directions on a test set, which allows us to determine the optimal number of
    canonical directions to explain the data with the model.

    Parameters
    ----------
    n_components : int | None
        Number of canonical directions, equivalent to the dimensionality of the
        latent space, the number of components of ``X`` that are obtained after
        ``transform``.  If None, n_components is set to the number of features.

    tol : float
        Stopping tolerance for EM algorithm.

    copy : bool
        Whether to make a copy of X. If ``False``, the input X gets overwritten
        during fitting.

    max_iter : int
        Maximum number of iterations.

    noise_variance_init : None | array, shape=(n_features,)
        The initial guess of the noise variance for each feature.
        If None, it defaults to np.ones(n_features)

    random_state : int, RandomState instance or None, optional (default=0)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`. Only used when ``svd_method`` equals 'randomized'.

    Attributes
    ----------
    components_1_ : array, [n_components, n_features_1]
        Canonical directions for signal 1

    components_2_ : array, [n_components, n_features_2]
        Canonical directions for signal 2

    loglike_ : list, [n_iterations]
        The log likelihood at each iteration.

    noise_variance_1_ : array, shape=(n_features_1,n_features_1)
        The estimated noise variance for signal 1

    noise_variance_2_ : array, shape=(n_features_2,n_features_2)
        The estimated noise variance for signal 2

    n_iter_ : int
        Number of iterations run.

    Examples
    --------
    >>> from sklearn.datasets import load_digits
    >>> from somewhere import ProbabilisticCanonicalCorrelationAnalysis
    >>> X, Y = load_digits(return_X_y=True)
    >>> transformer = ProbabilisticCanonicalCorrelationAnalysis(n_components=7, random_state=0)
    >>> X_transformed = transformer.fit_transform(X, Y)
    >>> X_transformed.shape
    (1797, 7)

    References
    ----------
    ..  Bach and Jordan 2006. A Probabilistic Interpretation of Canonical Correlation
         Analysis. https://www.di.ens.fr/~fbach/probacca.pdf

    See also
    --------
    FA: Factor analysis is also a latent linear variable model
        which assumed each features has independent variance (as opposed to
        grouping signals into two sets each of which have a full rank noise
        variance) 
    """
    def __init__(self, n_components=None, tol=1e-8, copy=True, max_iter=int(1e8),
                 noise_variance_init=None, random_state=0):
        self.n_components = n_components
        self.copy = copy
        self.tol = tol
        self.max_iter = max_iter
        self.fit_style_ = 'pcca'

        self.noise_variance_init = noise_variance_init
        self.random_state = random_state

    def fit(self, X_1, X_2):
        """Fit the pCCA model to X and Y using EM

        Parameters
        ----------
        X_1 : array-like, shape (n_samples, n_features_1)
            Training data.

        X_2 : array-like, shape (n_samples, n_features_2)
            Training data.

        Returns
        -------
        self
        """
        X_1 = check_array(X_1, copy=self.copy, dtype=np.float64)
        X_2 = check_array(X_2, copy=self.copy, dtype=np.float64)

        n_samples_1, n_features_1 = X_1.shape
        n_samples_2, n_features_2 = X_2.shape

        if n_samples_1 != n_samples_2:
            raise ValueError('number of observations in both signals must be equivalent')
        n_samples = n_samples_1

        n_components = self.n_components
        self.mean_1_ = np.mean(X_1, axis=0)
        self.mean_2_ = np.mean(X_2, axis=0)
        self.mean_ = np.hstack([self.mean_1_, self.mean_2_])
        n_features = n_features_1+n_features_2

        X = np.hstack([X_1, X_2])
        Xr = X - self.mean_
        XrXr = Xr.T @ Xr

        covX = np.cov(X, rowvar=False, ddof=0)

        if n_components is None:
            n_components = np.min([n_features_1,n_features_2])
        elif n_components==0:
            self.fit_style_ = "uncorrelated areas"
            self.components_1_ = np.zeros((n_features_1, 0))
            self.components_2_ = np.zeros((n_features_2, 0))
            self.components_ = np.zeros((n_features, 0))
            self.noise_variance_1_ = covX[:n_features_1, :n_features_1]
            self.noise_variance_2_ = covX[n_features_1:, n_features_1:]

            noise_variance = np.hstack([np.vstack([self.noise_variance_1_, np.zeros((n_features_2, n_features_1))]), np.vstack([np.zeros((n_features_1, n_features_2)),self.noise_variance_2_])])
            self.noise_variance_ = noise_variance
            n_obs = n_samples

            XrXrUncorr = np.hstack([np.vstack([XrXr[:n_features_1, :n_features_1], np.zeros((n_features_2, n_features_1))]), np.vstack([np.zeros((n_features_1, n_features_2)),XrXr[n_features_1:, n_features_1:]])])
            log_like = -0.5 * (n_obs*n_features*np.log(2*np.pi) + n_obs*fast_logdet(noise_variance) + np.sum(np.sum(np.linalg.inv(noise_variance) * XrXrUncorr)))

            self.loglike_ = [log_like]
            self.n_iter_ = 0
            self.finalRatioChange_ = np.nan
            self.finalDiffChange_ = np.nan
            return self



        # some constant terms
        nsqrt = sqrt(n_samples)
        llconst = -n_features/2 * log(2. * np.pi)

        # if self.noise_variance_init is None:
        #     psi = np.ones(n_features, dtype=X.dtype)
        # else:
        #     if len(self.noise_variance_init) != n_features:
        #         raise ValueError("noise_variance_init dimension does not "
        #                          "with number of features : %d != %d" %
        #                          (len(self.noise_variance_init), n_features))
        #     psi = np.array(self.noise_variance_init)

        loglike = []
        old_ll = -np.inf
        SMALL = 1e-12

        if np.linalg.matrix_rank(covX) == n_features:
            scale = np.exp(2*np.sum(np.log(np.diag(np.linalg.cholesky(covX))))/n_features)
        else:
            # unlike Matlab, np's cholesky below fails if you're not full
            # rank... kind of correctly, honestly
            # At least I remember it doing so... let's try again
            warnings.warn('ProbabilisticCanonicalCorrelationAnalysis data matrix is not full rank')
            r = np.linalg.matrix_rank(covX)
            e = np.sort(np.linalg.eig(covX)[0])[::-1]
            scale = sp.stats.mstats.gmean(e[:r])
        #    raise Exception("pCCA:NumObs", "Not enough observations! Rank covariance mat = " + str(np.linalg.matrix_rank(covX)) + ". Num features = " + str(n_features))

        Uinit1, _, _ = np.linalg.svd(np.random.randn(n_features_1, n_components))
        Uinit1 = Uinit1[:, :n_components]
        Uinit2, _, _ = np.linalg.svd(np.random.randn(n_features_2, n_components))
        Uinit2 = Uinit2[:, :n_components]
        Winit = []
        Winit.append(covX[:n_features_1, :n_features_1] @ Uinit1) # np.random(cov1, dir1, M1)
        Winit.append(covX[n_features_1:, n_features_1:] @ Uinit2) # (cov2, dir2, M2)
        Wnew = np.vstack(Winit)
        Wnew = Wnew/np.linalg.norm(Wnew,axis=0)

        psiNew = covX - Wnew @ Wnew.T
        psiNew[:n_features_1, n_features_1:] = 0
        psiNew[n_features_1:, :n_features_1] = 0

        I = np.eye(n_components)
        for i in range(self.max_iter):
            ## MY STUFF
            Wold = Wnew
            psiOld = psiNew

            C = Wold @ Wold.T + psiOld
            # At some point Cinv could be sped up with the Woodbury matrix
            # identity and speeding up the inversion of the block matrix
            # psiOld...
            Cinv = np.linalg.inv(C)
            # note below that (XrXr @ Cinv).trace() == (trainSignalNoMean.T @ Cinv @ trainSignalNoMean).trace()
            ll_new = -n_samples*n_features/2*np.log(2*np.pi) - n_samples/2*fast_logdet(C) - 1/2 * (XrXr @ Cinv).trace()
            
            # print('{} ; {}'.format(numObservations/2*np.prod(np.linalg.slogdet(C)), (XrXr @ Cinv).trace()))

            loglike.append(ll_new)

            # M-step
            Wnew = (XrXr @ Cinv @ Wold) @ np.linalg.inv(n_samples*I - n_samples*Wold.T @ Cinv @ Wold + Wold.T @ Cinv @ XrXr @ Cinv @ Wold)
            psiNewInit = 1/n_samples * (XrXr-Wnew @ Wold.T @ Cinv @ XrXr)
            psiNew = np.vstack([np.hstack([psiNewInit[:n_features_1, :n_features_1], np.zeros((n_features_1, n_features_2))]),
                np.hstack([np.zeros((n_features_2, n_features_1)), psiNewInit[n_features_1:, n_features_1:]])])

            if i <= 2:
                ll_base = ll_new
            elif ll_new < old_ll:
                print("VIOLATION")
            elif ((ll_new - ll_base) < (1+self.tol)*(old_ll - ll_base)):
                break

            old_ll = ll_new
        else:
            warnings.warn('ProbabilisticCanonicalCorrelationAnalysis did not converge.' +
                          ' You might want' +
                          ' to increase the number of iterations.',
                          ConvergenceWarning)

        self.components_ = Wnew

        self.noise_variance_1_ = psiNew[:n_features_1, :n_features_1]
        self.noise_variance_2_ = psiNew[n_features_1:, n_features_1:]
        self.noise_variance_ = psiNew

        W1 = Wnew[:n_features_1]
        W2 = Wnew[n_features_1:]
        psi1 = self.noise_variance_1_
        psi2 = self.noise_variance_2_

        canonicalDirsOrthBaseArea1,s,canonicalDirsOrthBaseArea2T = np.linalg.svd(np.linalg.inv(np.linalg.cholesky(W1 @ W1.T + psi1)) @ W1 @ W2.T @ np.linalg.inv(np.linalg.cholesky(W2 @ W2.T + psi2)).T) 
        canonicalDirsOrthBaseArea2 = canonicalDirsOrthBaseArea2T.T
        self.components_1_ = (np.linalg.inv(np.linalg.cholesky(W1 @ W1.T + psi1)).T @ canonicalDirsOrthBaseArea1[:,:n_components])
        self.components_2_ = (np.linalg.inv(np.linalg.cholesky(W2 @ W2.T + psi2)).T @ canonicalDirsOrthBaseArea2[:,:n_components])


        if np.any(np.array(loglike) > 0):
            breakpoint()
        self.loglike_ = loglike
        self.n_iter_ = i + 1
        self.finalRatioChange_ = (ll_new-ll_base)/(old_ll-ll_base) - 1;
        self.finalDiffChange_ = (ll_new-old_ll);
        return self

    def transform(self, X_1, X_2):
        """Apply dimensionality reduction to X using the model.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data.

        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)
            The latent variables of X.
        """
        check_is_fitted(self, 'components_')

        X_1 = check_array(X_1, copy=self.copy, dtype=np.float64)
        X_2 = check_array(X_2, copy=self.copy, dtype=np.float64)

        X = np.hstack([X_1, X_2])

        X_transformed = X - self.mean_

        precision = self.get_precision()
        W = self.components_
        beta = W.T @ precision
        X_new = X_transformed @ beta.T

        return X_new

    def get_covariance(self):
        """Compute data covariance with the ProbabilisticCanonicalCorrelationAnalysis model.

        ``cov = components_.T * components_ + noise_variance``

        Returns
        -------
        cov : array, shape (n_features, n_features)
            Estimated covariance of data.
        """
        check_is_fitted(self, 'components_')

        W = self.components_
        psi = self.noise_variance_


        cov = W @ W.T + psi
        return cov

    def get_precision(self):
        """Compute data precision matrix with the FactorAnalysis model.

        Returns
        -------
        precision : array, shape (n_features, n_features)
            Estimated precision of data.
        """
        check_is_fitted(self, 'components_')

        n_features = self.components_.shape[1]

        # handle corner cases first
        if self.n_components == 0:
            return linalg.inv(self.noise_variance_)
        if self.n_components == n_features:
            return linalg.inv(self.get_covariance())

        psi = self.noise_variance_
        W = self.components_
        invPsi = linalg.inv(psi)

        iPsiW = invPsi @ W
        # Get precision using matrix inversion lemma
        precision = invPsi - iPsiW @ np.linalg.inv(np.eye(self.n_components) + W.T @ iPsiW) @ iPsiW.T
        return precision

    def score_samples(self, X_1, X_2):
        """Compute the log-likelihood of each sample

        Parameters
        ----------
        X_1 : array, shape (n_samples, n_features_1)
        X_2 : array, shape (n_samples, n_features_2)
            The data

        Returns
        -------
        ll : array, shape (n_samples,)
            Log-likelihood of each sample under the current model
        """
        check_is_fitted(self, 'components_')

        X_1 = check_array(X_1, copy=self.copy, dtype=np.float64)
        X_2 = check_array(X_2, copy=self.copy, dtype=np.float64)

        n_obs, n_features_1 = X_1.shape
        _, n_features_2 = X_2.shape

        X = np.hstack([X_1, X_2])

        Xr = X - self.mean_
        XrXr = Xr.T @ Xr
        n_features = X.shape[1]

        if self.fit_style_ == "uncorrelated areas":
            XrXrUncorr = np.hstack([np.vstack([XrXr[:n_features_1, :n_features_1], np.zeros((n_features_2, n_features_1))]), np.vstack([np.zeros((n_features_1, n_features_2)),XrXr[n_features_1:, n_features_1:]])])
            noise_variance = self.noise_variance_

            log_like = -0.5 * (n_obs*n_features*np.log(2*np.pi) + n_obs*fast_logdet(noise_variance) + np.sum(np.sum(np.linalg.inv(noise_variance) * XrXrUncorr)))
            return log_like

        precision = self.get_precision()

        const = -n_features/2*np.log(2*np.pi)


        W = self.components_
        log_like =  n_obs*const + 0.5*n_obs*fast_logdet(precision) - 0.5*np.sum(np.sum(precision * XrXr))       
        # log_like = -.5 * (Xr * (np.dot(Xr, precision))).sum(axis=1)
        # log_like -= .5 * (n_features * log(2. * np.pi)
        #                   - fast_logdet(precision))
        return log_like

    def score(self, X_1, X_2):
        """Compute the average log-likelihood of the samples

        Parameters
        ----------
        X_1 : array, shape (n_samples, n_features_1)
        X_2 : array, shape (n_samples, n_features_2)
            The data

        Returns
        -------
        ll : float
            Average log-likelihood of the samples under the current model
        """
        return np.mean(self.score_samples(X_1, X_2))
