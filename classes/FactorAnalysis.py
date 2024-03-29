"""Factor Analysis.

NOTE: Most of these comments (and even a lot of these methods) are from the
sklearn version, but this has been updated to reflect fastfa.m from Byron Yu!

A latent linear variable model.

FactorAnalysis is similar to probabilistic PCA implemented by PCA.score
While PCA assumes Gaussian noise with the same variance for each
feature, the FactorAnalysis model assumes different variances for
each of them.

This implementation is based on David Barber's Book,
Bayesian Reasoning and Machine Learning,
http://www.cs.ucl.ac.uk/staff/d.barber/brml,
Algorithm 21.1
"""

# Author: Christian Osendorfer <osendorf@gmail.com>
#         Alexandre Gramfort <alexandre.gramfort@inria.fr>
#         Denis A. Engemann <denis-alexander.engemann@inria.fr>

# License: BSD3

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


class FactorAnalysis(BaseEstimator, TransformerMixin):
    """Factor Analysis (FA)

    A simple linear generative model with Gaussian latent variables.

    The observations are assumed to be caused by a linear transformation of
    lower dimensional latent factors and added Gaussian noise.
    Without loss of generality the factors are distributed according to a
    Gaussian with zero mean and unit covariance. The noise is also zero mean
    and has an arbitrary diagonal covariance matrix.

    If we would restrict the model further, by assuming that the Gaussian
    noise is even isotropic (all diagonal entries are the same) we would obtain
    :class:`PPCA`.

    FactorAnalysis performs a maximum likelihood estimate of the so-called
    `loading` matrix, the transformation of the latent variables to the
    observed ones, using expectation-maximization (EM).

    Read more in the :ref:`User Guide <FA>`.

    Parameters
    ----------
    n_components : int | None
        Dimensionality of latent space, the number of components
        of ``X`` that are obtained after ``transform``.
        If None, n_components is set to the number of features.

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

    svd_method : {'lapack', 'randomized'}
        Which SVD method to use. If 'lapack' use standard SVD from
        scipy.linalg, if 'randomized' use fast ``randomized_svd`` function.
        Defaults to 'randomized'. For most applications 'randomized' will
        be sufficiently precise while providing significant speed gains.
        Accuracy can also be improved by setting higher values for
        `iterated_power`. If this is not sufficient, for maximum precision
        you should choose 'lapack'.

    iterated_power : int, optional
        Number of iterations for the power method. 3 by default. Only used
        if ``svd_method`` equals 'randomized'

    random_state : int, RandomState instance or None, optional (default=0)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`. Only used when ``svd_method`` equals 'randomized'.

    Attributes
    ----------
    components_ : array, [n_components, n_features]
        Components with maximum variance.

    loglike_ : list, [n_iterations]
        The log likelihood at each iteration.

    noise_variance_ : array, shape=(n_features,)
        The estimated noise variance for each feature.

    n_iter_ : int
        Number of iterations run.

    Examples
    --------
    >>> from sklearn.datasets import load_digits
    >>> from sklearn.decomposition import FactorAnalysis
    >>> X, _ = load_digits(return_X_y=True)
    >>> transformer = FactorAnalysis(n_components=7, random_state=0)
    >>> X_transformed = transformer.fit_transform(X)
    >>> X_transformed.shape
    (1797, 7)

    References
    ----------
    .. David Barber, Bayesian Reasoning and Machine Learning,
        Algorithm 21.1

    .. Christopher M. Bishop: Pattern Recognition and Machine Learning,
        Chapter 12.2.4

    See also
    --------
    PCA: Principal component analysis is also a latent linear variable model
        which however assumes equal noise variance for each feature.
        This extra assumption makes probabilistic PCA faster as it can be
        computed in closed form.
    FastICA: Independent component analysis, a latent variable model with
        non-Gaussian latent variables.
    """
    def __init__(self, n_components=None, tol=1e-8, copy=True, max_iter=int(1e8),
                 noise_variance_init=None, svd_method='randomized',
                 iterated_power=3, random_state=0):
        self.n_components = n_components
        self.copy = copy
        self.tol = tol
        self.max_iter = max_iter
        self.fit_style_ = 'fa'
        if svd_method not in ['lapack', 'randomized']:
            raise ValueError('SVD method %s is not supported. Please consider'
                             ' the documentation' % svd_method)
        self.svd_method = svd_method

        self.noise_variance_init = noise_variance_init
        self.iterated_power = iterated_power
        self.random_state = random_state

    def fit(self, X, y=None):
        """Fit the FactorAnalysis model to X using EM

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data.

        y : Ignored

        Returns
        -------
        self
        """
        X = check_array(X, copy=self.copy, dtype=np.float64)

        n_samples, n_features = X.shape
        n_components = self.n_components
        if n_components is None:
            n_components = n_features
        elif n_components==0:
            var = np.var(X, axis=0)
            covX = np.cov(X, rowvar=False, ddof=0)

            self.mean_ = np.mean(X, axis=0)
            self.fit_style_ = "independent Gaussian"
            self.components_ = np.zeros((n_features, 0))
            self.noise_variance_ = var

            n_features = X.shape[1]
            n_obs = X.shape[0]
            Xr = X - self.mean_
            Xstd = Xr**2/self.noise_variance_
            log_like = -0.5 * (n_obs*n_features*np.log(2*np.pi) + n_obs*np.sum(np.log(self.noise_variance_)) + np.sum(Xstd));

            self.loglike_ = [log_like]
            self.n_iter_ = 0
            self.finalRatioChange_ = np.nan
            self.finalDiffChange_ = np.nan
            return self


        self.mean_ = np.mean(X, axis=0)
        X -= self.mean_

        # some constant terms
        nsqrt = sqrt(n_samples)
        llconst = -n_features/2 * log(2. * np.pi)
        var = np.var(X, axis=0)
        covX = np.cov(X, rowvar=False, ddof=0)

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

        # we'll modify svd outputs to return unexplained variance
        # to allow for unified computation of loglikelihood
        # if self.svd_method == 'lapack':
        #     def my_svd(X):
        #         _, s, V = linalg.svd(X, full_matrices=False)
        #         return (s[:n_components], V[:n_components],
        #                 squared_norm(s[n_components:]))
        # elif self.svd_method == 'randomized':
        #     random_state = check_random_state(self.random_state)

        #     def my_svd(X):
        #         _, s, V = randomized_svd(X, n_components,
        #                                  random_state=random_state,
        #                                  n_iter=self.iterated_power)
        #         return s, V, squared_norm(X) - squared_norm(s)
        # else:
        #     raise ValueError('SVD method %s is not supported. Please consider'
        #                      ' the documentation' % self.svd_method)

        if np.linalg.matrix_rank(covX) == n_features:
            scale = np.exp(2*np.sum(np.log(np.diag(np.linalg.cholesky(covX))))/n_features)
        else:
            # unlike Matlab, np's cholesky below fails if you're not full
            # rank... kind of correctly, honestly
            # At least I remember it doing so... let's try again
            warnings.warn('FactorAnalysis data matrix is not full rank')
            r = np.linalg.matrix_rank(covX)
            e = np.sort(np.linalg.eig(covX)[0])[::-1]
            scale = sp.stats.mstats.gmean(e[:r])
#            raise Exception("FA:NumObs", "Not enough observations! Rank covariance mat = " + str(np.linalg.matrix_rank(covX)) + ". Num features = " + str(n_features))

        L = np.random.randn(n_features, n_components)*np.sqrt(scale/n_components)
        I = np.eye(n_components)
        psi = np.diag(covX)
        for i in range(self.max_iter):
            invPsi = np.diag(1/psi)
            iPsiL = invPsi @ L
            MM = invPsi - iPsiL @ np.linalg.inv(I + L.T @ iPsiL) @ iPsiL.T

            beta = L.T @ MM

            cX_beta = covX @ beta.T
            EZZ = np.eye(n_components) - beta @ L + beta @ cX_beta

            ldM = np.sum(np.log(np.diag(np.linalg.cholesky(MM))))

            
            ll_new = n_samples*llconst + n_samples*ldM - 0.5*n_samples*np.sum(np.sum(MM * covX))

            loglike.append(ll_new)

            # M-step
            L = cX_beta @ np.linalg.inv(EZZ)
            psi = np.diag(covX) - np.sum(cX_beta * L, axis=1)

            if i <= 2:
                ll_base = ll_new
            elif ll_new < old_ll:
                print("VIOLATION")
            elif ((ll_new - ll_base) < (1+self.tol)*(old_ll - ll_base)):
                break

            old_ll = ll_new
        else:
            warnings.warn('FactorAnalysis did not converge.' +
                          ' You might want' +
                          ' to increase the number of iterations.',
                          ConvergenceWarning)

        self.components_ = L
        self.noise_variance_ = psi
        self.loglike_ = loglike
        self.n_iter_ = i + 1
        self.finalRatioChange_ = (ll_new-ll_base)/(old_ll-ll_base) - 1;
        self.finalDiffChange_ = (ll_new-old_ll);
        precision = self.get_precision()
        self.beta = L.T @ precision
        return self

    def transform(self, X):
        """Apply dimensionality reduction to X using the model.

        Compute the expected mean of the latent variables.
        See Barber, 21.2.33 (or Bishop, 12.66).

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

        X = check_array(X)

        X_transformed = X - self.mean_

        precision = self.get_precision()
        L = self.components_
        beta = L.T @ precision
        X_new = X_transformed @ beta.T

        Ih = np.eye(len(self.components_))

        # this was what the code used to be... which after thorough checking I
        # agree is identical to the above code, but takes another matrix
        # expansion approach (and originally I didn't understand it... which is
        # why I commented it out)
#        Wpsi = self.components_ / self.noise_variance_
#        cov_z = linalg.inv(Ih + np.dot(Wpsi, self.components_.T))
#        tmp = np.dot(X_transformed, Wpsi.T)
#        X_transformed = np.dot(tmp, cov_z)

        return X_new

    def get_covariance(self):
        """Compute data covariance with the FactorAnalysis model.

        ``cov = components_.T * components_ + diag(noise_variance)``

        Returns
        -------
        cov : array, shape (n_features, n_features)
            Estimated covariance of data.
        """
        check_is_fitted(self, 'components_')

        cov = np.dot(self.components_, self.components_.T)
        cov.flat[::len(cov) + 1] += self.noise_variance_  # modify diag inplace
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
            return np.diag(1. / self.noise_variance_)
        if self.n_components == n_features:
            return linalg.inv(self.get_covariance())

        psi = self.noise_variance_
        L = self.components_
        invPsi = np.diag(1/psi)

        iPsiL = invPsi @ L
        precision = invPsi - iPsiL @ np.linalg.inv(np.eye(self.n_components) + L.T @ iPsiL) @ iPsiL.T

        # Get precision using matrix inversion lemma
        # components_ = self.components_
        # precision = np.dot(components_ / self.noise_variance_, components_.T)
        # precision.flat[::len(precision) + 1] += 1.
        # precision = np.dot(components_.T,
        #                    np.dot(linalg.inv(precision), components_))
        # precision /= self.noise_variance_[:, np.newaxis]
        # precision /= -self.noise_variance_[np.newaxis, :]
        # precision.flat[::len(precision) + 1] += 1. / self.noise_variance_
        return precision

    def score_samples(self, X):
        """Compute the log-likelihood of each sample

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            The data

        Returns
        -------
        ll : array, shape (n_samples,)
            Log-likelihood of each sample under the current model
        """
        check_is_fitted(self, 'components_')

        Xr = X - self.mean_
        n_features = X.shape[1]
        n_obs = X.shape[0]

        if self.fit_style_ == "independent Gaussian":
            Xstd = Xr**2/self.noise_variance_
            log_like = -0.5 * (n_obs*n_features*np.log(2*np.pi) + n_obs*np.sum(np.log(self.noise_variance_)) + np.sum(Xstd));
            return log_like

        precision = self.get_precision()

        const = -n_features/2*np.log(2*np.pi)

        XrXr = Xr.T @ Xr

        L = self.components_
        beta = L.T @ precision
        log_like =  n_obs*const + 0.5*n_obs*fast_logdet(precision) - 0.5*np.sum(np.sum(precision * XrXr))       
        # log_like = -.5 * (Xr * (np.dot(Xr, precision))).sum(axis=1)
        # log_like -= .5 * (n_features * log(2. * np.pi)
        #                   - fast_logdet(precision))
        return log_like

    def score(self, X, y=None):
        """Compute the average log-likelihood of the samples

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            The data

        y : Ignored

        Returns
        -------
        ll : float
            Average log-likelihood of the samples under the current model
        """
        return np.mean(self.score_samples(X))
