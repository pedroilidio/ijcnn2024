import warnings
from numbers import Real, Integral

import numpy as np
import scipy.sparse as sp
from sklearn.utils import check_random_state
from sklearn.utils._param_validation import Interval
from sklearn.decomposition._truncated_svd import (
    TruncatedSVD,
    svds,
    _init_arpack_v0,
    safe_sparse_dot,
    mean_variance_axis,
    svd_flip,
)


class LOBPCGTruncatedSVD(TruncatedSVD):
    """Truncated SVD using the LOBPCG solver.
    """
    _parameter_constraints ={
        **TruncatedSVD._parameter_constraints,
        "n_components": [
            Interval(Real, 0, 1, closed="right"),
            Interval(Integral, 1, None, closed="left"),
        ],
        "max_components": [
            Interval(Integral, 1, None, closed="left"),
        ],
        "verbose": ["verbose"],
    }
    del _parameter_constraints["algorithm"]
    del _parameter_constraints["n_iter"]
    del _parameter_constraints["n_oversamples"]
    del _parameter_constraints["power_iteration_normalizer"]

    def __init__(
        self,
        *,
        n_components=1.0,
        max_components=None,
        tol=0.0,
        verbose=0,
        random_state=None,
    ):
        self.n_components = n_components
        self.max_components = max_components
        self.tol = tol
        self.verbose = verbose
        self.random_state = random_state

    def fit_transform(self, X, y=None):
        """Fit model to X and perform dimensionality reduction on X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training data.

        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_components)
            Reduced version of X. This will always be a dense array.
        """
        if isinstance(self.n_components, float):
            n_components = int(np.ceil(min(X.shape) * self.n_components))
            if self.max_components is not None and n_components > self.max_components:
                warnings.warn(
                    f"{self.n_components=} resulted in {n_components=} for"
                    f" {X.shape=}. {self.max_components=} will be used instead."
                )
                n_components = self.max_components
        else:
            n_components = min(self.n_components, *X.shape)
            if n_components != self.n_components:
                warnings.warn(
                    f"{self.n_components=} is too large for matrix of"
                    f" {X.shape=}. Only {n_components} components will be kept."
                )

        X = self._validate_data(X, accept_sparse=["csr", "csc"], ensure_min_features=2)
        random_state = check_random_state(self.random_state)

        # if self.algorithm == "arpack":
        v0 = _init_arpack_v0(min(X.shape), random_state)  # TODO: use?
        U, Sigma, VT = svds(X, k=n_components, tol=self.tol, v0=v0, solver="lobpcg")
        # svds doesn't abide by scipy.linalg.svd/randomized_svd
        # conventions, so reverse its outputs.
        Sigma = Sigma[::-1]
        U, VT = svd_flip(U[:, ::-1], VT[::-1])

        self.components_ = VT

        # As a result of the SVD approximation error on X ~ U @ Sigma @ V.T,
        # X @ V is not the same as U @ Sigma
        if self.tol > 0:
            X_transformed = safe_sparse_dot(X, self.components_.T)
        else:
            X_transformed = U * Sigma

        # Calculate explained variance & explained variance ratio
        self.explained_variance_ = exp_var = np.var(X_transformed, axis=0)
        if sp.issparse(X):
            _, full_var = mean_variance_axis(X, axis=0)
            full_var = full_var.sum()
        else:
            full_var = np.var(X, axis=0).sum()
        self.explained_variance_ratio_ = exp_var / full_var
        self.singular_values_ = Sigma  # Store the singular values.

        return X_transformed
 