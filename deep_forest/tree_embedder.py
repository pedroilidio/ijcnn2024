from numbers import Real, Integral
import numpy as np
import scipy.sparse
from sklearn.base import (
    BaseEstimator,
    TransformerMixin,
    MetaEstimatorMixin,
    clone,
)
from sklearn.ensemble._forest import BaseForest
from sklearn.tree._classes import BaseDecisionTree, DTYPE
from sklearn.utils.validation import check_is_fitted
from sklearn.utils._param_validation import (
    validate_params,
    StrOptions,
    Interval,
)

__all__ = [
    "embed_with_tree",
    "TreeEmbedder",
    "ForestEmbedder",
]

_COMMON_PARAMS = {
    "method": [StrOptions({"all_nodes", "path", "dense_path"})],
    "node_weights": [
        StrOptions({"neg_log_frac", "log_node_size", "norm_log_node_size"}),
        callable,
        None,
    ],
    "max_node_size": [
        Interval(Real, 0, 1, closed="right"),
        Interval(Integral, 1, None, closed="left"),
    ],
}


def _hstack(Xs):
    if any(scipy.sparse.issparse(f) for f in Xs):
        return scipy.sparse.hstack(Xs).tocsr()
    return np.hstack(Xs)


@validate_params(
    {
        "node_size": ["array-like"],
        "node_weights": _COMMON_PARAMS["node_weights"],
    },
)
def _get_node_weights(node_size, node_weights):
    """Calculate node weights to be used in the tree embedding."""
    if callable(node_weights):
        return node_weights(node_size)
    if node_weights == "neg_log_frac":
        # node_size[0] is the size of the root node
        return -np.log(node_size / node_size[0])
    if node_weights == "log_node_size":
        return 1 / (np.log(node_size) + 1)
    if node_weights == "norm_log_node_size":
        # 1/log(n) normalized to be between 1 and 0
        log_size = np.log(node_size)
        return (1 - log_size / np.log(node_size[0])) / (1 + log_size)
    if node_weights is None:
        return np.ones(node_size.shape)
    raise RuntimeError


@validate_params(
    {
        "tree": [BaseDecisionTree],
        "X": ["array-like"],
        **_COMMON_PARAMS,
    }
)
def embed_with_tree(
    tree_estimator,
    X,
    method="path",
    node_weights=None,
    max_node_size=1.0,
) -> np.ndarray | scipy.sparse.csr_matrix:
    """Use a decision tree to create data representations.

    Parameters
    ----------
    tree_estimator : BaseDecisionTree
        The decision tree estimator used to embed the data.
    X : array-like
        The input data to embed.
    method : str, optional (default="path")
        The embedding method to use. Valid options are "all_nodes" and "path".
        - "path": Binary output indicating whether the sample passed through
          each node of the tree. Yields a sparse matrix with one feature per
          node (shape=(n_samples, n_nodes)).
        - "dense_path": Binary output indicating whether the sample takes the
          left (0) or right (1) path on each level of the tree. Yields one
          feature per level (shape=(n_samples, n_levels)).
        - "all_nodes": Binary output indicating whether the sample passes each
          test, for all internal nodes (shape=(n_samples, n_nodes)).
        
        .. note::

            The "all_nodes" method requires considerable memory for large trees.

    node_weights : str, callable, or None, optional (default=None)
        The method used to weight the nodes. Valid options are:
        - "log_node_size": The node weights are proportional to the inverse
            logarithm of the number of training samples in each node.
        - "norm_log_node_size": The node weights are proportional to the
            1 / log(node_size), but normalized to the range [0, 1].
        - "neg_log_frac": The node weights are proportional to the negative
            logarithm of the fraction of training samples in each node.
        - callable: A callable that takes a 1D array of node sizes as input and
            returns a 1D array of the same shape with the node weights.
        - None: No weighting is applied.
    max_node_size : float or int, optional (default=1.0)
        The maximum number of training samples passing through a node for it to
        be considered in the embedding. If a float between 0 and 1, it
        represents the maximum fraction of the total number of samples. Nodes
        that exceed this size are not included in the output.

    Returns
    -------
    Xt : np.ndarray or scipy.sparse.csr_matrix
        The embedded data. See the description of the `method` parameter for
        details.
    """
    tree = tree_estimator.tree_
    if tree.max_depth <= 1:
        raise ValueError(
            "The tree has a single node. It cannot be used to embed the data."
        )

    if method in ("path", "all_nodes"):
        # Selects data corresponding to internal nodes, excluding leaves
        mask = tree.children_left != tree.children_right
        node_size = tree.weighted_n_node_samples[mask]

        if method == "path":
            # The data is encoded as binary values indicating whether the sample
            # passes through each node
            Xt = tree_estimator.decision_path(X)[:, mask]
            if node_weights is not None:
                # TODO: Scipy is switching to array interface, so that the
                # following will be valid for both method options in the future:
                # Xt *= _get_node_weights(node_size, node_weights)
                Xt = Xt.multiply(_get_node_weights(node_size, node_weights)).tocsr()
        else:  # method == "all_nodes"
            # The data is encoded as binary values indicating whether the sample
            # passes the test of each internal node
            Xt = (X[:, tree.feature[mask]] > tree.threshold[mask]).astype(DTYPE)
            if node_weights is not None:
                Xt *= _get_node_weights(node_size, node_weights)

        if isinstance(max_node_size, float):
            # node_size[0] is the size of the root node
            max_node_size = np.ceil(max_node_size * node_size[0])

        Xt = Xt[:, node_size <= max_node_size]
        return Xt

    if method == "dense_path":
        if node_weights is not None:
            raise ValueError(
                f"'node_weights' is not supported for {method=}.",
            )
        Xt = np.zeros((X.shape[0], tree.max_depth), dtype=DTYPE)
        path = tree_estimator.decision_path(X).toarray().astype(bool)
        node_idx = np.arange(tree.node_count)

        for i, mask in enumerate(path):
            is_right_child = tree.children_right[mask][:-1] == node_idx[mask][1:]
            Xt[i, : len(is_right_child)] = is_right_child

        return Xt

    raise ValueError


class BaseTreeEmbedder(
    BaseEstimator,
    TransformerMixin,
    MetaEstimatorMixin,
):
    _parameter_constraints = {
        "estimator": [BaseEstimator],
        **_COMMON_PARAMS,
    }

    def __init__(
        self,
        estimator,
        method="all_nodes",
        node_weights=None,
        max_node_size=1.0,
    ):
        self.estimator = estimator
        self.method = method
        self.node_weights = node_weights
        self.max_node_size = max_node_size

    def fit(self, X, y, **fit_params):
        self._validate_params()
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, y)
        return self


class TreeEmbedder(BaseTreeEmbedder):
    _parameter_constraints = {
        **BaseTreeEmbedder._parameter_constraints,
        "estimator": [BaseDecisionTree],
    }

    def transform(self, X):
        check_is_fitted(self)
        return embed_with_tree(
            tree_estimator=self.estimator_,
            X=X,
            method=self.method,
            node_weights=self.node_weights,
            max_node_size=self.max_node_size,
        )


class ForestEmbedder(BaseTreeEmbedder):
    _parameter_constraints = {
        **BaseTreeEmbedder._parameter_constraints,
        "estimator": [BaseForest],
    }

    def transform(self, X):
        check_is_fitted(self)
        return _hstack(
            [
                embed_with_tree(
                    tree,
                    X,
                    method=self.method,
                    node_weights=self.node_weights,
                    max_node_size=self.max_node_size,
                )
                for tree in self.estimator_.estimators_
            ]
        )
