from numbers import Real, Integral
import numpy as np
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
    "method": [StrOptions({"all_nodes", "path"})],
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
    method="all_nodes",
    node_weights=None,
    max_node_size=1.0,
):
    """Use a decision tree to create data representations.

    Parameters
    ----------
    tree_estimator : BaseDecisionTree
        The decision tree estimator used to embed the data.
    X : array-like
        The input data to embed.
    method : str, optional (default="all_nodes")
        The embedding method to use. Valid options are "all_nodes" and "path".
        - "all_nodes": Binary output indicating whether the sample passes the
          test of each internal node, optionally weighted by the node's size.
        - "path": Binary output indicating whether the sample takes the left (0)
          or right (1) path on each level of the tree.
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
    Xt : array-like
        The embedded data. If method="all_nodes", the columns correspond to the
        internal nodes of the tree, excluding the leaves. If method="path", the
        columns correspond to the direction (0=left, 1=right) took on each tree
        level on the path from the root to the leaf.
    """
    tree = tree_estimator.tree_
    if tree.max_depth <= 1:
        raise ValueError(
            "The tree has a single node. " "It cannot be used to embed the data."
        )

    if method == "all_nodes":
        # Selects data corresponding to internal nodes, excluding leaves
        mask = tree.children_left != tree.children_right
        node_size = tree.weighted_n_node_samples[mask]

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

    if method == "path":
        if node_weights is not None:
            raise ValueError(
                "'node_weights' is not supported for method='path'.",
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
        return np.hstack(
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
