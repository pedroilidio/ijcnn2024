from numbers import Real, Integral
import warnings

import numpy as np
import scipy.sparse
import scipy.stats
import sklearn.tree
from sklearn.base import (
    BaseEstimator,
    TransformerMixin,
    MetaEstimatorMixin,
    clone,
    _fit_context,
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
    "max_pvalue": [
        Interval(Real, 0, 1, closed="both"),
    ],
    "max_node_size": [
        Interval(Real, 0, 1, closed="right"),
        Interval(Integral, 1, None, closed="left"),
    ],
}


def _chi2_per_node(
    tree: sklearn.tree._tree.Tree,
    yates_correction=False,
    warn: bool = True,
):
    """Calculate the chi2 statistic for each node in the tree.

    Leverages numpy broadcasting for faster computation.
    Each node is regaded as a contingency table, with the classes
    being the columns and the node selection of labels being the rows.

    Parameters
    ----------
    tree : sklearn.tree._tree.Tree
        The tree to calculate the chi2 statistic for.
    """
    # Total number of negatives, total number of positives
    if tree.value.ndim == 3:  # Classification
        # shape=(n_nodes, n_labels, n_classes)
        class_counts = tree.value
    else:  # Regression  HACK: only works for binary input.
        # shape=(n_nodes, n_outputs) -> (n_nodes, n_labels, n_classes)
        class_counts = np.dstack([
            (1 - tree.value) * tree.weighted_n_node_samples,  # negatives
            tree.value * tree.weighted_n_node_samples,  # positives
        ])

    # shape=(n_labels, n_classes)
    margin_row = class_counts[0]  # Class counts at the root node
    total = tree.weighted_n_node_samples[0]

    # N samples in the node, N samples outside the node
    # shape=(n_nodes, n_groups)
    # n_groups = 2, for inside/outside the node
    margin_col = np.stack([
        tree.weighted_n_node_samples[1:],  # [:1] to skip the root node
        total - tree.weighted_n_node_samples[1:],
    ]).T

    # Expected values under the null hypothesis
    # (1, n_labels, 1, n_classes) * (n_nodes, 1, n_groups, 1) =
    #       = (n_nodes, n_labels, n_groups, n_classes)
    expected = (margin_row[None, :, None, :] * margin_col[:, None, :, None]) / total

    if 0. in expected:
        if warn:
            warnings.warn(
                "It seems that some classes are not observed in the dataset received by"
                " the tree. The following [y_column, class_index] are not found in y:"
                f" {np.argwhere(margin_row == 0).tolist()}"
            )
        expected[expected == 0] = np.finfo(expected.dtype).eps  # Avoid division by zero

    # shape=(n_nodes, n_labels, n_groups, n_classes)
    contingency_tables = np.stack(
        [
            class_counts[1:],  # shape=(n_nodes, n_labels, n_classes)
            margin_row - class_counts[1:],  # same shape
        ],
        axis=-2,  # so that classes is the last axis
    )

    diff = np.abs(contingency_tables - expected)
    if yates_correction:  # Yates' correction for continuity
        diff[diff <= 0.5] = 0.0
        diff[diff > 0.5] -= 0.5

    chi2 = (diff ** 2 / expected).sum(axis=(-1, -2))  # Sum over classes and groups

    # Add a row of zeros for the root node  # TODO: avoid copying
    return np.concatenate([
        np.zeros((1, tree.n_outputs), dtype=chi2.dtype),
        chi2,
    ])


def _hstack(Xs):
    if any(scipy.sparse.issparse(f) for f in Xs):
        return scipy.sparse.hstack(Xs, format='csr')
    return np.hstack(Xs)


@validate_params(
    {
        "node_size": ["array-like"],
        "node_weights": _COMMON_PARAMS["node_weights"],
    },
    prefer_skip_nested_validation=True,
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


# TODO RFC: store a mask or weights for each tree in the embedder transformer,
# avoiding recalculating boolean masks, weights and/or chi2 statistics every in
# every call to transform().
@validate_params(
    {
        "tree": [BaseDecisionTree],
        "X": ["array-like"],
        "warn_no_output": ["boolean"],
        **_COMMON_PARAMS,
    },
    prefer_skip_nested_validation=False,  # Trees are not validated yet
)
def embed_with_tree(
    tree_estimator,
    X,
    method="path",
    node_weights=None,
    max_pvalue=1.0,
    max_node_size=1.0,
    warn_no_output=True,
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
    max_pvalue : float, optional (default=1.0)
        If less than 1, a Chi-squared statistic is calculated for each node,
        considering the node as a contingency table. Nodes whose statistic
        yields a p-value greater than `max_pvalue` are discarded.
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
    original_max_node_size = max_node_size

    if method != "dense_path":  # TODO: move to fit
        mask = np.ones(tree.node_count, dtype=bool)

        if max_pvalue < 1.0:
            chi2 = _chi2_per_node(tree)  # shape=(n_nodes, n_labels)
            chi2 = chi2.max(axis=1)  # take maximum chi2 among labels, shape=(n_nodes,)
            chi2_threshold = scipy.stats.chi2.ppf(max_pvalue, df=1)
            mask &= chi2 > chi2_threshold

        node_size = tree.weighted_n_node_samples

        if isinstance(max_node_size, float):
            # node_size[0] is the size of the root node
            max_node_size = np.ceil(max_node_size * node_size[0])
        
        mask &= node_size <= max_node_size

        if method == "all_nodes":
            mask &= tree.children_left != tree.children_right

        weights = _get_node_weights(node_size[mask], node_weights)

    if method == "path":
        # The data is encoded as binary values indicating whether the sample
        # passes through each node
        Xt = tree_estimator.decision_path(X)[:, mask]

        if node_weights is not None:
            # TODO: Scipy is switching to array interface, so that the
            # following will be valid for both method options in the future:
            # Xt *= _get_node_weights(node_size, node_weights)
            Xt = Xt.multiply(weights).tocsr()

    elif method == "dense_path":
        if node_weights is not None:
            raise ValueError(
                f"'node_weights' is not supported for {method=}.",
            )
        Xt = np.zeros((X.shape[0], tree.max_depth), dtype=DTYPE)
        # TODO: keep sparse
        paths = tree_estimator.decision_path(X).toarray().astype(bool)
        node_idx = np.arange(tree.node_count)

        for i, path in enumerate(paths):
            is_right_child = tree.children_right[path][:-1] == node_idx[path][1:]
            Xt[i, :len(is_right_child)] = is_right_child

    elif method == "all_nodes":
        # Select data corresponding to internal nodes, excluding leaves
        node_size = tree.weighted_n_node_samples[mask]

        # The data is encoded as binary values indicating whether the sample
        # passes the test of each internal node
        Xt = X[:, tree.feature[mask]] > tree.threshold[mask]
        Xt = Xt.astype(DTYPE)

        if node_weights is not None:
            Xt *= weights

        Xt = Xt[:, node_size <= max_node_size]
    else:
        raise ValueError

    if Xt.shape[1] == 0 and warn_no_output:
        warnings.warn(
            f"Tree embedder: no nodes were selected, so no features produced."
            f" {tree.max_depth=}. Consider increasing"
            f" max_node_size={original_max_node_size} or {max_pvalue=}."
        )
    return Xt



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
        method="path",
        node_weights=None,
        max_pvalue=1.0,
        max_node_size=1.0,
    ):
        self.estimator = estimator
        self.method = method
        self.node_weights = node_weights
        self.max_pvalue = max_pvalue
        self.max_node_size = max_node_size

    @_fit_context(
        # self.estimator is not validated yet
        prefer_skip_nested_validation=False,
    )
    def fit(self, X, y, **fit_params):
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, y)
        return self
    
    @property
    def n_features_in_(self):
        return self.estimator_.n_features_in_


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
            max_pvalue=self.max_pvalue,
            max_node_size=self.max_node_size,
        )


class ForestEmbedder(BaseTreeEmbedder):
    _parameter_constraints = {
        **BaseTreeEmbedder._parameter_constraints,
        "estimator": [BaseForest],
    }

    def transform(self, X):
        check_is_fitted(self)
        # Convert to array of objects to avoid copying the data
        embeddings_iter = (
            embed_with_tree(
                tree,
                X,
                method=self.method,
                node_weights=self.node_weights,
                max_pvalue=self.max_pvalue,
                max_node_size=self.max_node_size,
                warn_no_output=False,
            )
            for tree in self.estimator_.estimators_
        )
        Xt = _hstack(np.fromiter(embeddings_iter, dtype='object'))

        if Xt.shape[1] == 0:
            warnings.warn(
                f"Tree embedder: no nodes were selected, so no features produced."
                f" Consider increasing {self.max_node_size=} or {self.max_pvalue=}."
            )
        return Xt
