import numpy as np
from sklearn.base import (
    BaseEstimator,
    TransformerMixin,
    MetaEstimatorMixin,
    clone,
)
from sklearn.ensemble._forest import BaseForest
from sklearn.tree._classes import BaseDecisionTree, DTYPE
from sklearn.tree._tree import Tree
from sklearn.utils.validation import check_is_fitted
from sklearn.utils._param_validation import (
    validate_params,
    StrOptions,
)

__all__ = [
    "embed_with_tree",
]


@validate_params(
    {
        "tree": [BaseDecisionTree],
        "X": ["array-like"],
        "method": [StrOptions({"all_nodes", "path"})],
    }
)
def embed_with_tree(tree_estimator, X, method="all_nodes"):
    tree = tree_estimator.tree_

    if method == "all_nodes":
        # Selects data corresponding to internal nodes, excluding leaves
        mask = tree.children_left != tree.children_right
        return (X[:, tree.feature[mask]] > tree.threshold[mask]).astype(DTYPE)

    if method == "path":
        result = np.zeros((X.shape[0], tree.max_depth), dtype=DTYPE)
        path = tree_estimator.decision_path(X).toarray().astype(bool)
        node_idx = np.arange(tree.node_count)

        for i, mask in enumerate(path):
            is_right_child = tree.children_right[mask][:-1] == node_idx[mask][1:]
            result[i, : len(is_right_child)] = is_right_child

        return result

    raise ValueError


class TreeEmbedder(
    BaseEstimator,
    TransformerMixin,
    MetaEstimatorMixin,
):
    _parameter_constraints = {
        "estimator": [BaseDecisionTree],
        "method": [StrOptions({"all_nodes", "path"})],
    }

    def __init__(self, estimator, method="all_nodes"):
        self.estimator = estimator
        self.method = method

    def fit(self, X, y, **fit_params):
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, y)
        return self

    def transform(self, X):
        check_is_fitted(self)
        return embed_with_tree(
            tree_estimator=self.estimator_,
            X=X,
            method=self.method,
        )


class ForestEmbedder(
    BaseEstimator,
    TransformerMixin,
    MetaEstimatorMixin,
):
    _parameter_constraints = {
        "estimator": [BaseForest],
        "method": [StrOptions({"all_nodes", "path"})],
    }

    def __init__(self, estimator, method="all_nodes"):
        self.estimator = estimator
        self.method = method

    def fit(self, X, y):
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, y)
        return self

    def transform(self, X):
        check_is_fitted(self)
        return np.hstack(
            [
                embed_with_tree(tree, X, method=self.method)
                for tree in self.estimator_.estimators_
            ]
        )


