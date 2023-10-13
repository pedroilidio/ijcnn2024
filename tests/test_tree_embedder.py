import copy

import numpy as np
import scipy.sparse
from sklearn.tree import (
    ExtraTreeClassifier,
    ExtraTreeRegressor,
    DecisionTreeClassifier,
)
from sklearn.ensemble import ExtraTreesClassifier
import pytest

from deep_forest.tree_embedder import (
    embed_with_tree,
    TreeEmbedder,
    ForestEmbedder,
    _chi2_per_node,
)


@pytest.fixture
def random_state():
    return np.random.RandomState(0)


@pytest.fixture
def data(random_state):
    return (
        random_state.rand(100, 10),  # X
        (random_state.rand(100, 3) > 0.5).astype(int),  # y
    )


@pytest.fixture
def fitted_clf(random_state, data):
    clf = DecisionTreeClassifier(max_depth=4, random_state=random_state)
    return clf.fit(*data)


def test_embed_with_path(fitted_clf, data):
    Xt = embed_with_tree(fitted_clf, data[0], method="path")
    assert isinstance(Xt, scipy.sparse.csr_matrix)
    assert Xt.shape == (data[0].shape[0], fitted_clf.tree_.node_count)
    # Assert is binary (sparse matrix)
    assert set(Xt.data) == {1}
    assert Xt.sum(axis=1).max() <= fitted_clf.tree_.max_depth + 1


def test_embed_with_path_node_weights(fitted_clf, data):
    Xt = embed_with_tree(
        fitted_clf,
        data[0],
        method="path",
        node_weights=lambda x: x,  # Assing node weight equal to node size
    )
    tree = fitted_clf.tree_
    manual_result = (
        tree.decision_path(data[0].astype('float32')).toarray()
        * tree.weighted_n_node_samples
    )
    assert isinstance(Xt, scipy.sparse.csr_matrix)
    assert np.all(Xt == manual_result)


def test_embed_with_dense_path(fitted_clf, data):
    Xt = embed_with_tree(fitted_clf, data[0], method="dense_path")
    assert Xt.shape == (data[0].shape[0], fitted_clf.tree_.max_depth)
    assert set(np.unique(Xt)) == {0, 1}


@pytest.mark.parametrize("node_weights", [None, "neg_log_frac", "log_node_size"])
@pytest.mark.parametrize("method", ["all_nodes", "path"])
def test_embed_with_tree_node_weights(fitted_clf, data, node_weights, method):
    Xt = embed_with_tree(
        fitted_clf,
        data[0],
        method=method,
        node_weights=node_weights,
    )

    n_nodes = fitted_clf.tree_.node_count
    if method == "all_nodes":
        n_nodes -= fitted_clf.tree_.n_leaves
        
    assert Xt.shape == (data[0].shape[0], n_nodes)

    if node_weights is None:
        if isinstance(Xt, scipy.sparse.csr_matrix):
            assert set(Xt.data) == {1}
        else:  # np.ndarray
            assert set(np.unique(Xt)) == {0, 1}


@pytest.mark.parametrize("max_node_size", [0.5, 0.7, 70])
@pytest.mark.parametrize("method", ["all_nodes", "path"])
def test_embed_with_tree_max_node_size(fitted_clf, data, max_node_size, method):
    Xt = embed_with_tree(
        fitted_clf,
        data[0],
        method=method,
        max_node_size=max_node_size,
        node_weights=lambda x: x,  # Set the output to the node size
    )
    if isinstance(max_node_size, float):
        max_node_size = np.ceil(max_node_size * data[0].shape[0])

    node_size = fitted_clf.tree_.n_node_samples
    
    if method == "all_nodes":
        node_size = node_size[
            fitted_clf.tree_.children_left != fitted_clf.tree_.children_right
        ]

    n_outputs = np.sum(node_size <= max_node_size)
    assert Xt.shape == (data[0].shape[0], n_outputs)

    if isinstance(Xt, scipy.sparse.csr_matrix):
        Xt = Xt.toarray()
    assert np.all(Xt <= max_node_size)  # Outputs are the node sizes



@pytest.mark.parametrize(
    "embedder",
    [
        TreeEmbedder(ExtraTreeClassifier()),
        ForestEmbedder(ExtraTreesClassifier(n_estimators=10)),
        ForestEmbedder(ExtraTreesClassifier(n_estimators=10)),
    ],
)
@pytest.mark.parametrize("method", ["all_nodes", "path", "dense_path"])
def test_embedder_classes(data, embedder, method):
    X, y = data
    embedder.set_params(method=method)
    embedder.fit(X, y)
    embedder.transform(X)


@pytest.mark.parametrize("correction", [True, False], ids=["yates", "no_yates"])
def test_chi2(data, correction):
    embedder = TreeEmbedder(ExtraTreeClassifier()).fit(*data)
    tree = embedder.estimator_.tree_
    total_counts = tree.value[0]
    
    chi2 = []
    for node in tree.value[1:]:  # Skip the root node
        label_contingency_tables = [
            [counts, label_total_counts - counts] 
            for counts, label_total_counts in zip(node, total_counts)
        ]
        label_chi2 = [
            scipy.stats.chi2_contingency(table, correction=correction)[0]
            for table in label_contingency_tables
        ]
        chi2.append(label_chi2)
    
    chi2_broadcast = _chi2_per_node(tree, yates_correction=correction)[1:]
    assert np.allclose(chi2, chi2_broadcast)


@pytest.mark.parametrize("correction", [True, False], ids=["yates", "no_yates"])
def test_chi2_regressor(data, correction):
    embedder = TreeEmbedder(ExtraTreeRegressor()).fit(*data)
    chi2 = _chi2_per_node(embedder.estimator_.tree_, yates_correction=correction)
    assert chi2.shape == (embedder.estimator_.tree_.node_count, data[1].shape[1])