import numpy as np
import pytest
from sklearn.tree import ExtraTreeClassifier, DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from deep_forest.tree_embedder import (
    embed_with_tree,
    TreeEmbedder,
    ForestEmbedder,
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


def test_embed_with_tree_path(fitted_clf, data):
    Xt = embed_with_tree(fitted_clf, data[0], method="path")
    assert Xt.shape == (data[0].shape[0], fitted_clf.max_depth)
    assert set(np.unique(Xt)) == {0, 1}


@pytest.mark.parametrize(
    "node_weights",
    [None, "neg_log_frac", "log_node_size"],
)
def test_embed_with_tree_node_weights(fitted_clf, data, node_weights):
    Xt = embed_with_tree(
        fitted_clf,
        data[0],
        method="all_nodes",
        node_weights=node_weights,
    )
    n_internal_nodes = fitted_clf.tree_.node_count - fitted_clf.tree_.n_leaves
    assert Xt.shape == (data[0].shape[0], n_internal_nodes)
    if node_weights is None:
        assert set(np.unique(Xt)) == {0, 1}


@pytest.mark.parametrize("max_node_size", [0.5, 0.7, 70])
def test_embed_with_tree_max_node_size(fitted_clf, data, max_node_size):
    Xt = embed_with_tree(
        fitted_clf,
        data[0],
        method="all_nodes",
        max_node_size=max_node_size,
        node_weights=lambda x: x,  # Set the output to the node size
    )
    if isinstance(max_node_size, float):
        max_node_size = np.ceil(max_node_size * data[0].shape[0])

    assert np.all(Xt <= max_node_size)  # Outputs are the node sizes

    internal_node_size = fitted_clf.tree_.n_node_samples[
        fitted_clf.tree_.children_left != fitted_clf.tree_.children_right
    ]
    n_outputs = np.sum(internal_node_size <= max_node_size)

    assert Xt.shape == (data[0].shape[0], n_outputs)


@pytest.mark.parametrize(
    "embedder",
    [
        TreeEmbedder(ExtraTreeClassifier()),
        ForestEmbedder(ExtraTreesClassifier(n_estimators=10)),
    ],
)
def test_embedder_classes(data, embedder):
    X, y = data
    embedder.fit(X, y)
    embedder.transform(X)
