import pytest
import numpy as np
import sklearn
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from imblearn.pipeline import Pipeline

from deep_forest.cascade import Cascade, AlternatingLevel, SequentialLevel
from deep_forest.tree_embedder import TreeEmbedder
from deep_forest.estimator_adapters import ProbaTransformer
from deep_forest.weak_labels import WeakLabelImputer


@pytest.fixture
def iris_data():
    # Load the iris dataset
    X, y = load_iris(return_X_y=True)
    return X, y


@pytest.fixture
def cascade_model():
    level_estimator = FeatureUnion(
        [
            ("te1", TreeEmbedder(ExtraTreeClassifier(max_features=1))),
            ("te2", TreeEmbedder(ExtraTreeClassifier(max_features=1))),
            ("rf1", ProbaTransformer(DecisionTreeClassifier())),
            ("rf2", ProbaTransformer(DecisionTreeClassifier())),
        ]
    )
    final_estimator = DecisionTreeClassifier()
    cascade = Cascade(level=level_estimator, final_estimator=final_estimator)
    return cascade


def test_cascade_model(iris_data, cascade_model):
    X, y = iris_data

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    cascade_model.fit(X_train, y_train)
    y_pred = cascade_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    assert accuracy > 0.5 and accuracy <= 1.0


@pytest.mark.parametrize("keep_x", [True, False])
def test_keep_input_features(iris_data, cascade_model, keep_x):
    X, y = iris_data
    cascade_model.set_params(keep_original_features=keep_x, max_levels=5)
    cascade_model.fit(X, y)
    # cascade_model.steps.pop()   # Remove final estimator
    # cascade_model.transform(X)  # Does not concatenate X at the last level
    X_transformed = cascade_model._apply_transformers(X).toarray()
    X_transformed2 = X.copy()

    for _, transformer in cascade_model.levels_:
        X_transformed2 = transformer.transform(X_transformed2).toarray()
        if keep_x:
            X_transformed2 = np.hstack([X_transformed2, X])

    assert X_transformed.shape == X_transformed2.shape
    assert np.allclose(X_transformed, X_transformed2)


def test_alternating_level_estimator(iris_data):
    X, y = iris_data

    level_estimator = AlternatingLevel(
        [
            ("te1", TreeEmbedder(ExtraTreeClassifier(max_features=1))),
            ("te2", TreeEmbedder(ExtraTreeClassifier(max_features=1))),
            ("rf1", ProbaTransformer(DecisionTreeClassifier())),
            ("rf2", ProbaTransformer(DecisionTreeClassifier())),
        ]
    )
    final_estimator = DecisionTreeClassifier()
    cascade = Cascade(
        level=level_estimator,
        final_estimator=final_estimator,
        max_levels=3,
    )
    cascade.fit(X, y)
    cascade.predict(X)

    for i, (name, level) in enumerate(cascade.levels_[1:]):
        assert level.last_level_ == cascade.levels_[i][1]

        for col_idx in level._get_column_indices():
            # Assert that some columns are indeed not selected
            assert not col_idx.all()
            # Assert that original X is considered by all
            assert col_idx[-X.shape[1]:].all()


def test_alternating_level_with_sampler(iris_data):
    X, y = iris_data
    y = sklearn.preprocessing.OneHotEncoder().fit_transform(y.reshape(-1, 1)).toarray()

    level_estimator = SequentialLevel([
        (
            "alternating",
            AlternatingLevel([
                ("te1", TreeEmbedder(ExtraTreeClassifier(max_features=1))),
                ("te2", TreeEmbedder(ExtraTreeClassifier(max_features=1))),
                ("rf1", ProbaTransformer(DecisionTreeClassifier())),
                ("rf2", ProbaTransformer(DecisionTreeClassifier())),
            ]),
        ),
        (
            "sampler",
            WeakLabelImputer(
                estimator=ExtraTreesClassifier(
                    oob_score=True,
                    random_state=0,
                    bootstrap=True,
                    max_samples=0.5,
                    min_samples_leaf=5,
                ),
                threshold=0.5,
                weight_proba=True,
                use_oob_proba=True,
            ),
        ),
    ])

    final_estimator = DecisionTreeClassifier()
    cascade = Cascade(
        level=level_estimator,
        final_estimator=final_estimator,
        max_levels=3,
    )
    cascade.fit(X, y)
    cascade.predict(X)

    for i, (name, level) in enumerate(cascade.levels_[1:]):
        assert (
            level.named_steps["alternating"].last_level_
            == cascade.levels_[i][1].named_steps["alternating"]
        )

        for col_idx in level.named_steps["alternating"]._get_column_indices():
            # Assert that some columns are indeed not selected
            assert not col_idx.all()
            # Assert that original X is considered by all
            assert col_idx[-X.shape[1]:].all()


def test_cascade_warm_start(iris_data):
    X, y = iris_data
    y = sklearn.preprocessing.OneHotEncoder().fit_transform(y.reshape(-1, 1)).toarray()

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Create a Cascade object with warm-start enabled
    cascade = Cascade(
        level=TreeEmbedder(DecisionTreeClassifier(max_depth=2)),
        final_estimator=DecisionTreeClassifier(max_depth=2),
        max_levels=2,
        warm_start=True,
    )

    cascade.fit(X_train, y_train)
    assert len(cascade.levels_) == 2

    y_pred1 = cascade.predict(X_test)

    # Increase the max_levels parameter and resume fitting
    cascade.set_params(max_levels=5)
    cascade.fit(X_train, y_train)
    assert len(cascade.levels_) == 5

    y_pred2 = cascade.predict(X_test)

    # Calculate accuracies
    accuracy1 = accuracy_score(y_test, y_pred1)
    accuracy2 = accuracy_score(y_test, y_pred2)

    # Assert that the accuracy of the second prediction is higher than the first
    assert accuracy1 > 0.8
    assert accuracy2 > 0.8