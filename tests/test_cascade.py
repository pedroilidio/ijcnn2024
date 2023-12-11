import pytest
import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier

from deep_forest.cascade import Cascade, AlternatingLevel
from deep_forest.tree_embedder import TreeEmbedder
from deep_forest.estimator_adapters import ProbaTransformer


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
    X_transformed = cascade_model._apply_transformers(X)
    X_transformed2 = X.copy()

    for _, transformer in cascade_model.steps[:-1]:
        X_transformed2 = transformer.transform(X_transformed2)
        if keep_x:
            X_transformed2 = np.hstack([X, X_transformed2])

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

    assert (
        cascade.named_steps["level0"].output_indices_
        == cascade.named_steps["level1"].last_output_indices
    )
    assert (
        cascade.named_steps["level1"].output_indices_
        == cascade.named_steps["level2"].last_output_indices
    )

    # Assert that some columns are indeed not selected
    assert all(
        not col_idx.all()
        for col_idx in cascade.named_steps["level1"]._get_column_indices()
    )
    assert all(
        not col_idx.all()
        for col_idx in cascade.named_steps["level2"]._get_column_indices()
    )