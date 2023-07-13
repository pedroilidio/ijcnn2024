import pytest
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from deep_forest.cascade import Cascade, AlternatingLevel
from deep_forest.tree_embedder import TreeEmbedder
from deep_forest.estimator_adapters import ClassifierTransformer
from sklearn.pipeline import FeatureUnion
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier


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
            ("rf1", ClassifierTransformer(DecisionTreeClassifier())),
            ("rf2", ClassifierTransformer(DecisionTreeClassifier())),
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


def test_alternating_level_estimator(iris_data):
    X, y = iris_data

    level_estimator = AlternatingLevel(
        [
            ("te1", TreeEmbedder(ExtraTreeClassifier(max_features=1))),
            ("te2", TreeEmbedder(ExtraTreeClassifier(max_features=1))),
            ("rf1", ClassifierTransformer(DecisionTreeClassifier())),
            ("rf2", ClassifierTransformer(DecisionTreeClassifier())),
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