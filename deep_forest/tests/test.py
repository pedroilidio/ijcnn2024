import pytest
from imblearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
from sklearn.datasets import load_iris
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import ExtraTreeClassifier, DecisionTreeClassifier
from sklearn.pipeline import FeatureUnion
from deep_forest.tree_embedder import (
    TreeEmbedder,
    ForestEmbedder,
    ClassifierTransformer,
)
from deep_forest.cascade import Cascade


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

    # Fit the cascade model
    cascade_model.fit(X_train, y_train)

    # Make predictions
    y_pred = cascade_model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)

    # Check accuracy is a valid score
    assert accuracy > 0.5 and accuracy <= 1.0


def test_embedder(iris_data):
    X, y = iris_data
    encoder = TreeEmbedder(ExtraTreeClassifier())
    encoder.fit(X, y)
    encoder.transform(X)
