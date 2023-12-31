import pytest
import numpy as np
import sklearn
import sklearn.datasets
from sklearn.tree import DecisionTreeClassifier
from deep_forest.tice import TIcEImputer

from test_weak_labels import (
    binary_classification_data,
    n_outputs,
    # random_state,
    n_classes,
    prior,
    drop,
)


@pytest.fixture
def random_state():
    return sklearn.utils.check_random_state(0)


@pytest.fixture
def iris_data(random_state):
    # Load the iris dataset
    X, y = sklearn.datasets.load_iris(return_X_y=True)

    idx = random_state.permutation(X.shape[0])
    X, y = X[idx], y[idx]

    X = sklearn.preprocessing.MinMaxScaler().fit_transform(X)
    y = (
        sklearn.preprocessing.OneHotEncoder()
        .fit_transform(y.reshape(-1, 1)).toarray()
    )

    # Drop 40% of positive labels
    positive_indices = np.where(y == 1)[0]
    drop_indices = random_state.choice(
        positive_indices,
        int(0.4 * len(positive_indices)),
        replace=False,
    )
    y[drop_indices] = 0

    return X, y


# def test_tice_imputer_fit(binary_classification_data):
#     X, y = binary_classification_data
#     X = sklearn.preprocessing.MinMaxScaler().fit_transform(X)
def test_tice_imputer_fit(iris_data):
    X, y = iris_data

    # Create a TIcEImputer object
    imputer = TIcEImputer(
        estimator=DecisionTreeClassifier(),
        useMostPromisingOnly=False,
    )

    # Fit the imputer to the data
    imputer.fit(X, y)
    breakpoint()

    # Check that the estimator_ attribute is set
    assert hasattr(imputer, "estimator_")
    assert isinstance(imputer.estimator_, DecisionTreeClassifier)


def test_tice_imputer_estimate_c(iris_data):
    X, y = iris_data

    # Create a TIcEImputer object
    imputer = TIcEImputer(estimator=DecisionTreeClassifier())

    # Estimate c
    c_estimate, c_its_estimates = imputer._estimate_c(X, y)

    # Check that c_estimate and c_its_estimates are arrays
    assert isinstance(c_estimate, np.ndarray)
    assert isinstance(c_its_estimates, np.ndarray)

    # Check that c_estimate is a scalar value
    assert c_estimate.ndim == 0

    # Check that c_its_estimates has the same length as nbIts parameter
    assert len(c_its_estimates) == imputer.nbIts


def test_tice_imputer_fit_transform(iris_data):
    X, y = iris_data

    # Create a TIcEImputer object
    imputer = TIcEImputer(estimator=DecisionTreeClassifier())

    # Fit and transform the data
    X_imputed = imputer.fit_transform(X, y)

    # Check that the imputed data has the same shape as the original data
    assert X_imputed.shape == X.shape


def test_tice_imputer_alpha(iris_data):
    X, y = iris_data

    # Create a TIcEImputer object
    imputer = TIcEImputer(estimator=DecisionTreeClassifier())

    # Fit the imputer to the data
    imputer.fit(X, y)

    # Check that the alpha_ attribute is set
    assert hasattr(imputer, "alpha_")
    assert isinstance(imputer.alpha_, float)
    assert imputer.alpha_ >= 0.0 and imputer.alpha_ <= 1.0