import pytest
from pathlib import Path
import sys

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.datasets import make_classification
from sklearn.utils import check_random_state
import sklearn.ensemble

sys.path.insert(0, str(Path(__file__).parents[2]))
from scripts.model_building.positive_dropper import PositiveDropper


@pytest.fixture
def data():
    # Create a sample dataset
    return make_classification(n_samples=100, n_features=10, random_state=0)


@pytest.mark.parametrize(
    "random_state",
    [0, 42, check_random_state(1), np.random.RandomState(2)],
    ids=["0", "42", "check", "np"],
)
def test_cloning_and_dropping(data, random_state):
    X, y = data

    dropper = PositiveDropper(drop=0.3, random_state=random_state)

    cloned_dropper = clone(dropper)

    # Fit and transform the original estimator
    X_transformed_1, y_transformed_1 = dropper.fit_resample(X, y)

    # If random_state is an integer, then the cloned estimator should be
    # identical to the original estimator, even if it was already fitted

    if isinstance(random_state, int):
        # Fit and transform the cloned estimator
        X_transformed_2, y_transformed_2 = dropper.fit_resample(X, y)

        # Assert that the transformed data is the same
        assert (X_transformed_1 == X_transformed_2).all()
        assert (y_transformed_1 == y_transformed_2).all()

        cloned_dropper = clone(dropper)

    # Fit and transform the cloned estimator
    X_transformed_2, y_transformed_2 = cloned_dropper.fit_resample(X, y)

    # Assert that the transformed data is the same
    assert (X_transformed_1 == X_transformed_2).all()
    assert (y_transformed_1 == y_transformed_2).all()
