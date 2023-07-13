import pytest
import numpy as np
from sklearn.base import clone
from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score
from deep_forest.weak_labels import WeakLabelImputer


@pytest.fixture
def random_state():
    return np.random.RandomState(42)


@pytest.fixture(params=[1, 10])
def n_outputs(request):
    return request.param


@pytest.fixture
def n_classes():
    return 2  # OOB is not supported for multi-class


@pytest.fixture
def drop():
    """Fraction of labels to drop."""
    return 0.1


@pytest.fixture
def prior(random_state, n_classes):
    p = random_state.uniform(size=n_classes)
    return p / p.sum()


@pytest.fixture
def binary_classification_data(n_outputs, random_state, n_classes, prior, drop):
    # Generate multi-output binary classification data
    X, y = [], []
    for _ in range(n_outputs):
        X_, y_ = make_classification(
            n_samples=10_000,
            n_informative=n_classes * 2,
            n_redundant=1,
            n_classes=n_classes,
            weights=prior,
            random_state=random_state,
        )
        X.append(X_)
        y.append(y_.reshape(-1, 1))

    X, y = np.hstack(X), np.hstack(y)
    to_drop = random_state.choice([True, False], size=y.shape, p=[drop, 1 - drop])
    y[to_drop] = 0

    return X, y


@pytest.mark.parametrize("weight_proba", [False, True])
@pytest.mark.parametrize("use_oob", [False, True])
@pytest.mark.parametrize("threshold", [0.5, 0.8])
def test_imputer(
    weight_proba,
    use_oob,
    binary_classification_data,
    random_state,
    threshold,
):
    X, y = binary_classification_data

    # Create an instance of the imputer
    estimator = ExtraTreesClassifier(
        oob_score=use_oob,
        random_state=random_state,
        bootstrap=True,
        max_samples=0.9,
        min_samples_leaf=5,
    )
    imputer = WeakLabelImputer(
        estimator=clone(estimator),  # Preserve random state
        threshold=threshold,
        weight_proba=weight_proba,
        use_oob_proba=use_oob,
    )

    # Fit and resample the data
    estimator = estimator.fit(X, y)
    X_resampled, y_resampled = imputer.fit_resample(X, y)
    y_pred = estimator.predict(X)

    # Assert the shape of the resampled data
    assert X_resampled.shape[0] == y_resampled.shape[0]

    if use_oob:
        proba = estimator.oob_decision_function_
    else:
        proba = estimator.predict_proba(X)

    if y.ndim == 1:
        y = y.reshape(-1, 1)
    if y_resampled.ndim == 1:
        y_resampled = y_resampled.reshape(-1, 1)
        y_pred = y_pred.reshape(-1, 1)

    assert y.shape == y_resampled.shape
    
    if y.shape[1] == 1:  # If single output
        proba = [proba]
    elif use_oob:  # If multi-output and OOB
        # OOB probas are stored in a 3D array
        proba = list(proba.transpose(2, 0, 1))

    if weight_proba:
        new_probas = []
        classes = [estimator.classes_] if y.shape[1] == 1 else estimator.classes_

        for proba_col, y_, classes_ in zip(proba, y.T, classes):
            # Divide by each class prior (class frequency)
            prior = np.unique(y_, return_counts=True)[1] / y_.shape[0]
            proba_col /= prior[classes_]
            proba_col /= proba_col.sum(axis=1, keepdims=True)
            new_probas.append(proba_col)

        proba = new_probas

    # Assert that the threshold was respected
    max_proba = np.hstack([p.max(axis=1, keepdims=True) for p in proba])
    unchanged = max_proba < threshold
    assert (~unchanged).any()
    assert np.all(y_resampled[unchanged] == y[unchanged])
    assert np.all(y_resampled[~unchanged] == y_pred[~unchanged])