import numpy as np
from numbers import Real
from sklearn.utils import check_random_state
from sklearn.utils._param_validation import Interval
from imblearn.base import BaseSampler
from imblearn.pipeline import Pipeline


class PositiveDropper(BaseSampler):
    """Simulate weak labels in binary classification problems.

    A class for testing purposes that simulates missing data in binary
    classification problems.

    Parameters
    ----------
    drop : float
        The proportion of positive samples to drop. Dropping means setting the label of the positive samples to 0.
    random_state : int, RandomState instance or None, optional (default=None)
        Controls the random seed used for sampling.

    Attributes
    ----------
    random_state_ : RandomState
        The random state used for sampling.

    Notes
    -----
    This class is intended for use in testing and simulation of missing data in
    binary classification problems. It drops a proportion of positive samples
    from the input data by setting their labels to 0.
    Only binary classification is supported.
    """

    _parameter_constraints = {
        "drop": [Interval(Real, 0.0, 1.0, closed="left")],
        "random_state": ["random_state"],
    }

    def __init__(self, drop, *, random_state=None):
        self.drop = drop
        self.random_state = random_state

    # FIXME: we are skipping validation since imblearn does not support multilabel
    def _fit_resample(self, X, y):
        pass

    def fit_resample(self, X, y):
        if set(np.unique(y)) != {0.0, 1.0}:
            raise ValueError

        self.random_state_ = check_random_state(self.random_state)

        n_positives = int(y.sum())
        index_positives = np.nonzero(y)

        positives_to_mask = self.random_state_.choice(
            n_positives,
            size=int(self.drop * n_positives),
            replace=False,
        )
        indices_to_mask = (
            index_positives[0][positives_to_mask],
            index_positives[1][positives_to_mask],
        )

        y_sample = y.copy()
        y_sample[indices_to_mask] = 0.0

        return X, y_sample


def wrap_estimator(estimator, drop, random_state=None):
    return Pipeline(
        [
            ("dropper", PositiveDropper(drop, random_state=random_state)),
            ("estimator", estimator),
        ]
    )
