import numpy as np
from numbers import Real
from sklearn.base import (
    BaseEstimator,
    MetaEstimatorMixin,
    clone,
    is_classifier,
)
from sklearn.utils._param_validation import Interval, StrOptions
from imblearn.base import BaseSampler


class WeakLabelImputer(BaseSampler, MetaEstimatorMixin):
    _sampling_type = "clean-sampling"
    _parameter_constraints = {
        "estimator": [BaseEstimator],
        "threshold": [Interval(Real, 0, 1, closed="left")],
        "sampling_strategy": [StrOptions({"auto"})],
    }

    def __init__(self, estimator, threshold=0.8, sampling_strategy="auto"):
        self.estimator = estimator
        self.threshold = threshold
        self.sampling_strategy = sampling_strategy

    # FIXME: we are skipping validation since imblearn does not support multilabel
    def _fit_resample(self, X, y):
        return self.fit_resample(X, y)

    def fit_resample(self, X, y):
        if not is_classifier(self.estimator):
            raise ValueError(
                "'estimator' parameter must be a classifier instance. "
                f"Got {self.estimator}.",
            )

        classifier = clone(self.estimator).fit(X, y)

        y_pred = classifier.predict(X)
        proba = classifier.predict_proba(X)

        # Recover true labels for samples with low confidence
        # FIXME: test list of arrays for multilabel
        if isinstance(proba, list):
            mask = np.hstack(
                [np.max(p, axis=-1).reshape(-1, 1) < self.threshold for p in proba]
            )
        elif isinstance(proba, np.ndarray):
            mask = proba.max(axis=-1) < self.threshold
        else:
            raise TypeError

        y_pred[mask] = y[mask]
        return X, y_pred


class PositiveUnlabeledImputer(WeakLabelImputer):
    # FIXME: we are skipping validation since imblearn does not support multilabel
    # def _fit_resample(self, X, y):  (Correct)
    def fit_resample(self, X, y):
        if not is_classifier(self.estimator):
            raise ValueError(
                "'estimator' parameter must be a classifier instance. "
                f"Got {self.estimator}.",
            )

        # Set threshold for each class, based on the proportion of positive samples
        # NOTE: assumes binary classification, and that 0 is the majoritary class
        threshold = (1 - self.threshold) * (1 - y.mean(axis=0))

        classifier = clone(self.estimator).fit(X, y)

        y_pred = classifier.predict(X)
        proba = classifier.predict_proba(X)

        # Recover true labels for samples with low confidence
        # FIXME: test list of arrays for multilabel
        if isinstance(proba, list):
            mask = np.hstack(
                [p[:, 0].reshape(-1, 1) > t for p, t in zip(proba, threshold)]
            )
        elif isinstance(proba, np.ndarray):
            mask = proba[..., 0] > threshold
        else:
            raise TypeError

        # y[~mask] = 1
        y_pred[mask] = y[mask]
        return X, y_pred
