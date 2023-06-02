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

    def _fit_resample(self, X, y):
        if not is_classifier(self.estimator):
            raise ValueError(
                "'estimator' parameter must be a classifier instance. "
                "Got {self.estimator}.",
            )

        classifier = clone(self.estimator).fit(X, y)

        y_pred = classifier.predict(X)
        proba = classifier.predict_proba(X)

        # Recover true labels for samples with low confidence
        mask = proba.max(axis=-1) < self.threshold
        y_pred[mask] = y[mask]

        return X, y_pred
