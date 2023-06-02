from sklearn.base import (
    BaseEstimator, MetaEstimatorMixin, TransformerMixin, clone,
)
from sklearn.utils._param_validation import HasMethods
from imblearn.base import BaseSampler


class ClassifierTransformer(
    BaseEstimator,
    MetaEstimatorMixin,
    TransformerMixin,
):
    _parameter_constraints = {
        "estimator": [HasMethods(["fit", "predict_proba"])],
    }

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y):
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, y)
        return self

    def transform(self, X):
        # Use the predicted probabilities as features
        return self.estimator_.predict_proba(X)


class RegressorAsSampler(
    BaseSampler,
    MetaEstimatorMixin,
):
    def __init__(self, estimator):
        self.estimator = estimator

    def _fit_resample(self, X, y):
        estimator = clone(self.estimator)
        if hasattr(estimator, "fit_predict"):
            yt = estimator.fit_predict(X, y)
        else:
            yt = estimator.fit(X, y).predict(X)
        return X, yt.reshape(len(X[0]), len(X[1]))
