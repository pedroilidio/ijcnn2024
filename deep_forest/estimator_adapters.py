import numpy as np
from sklearn.base import (
    BaseEstimator, MetaEstimatorMixin, TransformerMixin, clone,
    ClassifierMixin,
)
from sklearn.utils._param_validation import HasMethods
from imblearn.base import BaseSampler


# MultiOutputVotingClassifier is not yet in scikit-learn
# https://github.com/scikit-learn/scikit-learn/pull/23603
class MultiOutputVotingClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, estimators):
        self.estimators = estimators

    def fit(self, X, y):
        self.estimators_ = [
            clone(estimator).fit(X, y)
            for name, estimator in self.estimators
        ]
        return self

    def predict_proba(self, X):
        probas = [
            estimator.predict_proba(X)
            for estimator in self.estimators_
        ]
        probas = [np.mean(label_probas, axis=0) for label_probas in zip(*probas)]
        return probas
    
    def predict(self, X):
        return np.hstack([
            label_proba.argmax(axis=1).reshape(-1, 1)
            for label_proba in self.predict_proba(X)
        ])

    def _more_tags(self):
        return {"multioutput": True}


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
        proba = self.estimator_.predict_proba(X)
        if isinstance(proba, list):  # If multioutput
            return np.hstack(proba)
        return proba


class RegressorAsSampler(
    BaseSampler,
    MetaEstimatorMixin,
):
    def __init__(self, estimator):
        self.estimator = estimator

    # def _fit_resample(self, X, y):  # FIXME
    def fit_resample(self, X, y):
        estimator = clone(self.estimator)
        if hasattr(estimator, "fit_predict"):
            yt = estimator.fit_predict(X, y)
        else:
            yt = estimator.fit(X, y).predict(X)
        return X, yt.reshape(len(X[0]), len(X[1]))
