import numpy as np
from sklearn.base import (
    BaseEstimator, MetaEstimatorMixin, TransformerMixin, clone,
    ClassifierMixin,
)
from sklearn.utils.metaestimators import _BaseComposition
from sklearn.utils._param_validation import HasMethods
from imblearn.base import BaseSampler


# MultiOutputVotingClassifier is not yet in scikit-learn
# https://github.com/scikit-learn/scikit-learn/pull/23603
class MultiOutputVotingClassifier(_BaseComposition, ClassifierMixin):
    def __init__(self, estimators):
        self.estimators = estimators

    def get_params(self, deep=True):
        return self._get_params("estimators", deep=deep)

    def set_params(self, **params):
        return self._set_params("estimators", **params)

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

        if isinstance(probas[0], list):  # multilabel-indicator
            new_probas = []
            for label_probas in zip(*probas):
                new_probas.append(np.mean(label_probas, axis=0))
            return new_probas

        return np.mean(probas, axis=0)
    
    def predict(self, X):
        probas = self.predict_proba(X)
        if isinstance(probas, list):  # multilabel-indicator
            return np.hstack([
                label_proba.argmax(axis=1).reshape(-1, 1)
                for label_proba in probas
            ])
        return probas.argmax(axis=1).reshape(-1, 1)
    
    @property
    def classes_(self):
        return self.estimators_[0].classes_

    def _more_tags(self):
        return {"multioutput": True}


class ProbaTransformer(
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

    def fit_transform(self, X, y):
        self.estimator_ = clone(self.estimator)

        if hasattr(self.estimator_, "fit_predict_proba"):
            proba = self.estimator_.fit_predict_proba(X, y)
        else:
            proba = self.estimator_.fit(X, y).predict_proba(X)

        if isinstance(proba, list):  # If multioutput
            return np.hstack(proba)
        return proba


class EstimatorAsTransformer(
    BaseEstimator,
    MetaEstimatorMixin,
    TransformerMixin,
):
    _parameter_constraints = {
        "estimator": [
            HasMethods(["fit", "predict"]),
            HasMethods(["fit_predict"]),
        ],
    }

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y):
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, y)
        return self

    def transform(self, X):
        # Use predictions as new features.
        return self.estimator_.predict(X)
    
    def fit_transform(self, X, y):
        self.estimator_ = clone(self.estimator)
        if hasattr(self.estimator_, "fit_predict"):
            return self.estimator_.fit_predict(X, y)
        return self.estimator_.fit(X, y).predict(X)


class EstimatorAsSampler(
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
