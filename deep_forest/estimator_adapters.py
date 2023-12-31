import numpy as np
from sklearn.base import (
    BaseEstimator,
    MetaEstimatorMixin,
    TransformerMixin,
    ClassifierMixin,
    RegressorMixin,
    clone,
    _fit_context,
)
from sklearn.utils.metaestimators import _BaseComposition
from sklearn.utils._param_validation import HasMethods, StrOptions
from imblearn.base import BaseSampler

from deep_forest.tree_embedder import BaseTreeEmbedder, _hstack


class RegressorAsBinaryClassifier(_BaseComposition, ClassifierMixin):
    def __init__(self, estimator):
        self.estimator = estimator

    def get_params(self, deep=True):
        return self._get_params("estimator", deep=deep)

    def set_params(self, **params):
        return self._set_params("estimator", **params)

    def fit(self, X, y, **params):
        self.estimator_ = clone(self.estimator).fit(X, y, **params)
        return self
    
    def predict_proba(self, X, **params):
        proba = self.estimator_.predict(X, **params)
        # self.n_outputs_ == 1:
        if proba.ndim == 1:
            proba = proba.reshape(-1, 1)
            return np.hstack([1 - proba, proba])
        return [np.vstack([1 - col, col]).T for col in proba.T]
    
    def predict(self, X, **params):
        proba = self.estimator_.predict(X, **params)
        return (proba > 0.5).astype(int)

    @property
    def classes_(self):
        if self.n_outputs_ == 1:
            return np.array([0, 1])
        return [np.array([0, 1])] * self.n_outputs_

    @property
    def n_outputs_(self):
        return self.estimator_.n_outputs_

    @property
    def n_features_in_(self):
        return self.estimator_.n_features_in_

    @property
    def oob_decision_function_(self):
        proba = self.estimator_.oob_prediction_

        if proba.ndim == 1:
            proba = proba.reshape(-1, 1)
            return np.hstack([1 - proba, proba])

        # (n_samples, n_outputs, n_classes) -> (n_samples, n_classes, n_outputs)
        return np.dstack([1 - proba, proba]).transpose(0, 2, 1)


class MultiOutputVotingRegressor(_BaseComposition, RegressorMixin):
    def __init__(self, estimators):
        self.estimators = estimators

    def get_params(self, deep=True):
        return self._get_params("estimators", deep=deep)

    def set_params(self, **params):
        return self._set_params("estimators", **params)

    def fit(self, X, y, **params):
        self.estimators_ = [
            clone(estimator).fit(X, y, **params)
            for name, estimator in self.estimators
        ]
        return self

    def predict(self, X, **params):
        outputs = [
            estimator.predict(X, **params)
            for estimator in self.estimators_
        ]
        return np.mean(outputs, axis=0)
    
    @property
    def oob_prediction_(self):
        probas = [
            estimator.oob_prediction_
            for estimator in self.estimators_
        ]
        return np.mean(probas, axis=0)

    @property
    def n_outputs_(self):
        return self.estimators_[0].n_outputs_

    def _more_tags(self):
        return {"multioutput": True}


# MultiOutputVotingClassifier is not yet in scikit-learn
# https://github.com/scikit-learn/scikit-learn/pull/23603
class MultiOutputVotingClassifier(_BaseComposition, ClassifierMixin):
    def __init__(self, estimators):
        self.estimators = estimators

    def get_params(self, deep=True):
        return self._get_params("estimators", deep=deep)

    def set_params(self, **params):
        return self._set_params("estimators", **params)

    def fit(self, X, y, **params):
        self.estimators_ = [
            clone(estimator).fit(X, y, **params)
            for name, estimator in self.estimators
        ]
        return self

    def predict_proba(self, X, **params):
        probas = [
            estimator.predict_proba(X, **params)
            for estimator in self.estimators_
        ]

        if isinstance(probas[0], list):  # multilabel-indicator
            new_probas = []
            for label_probas in zip(*probas):
                new_probas.append(np.mean(label_probas, axis=0))
            return new_probas

        return np.mean(probas, axis=0)
    
    def predict(self, X, **params):
        probas = self.predict_proba(X, **params)
        if isinstance(probas, list):  # multilabel-indicator
            return np.hstack([
                label_proba.argmax(axis=1).reshape(-1, 1)
                for label_proba in probas
            ])
        return probas.argmax(axis=1).reshape(-1, 1)
    
    # def decision_function(self, X):
    #     return self.predict_proba(X)
    
    @property
    def classes_(self):
        return self.estimators_[0].classes_

    @property
    def n_outputs_(self):
        return self.estimators_[0].n_outputs_

    @property
    def n_features_in_(self):
        return self.estimators_[0].n_features_in_

    @property
    def oob_decision_function_(self):
        probas = [
            estimator.oob_decision_function_
            for estimator in self.estimators_
        ]
        return np.mean(probas, axis=0)

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

    @_fit_context(prefer_skip_nested_validation=False)
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

    @_fit_context(prefer_skip_nested_validation=False)
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

    @_fit_context(prefer_skip_nested_validation=False)
    def fit(self, X, y):
        self._validate_params()
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, y)
        return self

    def transform(self, X):
        # Use predictions as new features.
        return self.estimator_.predict(X)
    
    @_fit_context(prefer_skip_nested_validation=False)
    def fit_transform(self, X, y):
        self._validate_params()
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
    @_fit_context(prefer_skip_nested_validation=False)
    def fit_resample(self, X, y):
        self._validate_params()
        estimator = clone(self.estimator)
        if hasattr(estimator, "fit_predict"):
            yt = estimator.fit_predict(X, y)
        else:
            yt = estimator.fit(X, y).predict(X)
        return X, yt.reshape(len(X[0]), len(X[1]))


class TreeEmbedderWithOutput(
    BaseEstimator,
    MetaEstimatorMixin,
    TransformerMixin,
):
    _parameter_constraints = {
        "estimator": [BaseTreeEmbedder],
        "method": [
            StrOptions({
                "predict", "predict_proba", "predict_log_proba", "decision_function"
            })
        ],
        "post_transformer": [HasMethods(["fit", "transform"])],
    }

    def __init__(self, estimator, method="predict", post_transformer=None):
        self.estimator = estimator
        self.method = method
        self.post_transformer = post_transformer

    @_fit_context(prefer_skip_nested_validation=False)
    def fit(self, X, y):
        self._validate_params()
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, y)
        if self.post_transformer is not None:
            self.post_transformer_ = clone(self.post_transformer)
            Xt = self.estimator_.transform(X)
            self.post_transformer_.fit(Xt)
        return self

    def transform(self, X):
        # Use tree predictions as new features, but still add proba.
        Xt = self.estimator_.transform(X)
        if self.post_transformer is not None:
            Xt = self.post_transformer_.transform(Xt)
        return _hstack([
            Xt,
            getattr(self.estimator_.estimator_, self.method)(X),
        ])
    
    @_fit_context(prefer_skip_nested_validation=False)
    def fit_transform(self, X, y):
        self._validate_params()
        self.estimator_ = clone(self.estimator)
        Xt = self.estimator_.fit_transform(X, y)
        if self.post_transformer is not None:
            self.post_transformer_ = clone(self.post_transformer)
            Xt = self.post_transformer_.fit_transform(Xt, y)
        return _hstack([
            Xt,
            getattr(self.estimator_.estimator_, self.method)(X),
        ])

