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
from sklearn import metrics
from sklearn.model_selection import check_cv
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

    # def get_params(self, deep=True):
    #     return self._get_params("estimators", deep=deep)

    # def set_params(self, **params):
    #     return self._set_params("estimators", **params)

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


class UnanimityClassifier(MultiOutputVotingClassifier):
    def __init__(self, estimators, threshold=0.5):
        self.estimators = estimators
        self.threshold = threshold

    def predict_proba(self, X, **params):
        probas = [
            estimator.predict_proba(X, **params)
            for estimator in self.estimators_
        ]

        if isinstance(probas[0], list):  # multilabel-indicator
            new_probas = []
            for label_probas in zip(*probas):
                new_probas.append(np.prod(label_probas, axis=0))  # FIXME: threshold
            return new_probas

        return np.prod(probas, axis=0)

    def predict(self, X, **params):
        probas = [
            estimator.predict_proba(X, **params)
            for estimator in self.estimators_
        ]

        if self.n_outputs_ == 1:
            return np.all(
                [proba[:, 1] >= self.threshold for proba in probas],
                axis=0,
            ).astype(int)
        
        pred = []
        for output_probas in zip(*probas):
            pred.append(
                np.all(
                    [proba[:, 1] >= self.threshold for proba in output_probas],
                    axis=0,
                ).astype(int)
            )
        return np.vstack(pred).T

    @property
    def oob_decision_function_(self):
        probas = [
            estimator.oob_decision_function_
            for estimator in self.estimators_
        ]
        return np.prod(probas, axis=0)


class CVClassifier(MultiOutputVotingClassifier):
    def __init__(self, estimator, cv=None, groups=None, oob_score=False):
        self.estimator = estimator
        self.cv = cv
        self.groups = groups
        self.oob_score = oob_score

    def get_params(self, deep=True):
        return self._get_params("estimator", deep=deep)

    def set_params(self, **params):
        return self._set_params("estimator", **params)

    #   @property
    #   def estimators(self):
    #       return [self.estimator]

    def fit(self, X, y, **params):
        cv = check_cv(self.cv, y, classifier=True)
        
        self.estimators_ = []
        if self.oob_score:
            self._oob_decision_function = np.zeroslike(y)
            self.oob_count_ = np.zeroslike(y, dtype=int)

        for train, test in cv.split(X, y, self.groups):
            estimator = clone(self.estimator).fit(X[train], y[train], **params)
            self.estimators_.append(estimator)

            if self.oob_score:
                self.oob_count_[test] += 1

                if hasattr(self.estimator, "decision_function"):
                    self._oob_decision_function[test] += estimator.decision_function(X[test])
                elif hasattr(self.estimator, "predict_proba"):
                    self._oob_decision_function[test] += np.vstack([
                        proba[:, 1]
                        for proba in estimator.predict_proba(X[test])
                    ]).T

        if self.oob_score:
            self._oob_decision_function /= self.oob_count_

            if callable(self.oob_score):
                self.oob_score_ = self.oob_score(y, self._oob_decision_function)
            else:
                self.oob_score_ = metrics.accuracy_score(
                    y, self._oob_decision_function > 0.5, average="micro",
                )

        return self

    @property
    def oob_decision_function_(self):
        return self._oob_decision_function


# class CVRegressor(MultiOutputVotingRegressor):
class CVRegressor(BaseEstimator):
    def __init__(self, estimator, cv=None, groups=None, oob_score=False):
        self.estimator = estimator
        self.cv = cv
        self.groups = groups
        self.oob_score = oob_score

    # @property
    # def estimators(self):
    #     return [self.estimator]

    # def get_params(self, deep=True):
    #     return self._get_params("estimator", deep=deep)

    # def set_params(self, **params):
    #     return self._set_params("estimator", **params)

    def predict(self, X, **params):
        outputs = [
            estimator.predict(X, **params)
            for estimator in self.estimators_
        ]
        return np.mean(outputs, axis=0)
    
    @property
    def n_outputs_(self):
        return self.estimators_[0].n_outputs_

    def _more_tags(self):
        return {"multioutput": True}

    def fit(self, X, y, **params):
        cv = check_cv(self.cv, y, classifier=False)
        
        self.estimators_ = []
        if self.oob_score:
            self._oob_prediction = np.zeroslike(y)
            self.oob_count_= np.zeroslike(y, dtype=int)

        for train, test in cv.split(X, y, self.groups):
            estimator = clone(self.estimator).fit(X[train], y[train], **params)
            self.estimators_.append(estimator)

            if self.oob_score:
                self.oob_count_[test] += 1
                pred = estimator.predict(X[test])
                self._oob_prediction[test] += pred

        if self.oob_score:
            self._oob_prediction /= self.oob_count_

            if callable(self.oob_score):
                self.oob_score_ = self.oob_score(y, self._oob_prediction)
            else:
                self.oob_score_ = metrics.mean_squared_error(y, self._oob_prediction)

        return self
    
    @property
    def oob_prediction_(self):
        return self._oob_prediction


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

