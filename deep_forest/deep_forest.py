from abc import ABCMeta, abstractmethod
import numpy as np
from sklearn.ensemble._forest import BaseForest
from sklearn.base import (
    BaseEstimator,
    clone,
    MetaEstimatorMixin,
    check_is_fitted,
)
from sklearn.utils.metaestimators import available_if
from sklearn.utils._param_validation import (
    validate_params,
    StrOptions,
    HasMethods,
)
from sklearn.pipeline import _final_estimator_has
from .tree_embedder import embed_with_tree


class BaseLevel(BaseEstimator, metaclass=ABCMeta):
    def __init__(self, forests: list[BaseForest]):
        self.forests = forests

    def fit(self, X, y, **fit_params):
        self._validate_params()
        self.forests_ = []
        for forest in self.forests:
            self.forests_.append(clone(forest).fit(X, y, **fit_params))
        return self

    @abstractmethod
    def transform(self, X, y=None):
        ...

    @abstractmethod
    def predict(self, X):
        ...


class TransformerLevel(BaseLevel):
    def fit(self, X, y, **fit_params):
        self.y_fit_ = y
        super().fit(X, y, **fit_params)
        return self

    def transform(self, X, y=None):
        return np.hstack([forest.transform(X) for forest in self.forests_])

    def predict(self, X):
        return self.y_fit_


class ForestEmbedderLevel(BaseLevel):
    def __init__(
        self,
        forests: list[BaseForest],
        method="all_nodes",
        post_transformer=None,
    ):
        self.forests = forests
        self.method = method
        self.post_transformer = post_transformer

    def _forest_embedd(self, X):
        return np.hstack(
            [
                embed_with_tree(tree, X, self.method)
                for tree in forest
                for forest in self.forests_
            ]
        )

    def fit(self, X, y, **fit_params):
        self._validate_params()
        self.forests_ = []
        for forest in self.forests:
            self.forests_.append(clone(forest).fit(X, y, **fit_params))

        if self.post_transformer is not None:
            X, y = self._forest_embedd(X), self.predict(X)
            # TODO: fit_transform() on post_transformer
            self.post_transformer_ = clone(self.post_transformer).fit(X, y)

        return self

    def transform(self, X):
        check_is_fitted(self)
        X = self._forest_embedd(X)
        if self.post_transformer is None:
            return X
        return self.post_transformer_.transform(X)

    def predict(self, X):
        return np.mean([forest.predict(X) for forest in self.forests_], axis=0)


class DeepForest(BaseEstimator):
    _parameter_constraints = {
        "level": [BaseLevel, HasMethods(["fit", "transform", "predict"])],
        "final_estimator": [BaseEstimator],
        "max_levels": [int],
        "keep_original_features": ["boolean"],
    }

    def __init__(
        self,
        *,
        level: BaseLevel,
        final_estimator: BaseEstimator,
        max_levels: int = 10,
        keep_original_features: bool = True,
    ):
        self.level = level
        self.final_estimator = final_estimator
        self.max_levels = max_levels
        self.keep_original_features = keep_original_features

    @property
    def _final_estimator(self):
        return self.final_estimator_

    def _more_tags(self):
        # TODO: check other tags
        return {"pairwise": self.level._get_tags("pairwise")}

    def fit(self, X, y, **fit_params):
        self._validate_params()
        self.levels_ = []
        self.final_estimator_ = clone(self.final_estimator)

        original_X = X

        for level_id in range(self.max_levels):
            print(f"Fitting level {level_id}")
            level = clone(self.level)
            level.fit(X, y, **fit_params)
            self.levels_.append(level)

            X, y = level.transform(X, y), level.predict(X)
            X = self._append_original_features(original_X, X)

            if self._stop_criterion(X, y):
                break

        print("Fitting final estimator")
        self.final_estimator_.fit(X, y)

        return self

    def _stop_criterion(self, X, y):
        return False  # TODO: implement

    def _append_original_features(self, original_X, X):
        if not self.keep_original_features:
            return X
        return np.hstack((original_X, X))

    def _apply_levels(self, X):
        original_X = X
        for level in self.levels_:
            X = level.transform(X)
            X = self._append_original_features(original_X, X)
        return X

    @available_if(_final_estimator_has("predict"))
    def predict(self, X):
        X_ = self._apply_levels(X)
        return self.final_estimator_.predict(X_)

    @available_if(_final_estimator_has("fit_predict"))
    def fit_predict(self, X):
        X_ = self._apply_levels(X)
        return self.final_estimator_.fit_predict(X_)

    @available_if(_final_estimator_has("predict_proba"))
    def predict_proba(self, X):
        X_ = self._apply_levels(X)
        return self.final_estimator_.predict_proba(X_)

    @available_if(_final_estimator_has("predict_log_proba"))
    def predict_log_proba(self, X):
        X_ = self._apply_levels(X)
        return self.final_estimator_.predict_log_proba(X_)

    @available_if(_final_estimator_has("decision_function"))
    def decision_function(self, X):
        X_ = self._apply_levels(X)
        return self.final_estimator_.decision_function(X_)

    @available_if(_final_estimator_has("score"))
    def score(self, X):
        X_ = self._apply_levels(X)
        return self.final_estimator_.score(X_)

    @available_if(_final_estimator_has("score_samples"))
    def score_samples(self, X):
        X_ = self._apply_levels(X)
        return self.final_estimator_.score_samples(X_)
