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
    def __init__(
        self,
        predictors: list[BaseEstimator],
        transformers: list[BaseEstimator],
    ):
        self.predictors = predictors
        self.transformers = transformers

    def first_fit(self, X, y, **fit_params):
        self._validate_params()
        self.forests_ = []
        for forest in self.forests:
            self.forests_.append(clone(forest).fit(X, y, **fit_params))
        return self

    def internal_fit(self, XX_last, y, **fit_params):
        self.transformers_ = []

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

    def _transform_with_forest(self, X, forest):
        np.hstack([embed_with_tree(tree, X, method="all_nodes") for tree in forest])

    def fit(self, X, y, **fit_params):
        self._validate_params()
        self.levels_ = [  # Initialize first level
            [clone(forest).fit(X, y, **fit_params) for forest in self.level]
        ]

        # Only original_X will be appended in the first level
        XX_last = [[] for forest in self.levels_[-1]]

        for level_id in range(self.max_levels):
            print(f"Fitting level {level_id}")

            # TODO: X, y = level.transform_data(X, y)
            XX_last = [
                self._transform_with_forest(
                    # Exclude own last predictions in the last level
                    np.hstack([X] + XX_last[:i] + XX_last[i:]),
                    forest,
                )
                for i, forest in enumerate(self.levels_[-1])
            ]

            self._preprocess_X(X)

            self._target_transform()

            new_level = [clone(forest) for forest in self.level]

            # Fit new level
            for i, forest in enumerate(new_level):
                forest.fit(
                    # Exclude own last predictions in the last level
                    np.hstack([X] + XX_last[:i] + XX_last[i:]),
                    y,
                    **fit_params,
                )

            self.levels_.append(new_level)

            if self._stop_criterion(X, y):
                break

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
