from typing import Self

import numpy as np
from sklearn.base import MetaEstimatorMixin, _fit_context
from sklearn.model_selection import cross_val_predict
from sklearn.utils._param_validation import HasMethods
from imblearn.base import BaseSampler

from deep_forest import tice


class LabelComplementImputer(BaseSampler, MetaEstimatorMixin):
    _sampling_type = "clean-sampling"
    _parameter_constraints = {
        "estimator": [HasMethods(["fit", "predict"])],
        "tice_params": [None, dict],
        "cv_params": [None, dict],
        "verbose": ["verbose"],
    }

    def __init__(
        self,
        estimator,
        *,
        verbose=False,
        tice_params=None,
        cv_params=None,
        last_level: Self | None = None,
    ):
        """Impute missing labels under the SCAR assumption.

        TIcE inspired. Selected completely at random (SCAR).
        """
        self.estimator = estimator
        self.tice_params = tice_params
        self.cv_params = cv_params
        self.verbose = verbose
        self.last_level = last_level

    def _estimate_label_frequencies(self, X, y):
        self.tice_params_ = dict(
            n_folds=5,
            max_bepp=5,
            delta=None,
            n_iter=2,
            max_splits=500,
            most_promising_only=False,
            min_set_size=10,
            random_state=None,
        )
        self.tice_params_ |= self.tice_params or {}

        avg_estimates, _ = tice.estimate_label_frequency_lower_bound(
            X, y, **self.tice_params_,
        )
        self.print_message(
            f"Estimated label frequencies (c) for each output:"
            f" [{' '.join(f'{c:.2f}' for c in avg_estimates)}]"
        )
        return np.array(avg_estimates)

    def _init_level(self, X, y):
        if self.last_level is None:
            self.enable_imputation_ = np.ones(y.shape[1], dtype=bool)
            self.next_enable_imputation_ = self.enable_imputation_.copy()
            self.original_y_ = y

            self.label_frequency_estimates_ = self._estimate_label_frequencies(X, y)
            self.label_count_upper_bounds_ = np.floor(
                self.original_y_.sum(0) / self.label_frequency_estimates_
            ).astype(int)
        else:
            self.enable_imputation_ = self.last_level.next_enable_imputation_.copy()
            self.next_enable_imputation_ = self.enable_imputation_.copy()
            self.original_y_ = self.last_level.original_y_
            self.label_frequency_estimates_ = self.last_level.label_frequency_estimates_
            self.label_count_upper_bounds_ = self.last_level.label_count_upper_bounds_

    # FIXME: we are skipping validation since imblearn does not support multilabel
    def fit_resample(self, X, y, **params):
        return self._fit_resample(X, y, **params)

    @_fit_context(
        # Estimator is not validated yet.
        prefer_skip_nested_validation=False,
    )
    def _fit_resample(self, X, y, **params):
        self._init_level(X, y)

        if not self.enable_imputation_.any():
            self.print_message("No labels to impute.")
            return X, y

        self.print_message(
            f"Imputation is active for {self.enable_imputation_.sum()} / {y.shape[1]}"
            " label columns."
        )

        active_y = y[:, self.enable_imputation_]
        active_original_y = self.original_y_[:, self.enable_imputation_]

        yt = cross_val_predict(
            self.estimator, X=X, y=active_y, **(self.cv_params or {}),
        )
        # Keep original label occurrences
        yt = (active_original_y.astype(int) | yt.astype(int)).astype(y.dtype)

        self.next_enable_imputation_[self.enable_imputation_] = (
            yt.sum(0) < self.label_count_upper_bounds_[self.enable_imputation_]
        )

        yt_full = y.copy()
        yt_full[:, self.enable_imputation_] = yt

        self.print_message(
            f"Imputed {yt.sum() - active_y.sum()} new labels."
            f" Before we had {y.sum()}. Label density is now {yt_full.mean():.5f}"
        )

        return X, yt_full

    def print_message(self, message):
        if self.verbose:
            print(f"[{self.__class__.__name__}] " + message)
    