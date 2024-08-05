from numbers import Real
from typing import Self
from warnings import warn

import numpy as np
from sklearn.base import (
    BaseEstimator,
    MetaEstimatorMixin,
    clone,
    is_classifier,
    check_is_fitted,
    _fit_context,
)
import sklearn.utils
from sklearn.utils._param_validation import (
    Interval, StrOptions, HasMethods, Hidden,
)
from imblearn.base import BaseSampler


def _standardize_proba(proba):
    """Standardize the probabilities returned by estimators.

    Single-output estimators, multi-output estimators, and multi-output
    oob_decision_function_ all return objects in different formats. This function
    standardizes them to a list of arrays of shape (n_samples, n_classes), being
    one array for each output (for each column of y).

    Parameters
    ----------
    proba : list of array-like of shape (n_samples, n_classes) or
            array-like of shape (n_samples, n_classes) or
            array-like of shape (n_samples, n_classes, n_outputs)
        The probabilities returned by an estimator.
    
    Returns
    -------
    proba : list of array-like of shape (n_samples, n_classes)
    """
    if isinstance(proba, list):
        # Multi-output, shape=(n_outputs, n_samples, n_classes)
        return proba
    if isinstance(proba, np.ndarray):
        if proba.ndim == 2:
            # Single output, shape=(n_samples, n_classes)
            return [proba]
        if proba.ndim == 3:
            # Multi-output, shape=(n_samples, n_classes, n_outputs)
            return list(proba.transpose(2, 0, 1))

    shape = getattr(proba, "shape", None) or f"{len(proba)=}"
    raise ValueError(
        "Expected proba to be a list or numpy array of shape"
        " (n_samples, n_classes) or (n_samples, n_classes, n_outputs)."
        f" Got {type(proba)=}. Shape: {shape}."
    )


def _get_classes(estimator):
    if hasattr(estimator, "classes_"):
        return estimator.classes_
    if estimator.n_outputs_ == 1:
        return np.array([0, 1])
    return [np.array([0, 1])] * estimator.n_outputs_


class WeakLabelImputer(BaseSampler, MetaEstimatorMixin):
    _sampling_type = "clean-sampling"
    _parameter_constraints = {
        "estimator": [
            HasMethods(["fit", "predict_proba"]),
            HasMethods(["fit", "predict"]),
        ],
        "threshold": [Interval(Real, 0, 1, closed="left")],
        "use_oob_proba": ["boolean"],
        "weight_proba": ["boolean"],
        "verbose": ["verbose"],
        "sampling_strategy": [StrOptions({"auto"})],
    }

    def __init__(
        self,
        estimator,
        threshold=0.8,
        use_oob_proba=False,
        weight_proba=False,
        verbose=False,
        sampling_strategy="auto",
    ):
        self.estimator = estimator
        self.threshold = threshold
        self.use_oob_proba = use_oob_proba
        self.weight_proba = weight_proba
        self.verbose = verbose
        self.sampling_strategy = sampling_strategy

    def _weight_proba(
        self,
        proba: list[np.ndarray],
        y: np.ndarray,
        classes: list[np.ndarray] | np.ndarray,
    ):
        """Weight the predicted probabilities based on the prior of each class.

        Parameters:
        -----------
        proba : list of array-like of shapes (n_samples, n_classes)
            The predicted probabilities of the samples belonging to each class.
            Each array represents the predicted probabilities for a different
            output (column of y).
        y : array-like of shape (n_samples, n_outputs)
            The true labels of the samples.
        classes : array-like of shape (n_classes,) or list of array-like
            The class labels. If a list of arrays, each array represents the class
            labels for a different output.

        Returns:
        --------
        new_probas : list of array-like
            The weighted predicted probabilities of the samples belonging to each class.

        Notes:
        ------
        This method weights the predicted probabilities by the prior (class frequency)
        of each class. The prior is calculated based on the class distribution in the
        training labels. The weighted probabilities are divided by the sum along the
        second axis to ensure they sum up to 1 for each sample.

        If `self.weight_proba` is False, the original predicted probabilities are returned
        without any weighting.
        """
        if not self.weight_proba:
            return proba

        # We assume proba was already standardized as:
        # proba = _standardize_proba(proba)

        # Weight probabilities by the prior of each class
        new_probas = []
        # TODO: do not use classes and y here, set weights before
        # (label_frequency_estimates would be the weights, for example)
        for proba_col, y_col, class_names in zip(proba, y.T, classes):
            counts_idx, counts = np.unique(y_col, return_counts=True)

            # Ensure counts are in the same order as class_names
            counts_order = [np.where(counts_idx == c)[0][0] for c in class_names]
            counts = counts[counts_order]

            # Divide by each class prior (class frequency)
            proba_col *= y.shape[0] / counts
            proba_col /= proba_col.sum(axis=1, keepdims=True)
            new_probas.append(proba_col)

        return new_probas

    def _apply_threshold(self, proba, threshold):
        return np.hstack(
            [np.max(p, axis=1, keepdims=True) < threshold for p in proba]
        )
    
    def _proba_to_pred(self, estimator, proba, n_samples):
        # Based on sklearn.tree._classes.BaseDecisionTree.predict
        classes = _get_classes(estimator)

        if estimator.n_outputs_ == 1:
            return classes.take(np.argmax(proba[0], axis=1), axis=0).reshape(-1, 1)

        # (n_samples, n_classes, n_outputs) -> (n_samples, n_outputs, n_classes)
        proba = np.dstack(proba).transpose(0, 2, 1)

        class_type = classes[0].dtype
        predictions = np.zeros((n_samples, estimator.n_outputs_), dtype=class_type)

        for k in range(estimator.n_outputs_):
            predictions[:, k] = classes[k].take(
                np.argmax(proba[:, k], axis=1), axis=0
            )

        return predictions

    def _fit_predict_proba(self, X, y, **params):
        estimator = clone(self.estimator).fit(X, y, **params)

        # TODO: use sklearn's check_array
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        # TODO: tests
        if is_classifier(estimator):
            if self.use_oob_proba:
                proba = estimator.oob_decision_function_
            else:
                proba = estimator.predict_proba(X)
        else:
            if self.use_oob_proba:
                proba = estimator.oob_prediction_
            else:
                proba = estimator.predict(X)
            
            # if estimator.n_outputs_ == 1:
            if proba.ndim == 1:
                proba = proba.reshape(-1, 1)

            proba = np.dstack([1 - proba, proba]).transpose(0, 2, 1)
            
        proba = _standardize_proba(proba)  # Convert to list of arrays

        return estimator, proba

    @_fit_context(
        # Estimator is not validated yet.
        prefer_skip_nested_validation=False,
    )
    def _fit_resample(self, X, y, **params):
        estimator, proba = self._fit_predict_proba(X, y, **params)
        proba = self._weight_proba(proba, y, _get_classes(estimator))

        # Recover true labels for samples with low confidence
        # kept is True for samples with low confidence in all classes, selecting
        # the labels to NOT change
        kept = self._apply_threshold(proba, self.threshold)

        # Take the class with highest probability for each output
        y_pred = self._proba_to_pred(estimator, proba, y.shape[0])
        # This would not take weighted proba into account:
        #   y_pred = estimator.predict(X)  
        
        y = y.reshape(y_pred.shape)
        kept = kept.reshape(y_pred.shape)
        
        if self.verbose:
            n_changed = (y_pred[~kept] != y[~kept]).sum()
            self.print_message(
                f"Changing {n_changed}/{y.size} ({n_changed / y.size:.2%}) labels."
            )

        # Predictions with probability larger than threshold are imputed into y
        y_pred[kept] = y[kept]

        return X, y_pred

    # FIXME: we are skipping validation since imblearn does not support multilabel
    def fit_resample(self, X, y, **params):
        return self._fit_resample(X, y, **params)
    
    def print_message(self, message):
        if self.verbose:
            print(f"[{self.__class__.__name__}] " + message)
    

class PositiveUnlabeledImputer(WeakLabelImputer):
    def _fit_resample(self, X, y):
        Xt, yt = super().fit_resample(X, y)
        final_y = (y.astype(bool) | yt.astype(bool)).astype(int)

        if self.verbose:
            n_changed = (final_y != y).sum()
            n_pos = y.sum()
            self.print_message(
                f"Imputted {n_changed}/{n_pos:.0f}"
                f" ({n_changed / n_pos:.2%}) positive labels."
            )
        
        return Xt, final_y


class SCARImputer(WeakLabelImputer):
    _parameter_constraints = {
        **WeakLabelImputer._parameter_constraints,
        "label_freq_percentile": [Interval(Real, 0, 1, closed="neither")],
        "last_level": [Hidden(None), Hidden(WeakLabelImputer)],
    }

    def __init__(
        self,
        estimator,
        *,
        label_freq_percentile=0.5,
        threshold=0.5,
        verbose=False,
        use_oob_proba=True,
        last_level: Self | None = None,
    ):
        """Impute missing labels under the SCAR assumption.

        TIcE inspired. Selected completely at random (SCAR).

        label_freq_percentile is delta
        """
        self.last_level = last_level
        self.label_freq_percentile = label_freq_percentile

        super().__init__(
            estimator=estimator,
            threshold=threshold,
            verbose=verbose,
            weight_proba=True,  # Does not make sense otherwise
            sampling_strategy="auto",
            use_oob_proba=use_oob_proba,
        )

    def _estimate_label_frequencies(self, y_original, proba, X=None, y=None):
        label_frequency_estimates = []

        for y_col, proba_col in zip(y_original.T, proba):
            proba_col = proba_col[:, 1]
            # TODO: move to tests
            assert proba_col.shape == y_col.shape

            positives_proba = proba_col[y_col.astype(bool)]
            freq_estimate = np.percentile(
                positives_proba,
                100 * self.label_freq_percentile,
            )
            label_frequency_estimates.append(freq_estimate)

        self.print_message(
            f"Estimated label frequencies (c) for each output:"
            f" [{' '.join(f'{c:.2f}' for c in label_frequency_estimates)}]"
        )

        return np.array(label_frequency_estimates)

    def _weight_proba(
        self,
        proba: list[np.ndarray],
        y: np.ndarray,
        classes: list[np.ndarray] | np.ndarray,
    ):
        """Weight the predicted probabilities based on the prior of each class.

        Parameters:
        -----------
        proba : list of array-like of shapes (n_samples, n_classes)
            The predicted probabilities of the samples belonging to each class.
            Each array represents the predicted probabilities for a different
            output (column of y).
        y : array-like of shape (n_samples, n_outputs)
            The true labels of the samples.
        classes : array-like of shape (n_classes,) or list of array-like
            The class labels. If a list of arrays, each array represents the class
            labels for a different output.

        Returns:
        --------
        new_probas : list of array-like
            The weighted predicted probabilities of the samples belonging to each class.

        Notes:
        ------
        This method weights the predicted probabilities by the prior (class frequency)
        of each class. The prior is calculated based on the class distribution in the
        training labels. The weighted probabilities are divided by the sum along the
        second axis to ensure they sum up to 1 for each sample.

        If `self.weight_proba` is False, the original predicted probabilities are returned
        without any weighting.
        """
        # We assume proba was already standardized as:
        # proba = _standardize_proba(proba)
        
        if not self.weight_proba:
            return proba

        # Weight probabilities by the estimated label frequency
        new_probas = []
        for proba_col, label_freq in zip(
            proba,
            self.label_frequency_estimates_[self.enable_imputation_]
        ):
            proba_col = proba_col.copy()
            proba_col[:, 1] = (proba_col[:, 1] / label_freq).clip(0, 1)
            proba_col[:, 0] = 1 - proba_col[:, 1]
            new_probas.append(proba_col)

        return new_probas
    
    # TODO: receive y
    def _apply_threshold(self, proba, threshold):
        keep = super()._apply_threshold(proba, threshold)
        active_original_y = self.original_y_[:, self.enable_imputation_]
        keep[active_original_y.astype(bool)] = True
        return keep

    def _fit_predict_proba(self, X, y, **params):
        if self.last_level is None:
            self.enable_imputation_ = np.ones(y.shape[1], dtype=bool)
            self.next_enable_imputation_ = self.enable_imputation_.copy()
            self.original_y_ = y
        else:
            self.enable_imputation_ = self.last_level.next_enable_imputation_.copy()
            self.next_enable_imputation_ = self.enable_imputation_.copy()
            self.original_y_ = self.last_level.original_y_

            if not self.enable_imputation_.any():
                self.print_message("No labels to impute.")
                return None, y

        active_y = y[:, self.enable_imputation_]
        estimator, proba = super()._fit_predict_proba(X, active_y, **params)

        self.label_frequency_estimates_ = (
            self._estimate_label_frequencies(
                self.original_y_[:, self.enable_imputation_],
                proba,
                X=X,
                y=active_y,
            )
        )
        return estimator, proba

    def _fit_resample(self, X, y, **params):
        # TODO: enforce binary/multilabel y
        estimator, proba = self._fit_predict_proba(X, y, **params)

        if estimator is None:  # No labels to impute
            return X, y

        active_y = y[:, self.enable_imputation_]
        active_original_y = self.original_y_[:, self.enable_imputation_]

        proba = self._weight_proba(
            proba, active_original_y, _get_classes(estimator),
        )

        # kept is True for samples with low confidence in all classes, selecting
        # the labels to NOT change
        kept = self._apply_threshold(proba, self.threshold)

        # Take the class with highest probability for each output
        y_pred = self._proba_to_pred(estimator, proba, active_y.shape[0])
        # This would not take weighted proba into account:
        #   y_pred = estimator.predict(X)  
        
        active_y = active_y.reshape(y_pred.shape)
        kept = kept.reshape(y_pred.shape)
        
        if self.verbose:
            n_changed = (y_pred[~kept] != active_y[~kept]).sum()
            self.print_message(
                f"Imputting {n_changed}/{y.size}"
                f" ({n_changed / y.size:.2%}) labels relative to last layer."
            )

        y_pred[kept] = active_original_y.reshape(y_pred.shape)[kept]
        y_pred_full = y.copy()
        y_pred_full[:, self.enable_imputation_] = y_pred

        return X, y_pred_full


class LabelComplementImputer(SCARImputer):
    def __init__(
        self,
        estimator,
        *,
        label_freq_percentile=0.5,
        threshold=0.5,
        weight_proba=False,
        verbose=False,
        use_oob_proba=True,
        last_level: Self | None = None,
    ):
        """Impute missing labels under the SCAR assumption.

        TIcE inspired. Selected completely at random (SCAR).
        """
        self.weight_proba = weight_proba

        super().__init__(
            estimator=estimator,
            threshold=threshold,
            verbose=verbose,
            last_level=last_level,
            label_freq_percentile=label_freq_percentile,
            use_oob_proba=use_oob_proba,
        )

    def _fit_predict_proba(self, X, y, **params):
        # Estimate label frequencies only in the first cascade level
        if self.last_level is None:
            return super()._fit_predict_proba(X, y, **params)

        self.enable_imputation_ = (
            self.last_level.next_enable_imputation_.copy()
        )
        self.next_enable_imputation_ = self.enable_imputation_.copy()

        self.label_frequency_estimates_ = (
            self.last_level.label_frequency_estimates_.copy()
        )
        self.original_y_ = self.last_level.original_y_

        if not self.enable_imputation_.any():
            self.print_message("No labels to impute.")
            return None, y

        active_y = y[:, self.enable_imputation_]

        return WeakLabelImputer._fit_predict_proba(self, X, active_y, **params)

    def _apply_threshold(self, proba, threshold):
        keep_masks = []
        n_samples = proba[0].shape[0]

        for i, proba_col, y_col, label_freq in zip(
            np.nonzero(self.enable_imputation_)[0],
            proba,
            self.original_y_.T[self.enable_imputation_],
            self.label_frequency_estimates_[self.enable_imputation_],
        ):
            keep_mask = np.ones_like(y_col, dtype=bool)
            proba_col = proba_col[:, 1]

            # Change at most max_labels labels. We use n_samples as max to avoid
            # integer overflow (if label_freq is 0 or too small).
            max_labels = int(min(np.sum(y_col) / label_freq, n_samples))

            if max_labels == n_samples:
                warn(
                    "Expected number of labels is too large:"
                    f" {np.sum(y_col) / label_freq=} > {n_samples=}"
                )

            # Change labels with highest probability of being positive.
            # We add 2 * y_col to the probabilities to include the original positives
            # in the "total number of labels" that must not exceed max_labels.
            # [::-1][:max_labels] is used instead of [-max_labels:] to avoid
            # the case where max_labels=0.
            change_idx = np.argsort(proba_col + 2 * y_col)[::-1][:max_labels]
            keep_mask[change_idx] = False

            # Filter out predictions with low confidence or negative.
            keep_mask |= proba_col < threshold

            # Using y_col to avoid original positives to be filtered out.
            # (TODO: which should never happen, the following should be enough).
            #     n_labeled = (~keep_mask).sum()
            n_labeled = ((~keep_mask) | y_col.astype(bool)).sum()

            if n_labeled >= max_labels:
                # Signal to the next level to stop imputing for this output
                self.next_enable_imputation_[i] = False
                self.print_message(f"Stopped imputation for output {i}.")
            if n_labeled > max_labels:
                warn(
                    f"Imputed more labels than expected: {n_labeled=} > {max_labels=}"
                )

            # Keep original positives, since change_idx marked them to be changed.
            keep_mask |= y_col.astype(bool)

            keep_masks.append(keep_mask)

        return np.vstack(keep_masks).T
