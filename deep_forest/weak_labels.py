import numpy as np
from numbers import Real
from sklearn.base import (
    BaseEstimator,
    MetaEstimatorMixin,
    clone,
    is_classifier,
)
import sklearn.utils
from sklearn.utils._param_validation import Interval, StrOptions, HasMethods
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


class WeakLabelImputer(BaseSampler, MetaEstimatorMixin):
    _sampling_type = "clean-sampling"
    _parameter_constraints = {
        "estimator": [BaseEstimator],
        "threshold": [Interval(Real, 0, 1, closed="left")],
        "use_oob_proba": ["boolean"],
        "weight_proba": ["boolean"],
        "verbose": ["boolean"],
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

    def _validate_estimator(self):
        if not is_classifier(self.estimator):
            raise ValueError(
                "'estimator' parameter must be a classifier instance. "
                f"Got {self.estimator}.",
            )
        if not self.use_oob_proba and not hasattr(self.estimator, "predict_proba"):
            raise ValueError(
                "'estimator' parameter must be a classifier with "
                "'predict_proba' method. "
                f"Got {self.estimator}."
            )
    
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

        proba = _standardize_proba(proba)  # Ensure proba is a list of arrays
        if len(proba) == 1:
            classes = [classes]  # Ensure classes is a list of arrays
        
        # Weight probabilities by the prior of each class
        new_probas = []
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

    def _apply_threshold(self, proba):
        return np.hstack(
            [np.max(p, axis=1, keepdims=True) < self.threshold for p in proba]
        )
    
    def _proba_to_pred(self, classifier, proba, n_samples):
        # Based on sklearn.tree._classes.BaseDecisionTree.predict
        # (n_samples, n_classes, n_outputs) -> (n_samples, n_outputs, n_classes)
        proba = np.dstack(proba).transpose(0, 2, 1)

        if classifier.n_outputs_ == 1:
            return classifier.classes_.take(np.argmax(proba, axis=1), axis=0)

        class_type = classifier.classes_[0].dtype
        predictions = np.zeros((n_samples, classifier.n_outputs_), dtype=class_type)

        for k in range(classifier.n_outputs_):
            predictions[:, k] = classifier.classes_[k].take(
                np.argmax(proba[:, k], axis=1), axis=0
            )

        return predictions

    def fit_resample(self, X, y):
        self._validate_estimator()
        if self.use_oob_proba and any(len(np.unique(y_col)) == 1 for y_col in y.T):
            # FIXME: this is specific to scikit-learn forests.
            raise ValueError(
                "Cannot use OOB estimates: there are y columns with a single label."
            )
        classifier = clone(self.estimator).fit(X, y)

        # TODO: use sklearn's check_array
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        if self.use_oob_proba:
            proba = classifier.oob_decision_function_
        else:
            proba = classifier.predict_proba(X)
        
        proba = _standardize_proba(proba)  # Convert to list of arrays
        proba = self._weight_proba(proba, y, classifier.classes_)

        # Recover true labels for samples with low confidence
        # kept is True for samples with low confidence in all classes, selecting
        # the labels to NOT change
        kept = self._apply_threshold(proba)

        # Take the class with highest probability for each output
        y_pred = self._proba_to_pred(classifier, proba, y.shape[0])
        # This would not take weighted proba into account:
        #   y_pred = classifier.predict(X)  
        
        y = y.reshape(y_pred.shape)
        kept = kept.reshape(y_pred.shape)
        
        if self.verbose:
            n_changed = (y_pred[~kept] != y[~kept]).sum()
            print(
                f"[{self.__class__.__name__}] Changing {n_changed}/{y.size}"
                f" ({n_changed / y.size:.2%}) labels."
            )

        # Predictions with probability larger than threshold are imputed into y
        y_pred[kept] = y[kept]

        return X, y_pred

    # FIXME: we are skipping validation since imblearn does not support multilabel
    def _fit_resample(self, X, y):
        return self.fit_resample(X, y)
    

class PositiveUnlabeledImputer(WeakLabelImputer):
    def fit_resample(self, X, y):
        Xt, yt = super().fit_resample(X, y)
        final_y = (y | yt).astype(int)

        if self.verbose:
            n_changed = (final_y != y).sum()
            n_pos = y.sum()
            print(
                f"[{self.__class__.__name__}] Imputted {n_changed}/{n_pos:.0f}"
                f" ({n_changed / n_pos:.2%}) positive labels."
            )
        
        return Xt, final_y
