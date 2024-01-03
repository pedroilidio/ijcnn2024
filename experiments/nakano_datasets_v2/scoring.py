"""Employ tree embeddings and weak-label inputting in deep forest models.
"""
import copy
from typing import Callable
from warnings import warn

import numpy as np
import sklearn.metrics
from sklearn.base import (
    BaseEstimator,
    MetaEstimatorMixin,
    TransformerMixin,
    clone,
    _fit_context,
    is_regressor,
    is_classifier,
)
from sklearn.datasets import load_iris
from sklearn.multioutput import MultiOutputClassifier
from sklearn.decomposition import PCA, SparsePCA, TruncatedSVD, NMF
from sklearn.ensemble import (
    BaggingClassifier,
    VotingClassifier,
    StackingClassifier,
    RandomForestClassifier,
    ExtraTreesClassifier,
    RandomForestRegressor,
    ExtraTreesRegressor,
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import (
    KFold,
    RepeatedKFold,
    RepeatedStratifiedKFold,
)
from sklearn.utils import check_random_state, estimator_html_repr
from sklearn.utils.multiclass import type_of_target
from sklearn.utils._param_validation import (
    Interval,
    HasMethods,
    StrOptions,
    validate_params,
)
from sklearn.metrics import get_scorer, make_scorer
from sklearn.metrics._scorer import _BaseScorer
from sklearn.utils.validation import _check_response_method
from sklearn.pipeline import Pipeline

from positive_dropper import PositiveDropper


def get_oob_proba(estimator, X=None, pos_label=1):
    """Get out-of-bag scores for each tree in a forest."""
    if hasattr(estimator, "oob_decision_function_"):
        assert all(classes[pos_label] == 1 for classes in estimator.classes_)
        # Select only the positive class
        return estimator.oob_decision_function_[:, pos_label, :]
    elif hasattr(estimator, "_final_estimator"):
        return get_oob_proba(estimator._final_estimator)
    elif hasattr(estimator, "estimator_"):
        return get_oob_proba(estimator.estimator_)
    elif hasattr(estimator, "estimators_"):
        return np.mean(
            [
                get_oob_proba(tree)
                for tree in estimator.estimators_
            ],
            axis=0,
        )
    else:
        raise ValueError


def get_oob_predictions(estimator, X=None, pos_label=1):
    return (get_oob_proba(estimator, X, pos_label) > 0.5).astype(int)


def tp(y_true, y_pred, **kwargs):
    return sklearn.metrics.confusion_matrix(
        y_true, y_pred, normalize="all", **kwargs,
    )[1, 1]


def tn(y_true, y_pred, **kwargs):
    return sklearn.metrics.confusion_matrix(
        y_true, y_pred, normalize="all", **kwargs,
    )[0, 0]


def fp(y_true, y_pred, **kwargs):
    return sklearn.metrics.confusion_matrix(
        y_true, y_pred, normalize="all", **kwargs,
    )[0, 1]


def fn(y_true, y_pred, **kwargs):
    return sklearn.metrics.confusion_matrix(
        y_true, y_pred, normalize="all", **kwargs,
    )[1, 0]


class MultiLabelScorer(_BaseScorer):
    """
    A scorer for multi-label classification tasks.

    Parameters:
    ----------
    score_func : callable
        The scoring function to use.
    sign : int, optional
        The sign of the score. Default is None.
    kwargs : dict, optional
        Additional keyword arguments to be passed to the scoring function. Default is None.
    response_method : str, optional
        The method used to obtain the predicted response from the estimator. Default is "predict".
    average : str, optional
        The averaging method to use for multi-label scores. Default is "micro".

    Attributes:
    ----------
    _average : str
        The averaging method used for multi-label scores.

    Methods:
    -------
    _score(self, method_caller, estimator, X, y_true, **kwargs)
        Compute the score for the given estimator and input data.

    """

    def __init__(
        self,
        score_func,
        *,
        sign=None,
        kwargs=None,
        response_method="predict",
        average="micro",
    ):
        kwargs = kwargs or {}

        if isinstance(score_func, _BaseScorer):
            sign = score_func._sign
            kwargs |= score_func._kwargs
            response_method = score_func._response_method
    
            if isinstance(score_func, MultiLabelScorer):
                average = score_func._average

            score_func = score_func._score_func

        self._average = average

        super().__init__(
            score_func=score_func,
            sign=sign,
            kwargs=kwargs,
            response_method=response_method,
        )
    
    def _score(self, method_caller, estimator, X, y_true, **kwargs):
        self._warn_overlap(
            message=(
                "There is an overlap between set kwargs of this scorer instance and"
                " passed metadata. Please pass them either as kwargs to `make_scorer`"
                " or metadata, but not both."
            ),
            kwargs=kwargs,
        )

        pos_label = None if is_regressor(estimator) else self._get_pos_label()

        response_method = _check_response_method(estimator, self._response_method)
        y_pred = method_caller(
            estimator, response_method.__name__, X, pos_label=pos_label
        )

        if isinstance(y_pred, list):  # multilabel probabilities
            y_pred = parse_multilabel_proba(y_pred)

        scoring_kwargs = {**self._kwargs, **kwargs}
        return self._sign * self._reshaped_score_func(y_true, y_pred, **scoring_kwargs)

    def _reshaped_score_func(self, y_true, y_pred, **scoring_kwargs):
        if y_true.ndim == 1:
            y_true = y_true.reshape(-1, 1)
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(-1, 1)

        if self._average is None:
            return self._score_func(y_true, y_pred, **scoring_kwargs)
        elif self._average == "micro":
            return self._score_func(
                y_true.reshape(-1, 1), y_pred.reshape(-1, 1), **scoring_kwargs,
            )
        elif self._average in ("macro", "weighted"):
            y_true = y_true.T
            y_pred = y_pred.T
        elif self._average != "samples":
            raise ValueError

        scores = np.full(len(y_true), np.nan)

        for i, (col_true, col_pred) in enumerate(zip(y_true, y_pred)):
            try:
                scores[i] = self._score_func(col_true, col_pred, **scoring_kwargs)
            except (IndexError, ValueError):
                pass

        if np.isnan(scores).all():
            return np.nan

        if self._average == "weighted":
            weights = y_true.sum(axis=1, dtype=np.float64)
            return np.nansum(scores * weights) / weights.sum()

        return np.nanmean(scores)


class OOBScorer(MultiLabelScorer):
    """
    Out-of-Bag Scorer for evaluating the performance of an estimator using out-of-bag samples.

    Parameters:
    -----------
    _BaseScorer : class
        Base class for scoring functions.

    Methods:
    --------
    _score(self, method_caller, estimator, X, y_true, **kwargs):
        Compute the score for the given estimator using out-of-bag samples.

    """

    def _score(self, method_caller, estimator, X, y_true, **kwargs):
        self._warn_overlap(
            message=(
                "There is an overlap between set kwargs of this scorer instance and"
                " passed metadata. Please pass them either as kwargs to `make_scorer`"
                " or metadata, but not both."
            ),
            kwargs=kwargs,
        )

        pos_label = None if is_regressor(estimator) else self._get_pos_label()
        response_method = _check_response_method(estimator, self._response_method)

        if is_classifier(estimator) and response_method.__name__ == "predict":
            y_pred = get_oob_predictions(estimator, pos_label=pos_label)
        else:
            y_pred = get_oob_proba(estimator)

        if y_pred.shape != y_true.shape:
            return np.nan

        scoring_kwargs = {**self._kwargs, **kwargs}
        return self._sign * self._reshaped_score_func(y_true, y_pred, **scoring_kwargs)


class DroppedLabelsScorer(MultiLabelScorer):
    def _score(self, method_caller, estimator, X, y_true, **kwargs):
        self._warn_overlap(
            message=(
                "There is an overlap between set kwargs of this scorer instance and"
                " passed metadata. Please pass them either as kwargs to `make_scorer`"
                " or metadata, but not both."
            ),
            kwargs=kwargs,
        )

        pos_label = None if is_regressor(estimator) else self._get_pos_label()

        response_method = _check_response_method(estimator, self._response_method)

        if not (
            isinstance(estimator, Pipeline)
            and isinstance(estimator.steps[0][1], PositiveDropper)
        ):
            return np.nan

        dropper = estimator.steps[0][1]

        if dropper.n_samples_fit_ != len(y_true):
            warn(
                f"{type(self).__name__} was meant to be used on training data only."
                f" Received {dropper.n_samples_fit_=} while {len(y_true)=}."
            )
            return np.nan

        y_pred = method_caller(
            estimator, response_method.__name__, X, pos_label=pos_label
        )

        if isinstance(y_pred, list):  # multilabel probabilities
            y_pred = parse_multilabel_proba(y_pred)

        # Mask all positives at first
        mask = y_true.copy().astype(bool)

        # Unmask positives that were dropped
        for i, idx in enumerate(dropper.masked_indices_):
            mask[idx, i] = False

        # Scikit-learn ignores masks, but we use them in self._reshaped_score_func
        y_true_masked = np.ma.masked_array(y_true, mask=mask)
        y_pred_masked = np.ma.masked_array(y_pred, mask=mask)

        scoring_kwargs = {**self._kwargs, **kwargs}

        return self._sign * self._reshaped_score_func(
            y_true_masked, y_pred_masked, **scoring_kwargs,
        )

    def _reshaped_score_func(self, y_true, y_pred, **scoring_kwargs):
        if y_true.ndim == 1 or self._average == "micro":
            # Implied: y_pred.ndim == 1 or self._average == "micro"
            y_true = y_true.data[~(y_true.mask)].reshape(-1, 1)
            y_pred = y_pred.data[~(y_pred.mask)].reshape(-1, 1)

        if self._average is None:
            raise ValueError("DroppedLabelsScorer does not support average=None.")
        elif self._average == "micro":
            return self._score_func(y_true, y_pred, **scoring_kwargs)
        elif self._average in ("macro", "weighted"):
            y_true = y_true.T
            y_pred = y_pred.T
            if self._average == "weighted":
                weights = y_true.sum(axis=1, dtype=np.float64).data
        elif self._average != "samples":
            raise ValueError

        y_true_cols = [col[~mask] for col, mask in zip(y_true.data, y_true.mask)]
        y_pred_cols = [col[~mask] for col, mask in zip(y_pred.data, y_pred.mask)]

        scores = np.full(len(y_true_cols), np.nan)

        for i, (col_true, col_pred) in enumerate(zip(y_true_cols, y_pred_cols)):
            try:
                scores[i] = self._score_func(col_true, col_pred, **scoring_kwargs)
            except (IndexError, ValueError):
                pass

        if np.isnan(scores).all():
            return np.nan

        if self._average == "weighted":
            return np.nansum(scores * weights) / weights.sum()

        return np.nanmean(scores)


class InternalScorer(_BaseScorer):
    """Score based on the modified data seen by the estimator.

    Labels are dropped before scoring.
    """
    def __init__(self, scorer: _BaseScorer) -> None:
        self.scorer = scorer
    
    @property
    def _score_func(self):
        return self.scorer._score_func
    
    @property
    def _sign(self):
        return self.scorer._sign
    
    @property
    def _kwargs(self):
        return self.scorer._kwargs
    
    @property
    def _response_method(self):
        return self.scorer._response_method
    
    def _score(self, method_caller, estimator, X, y_true, **kwargs):
        if not (
            isinstance(estimator, Pipeline)
            and isinstance(estimator.steps[0][1], PositiveDropper)
        ):
            return np.nan

        dropper = clone(estimator.steps[0][1])

        if not isinstance(dropper.random_state, int):
            raise ValueError(
                "InternalScorer only works with PositiveDropper with integer"
                " random_state."
            )

        _, yt = dropper.fit_resample(X, y_true)

        return self.scorer._score(method_caller, estimator, X, yt, **kwargs)


micro_scorers = {
    "tp": make_scorer(
        tp,
        greater_is_better=True,
        response_method="predict",
        labels=[0, 1],
    ),
    "tn": make_scorer(
        tn,
        greater_is_better=True,
        response_method="predict",
        labels=[0, 1],
    ),
    "fp": make_scorer(
        fp,
        greater_is_better=False,
        response_method="predict",
        labels=[0, 1],
    ),
    "fn": make_scorer(
        fn,
        greater_is_better=False,
        response_method="predict",
        labels=[0, 1],
    ),
    "f1": MultiLabelScorer(
        get_scorer("f1_micro"),
        average="micro",
        kwargs=dict(labels=[0, 1]),
    ),
    "precision": MultiLabelScorer(
        get_scorer("precision_micro"),
        average="micro",
        kwargs=dict(labels=[0, 1]),
    ),
    "recall": MultiLabelScorer(
        get_scorer("recall_micro"),
        average="micro",
        kwargs=dict(labels=[0, 1]),
    ),
    "jaccard": MultiLabelScorer(
        get_scorer("jaccard_micro"),
        average="micro",
        kwargs=dict(labels=[0, 1]),
    ),
    "matthews_corrcoef": MultiLabelScorer(
        sklearn.metrics.matthews_corrcoef,
        response_method="predict",
        sign=1,
        average="micro",
    ),
    "average_precision": MultiLabelScorer(
        sklearn.metrics.average_precision_score,
        response_method=("decision_function", "predict_proba", "predict"),
        sign=1,
        average="micro",
    ),
    "roc_auc": MultiLabelScorer(
        sklearn.metrics.roc_auc_score,
        response_method=("decision_function", "predict_proba", "predict"),
        sign=1,
        average="micro",
        kwargs=dict(labels=[0, 1]),
    ),
    "neg_hamming_loss": MultiLabelScorer(
        sklearn.metrics.hamming_loss,
        response_method="predict",
        sign=-1,
        average="micro",
    ),
    "label_ranking_average_precision_score": MultiLabelScorer(
        sklearn.metrics.label_ranking_average_precision_score,
        response_method=("decision_function", "predict_proba", "predict"),
        average=None,
        sign=1,
    ),
    "neg_label_ranking_loss": MultiLabelScorer(
        sklearn.metrics.label_ranking_loss,
        response_method=("decision_function", "predict_proba", "predict"),
        average=None,
        sign=-1,
    ),
    "neg_coverage_error": MultiLabelScorer(
        sklearn.metrics.coverage_error,
        response_method=("decision_function", "predict_proba", "predict"),
        average=None,
        sign=-1,
    ),
    "ndcg": MultiLabelScorer(
        sklearn.metrics.ndcg_score,
        response_method=("decision_function", "predict_proba", "predict"),
        average=None,
        sign=1,
    ),
}

all_scorers = {}
oob_scorers = {}
masked_scorers = {}

for metric, scorer in micro_scorers.items():
    for average in (
        ("micro", "macro", "weighted", "samples")
        if getattr(scorer, "_average", "micro") == "micro" else (None,)
    ):
        # Set version of scorer with corresponding average
        new_scorer = MultiLabelScorer(copy.deepcopy(scorer))
        new_scorer._average = average
        scorer_name = metric
        if average is not None:
            scorer_name += "_" + average

        all_scorers[scorer_name] = new_scorer

        # Set out-of-bag version of scorer (will only work on training data)
        oob_scorers[scorer_name] = OOBScorer(new_scorer)
        if average is not None:
            # Set dropped labels version of scorer (will only work on training data)
            masked_scorers[scorer_name] = DroppedLabelsScorer(new_scorer)


# Add OOB scorers
all_scorers |= {k + "_oob": v for k, v in oob_scorers.items()}

# Use train data and OOB to compute internal scores
internal_scorers = {k: InternalScorer(v) for k, v in all_scorers.items()}

all_scorers |= {k + "_internal": v for k, v in internal_scorers.items()}
all_scorers |= {k + "_masked": v for k, v in masked_scorers.items()}


def parse_multilabel_proba(y):
    return np.hstack([label[:, 1, np.newaxis] for label in y])


def get_level_scores(cascade, level, *args, **kwargs):
    """Get scores for each level of a cascade."""
    raise NotImplementedError
    return getattr(cascade, "level_scores_", [])
