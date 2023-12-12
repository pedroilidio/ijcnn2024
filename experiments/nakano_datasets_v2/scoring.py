"""Employ tree embeddings and weak-label inputting in deep forest models.
"""
import copy

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
from sklearn.pipeline import Pipeline, FeatureUnion
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
from sklearn.metrics import get_scorer
from sklearn.metrics._scorer import _BaseScorer
from sklearn.utils.validation import _check_response_method


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
        if isinstance(score_func, _BaseScorer):
            sign = score_func._sign
            kwargs = score_func._kwargs
            response_method = score_func._response_method
    
            if isinstance(score_func, MultiLabelScorer):
                average = score_func._average

            score_func = score_func._score_func

        kwargs = kwargs or {}
        self._average = average

        super().__init__(score_func, sign, kwargs, response_method)
    
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
        if self._average is None:
            return self._score_func(y_true, y_pred, **scoring_kwargs)
        elif self._average == "micro":
            return self._score_func(
                y_true.reshape(-1), y_pred.reshape(-1), **scoring_kwargs,
            )
        elif self._average in ("macro", "weighted"):
            y_true = y_true.T
            y_pred = y_pred.T
        elif self._average != "samples":
            raise ValueError

        scores = np.array([
            self._score_func(col_true, col_pred, **scoring_kwargs)
            for col_true, col_pred in zip(y_true, y_pred)
        ])

        if self._average == "weighted":
            weights = y_true.sum(axis=1, dtype=np.float64)
            return np.sum(scores * weights) / weights.sum()

        return np.mean(scores)


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


scoring_metrics = {
    "f1_macro": "f1_macro",
    "f1_micro": "f1_micro",
    "f1_weighted": "f1_weighted",
    "f1_samples": "f1_samples",
    "precision_macro": "precision_macro",
    "precision_micro": "precision_micro",
    "precision_weighted": "precision_weighted",
    "precision_samples": "precision_samples",
    "recall_macro": "recall_macro",
    "recall_micro": "recall_micro",
    "recall_weighted": "recall_weighted",
    "recall_samples": "recall_samples",
    "jaccard_macro": "jaccard_macro",
    "jaccard_micro": "jaccard_micro",
    "jaccard_weighted": "jaccard_weighted",
    "jaccard_samples": "jaccard_samples",
    "average_precision_macro": MultiLabelScorer(
        sklearn.metrics.average_precision_score,
        average="macro",
        response_method=("decision_function", "predict_proba", "predict"),
        sign=1,
    ),
    "average_precision_micro": MultiLabelScorer(
        sklearn.metrics.average_precision_score,
        average="micro",
        response_method=("decision_function", "predict_proba", "predict"),
        sign=1,
    ),
    "average_precision_weighted": MultiLabelScorer(
        sklearn.metrics.average_precision_score,
        average="weighted",
        response_method=("decision_function", "predict_proba", "predict"),
        sign=1,
    ),
    "average_precision_samples": MultiLabelScorer(
        sklearn.metrics.average_precision_score,
        average="samples",
        response_method=("decision_function", "predict_proba", "predict"),
        sign=1,
    ),
    "roc_auc_macro": MultiLabelScorer(
        sklearn.metrics.roc_auc_score,
        average="macro",
        response_method=("decision_function", "predict_proba", "predict"),
        sign=1,
    ),
    "roc_auc_micro": MultiLabelScorer(
        sklearn.metrics.roc_auc_score,
        average="micro",
        response_method=("decision_function", "predict_proba", "predict"),
        sign=1,
    ),
    "roc_auc_weighted": MultiLabelScorer(
        sklearn.metrics.roc_auc_score,
        average="weighted",
        response_method=("decision_function", "predict_proba", "predict"),
        sign=1,
    ),
    "roc_auc_samples": MultiLabelScorer(
        sklearn.metrics.roc_auc_score,
        average="samples",
        response_method=("decision_function", "predict_proba", "predict"),
        sign=1,
    ),
    "neg_hamming_loss": MultiLabelScorer(
        sklearn.metrics.hamming_loss,
        response_method="predict",
        sign=-1,
    ),
    "label_ranking_average_precision_score": sklearn.metrics.make_scorer(
        sklearn.metrics.label_ranking_average_precision_score,
        response_method=("decision_function", "predict_proba", "predict"),
        greater_is_better=True,
    ),
    "neg_label_ranking_loss": sklearn.metrics.make_scorer(
        sklearn.metrics.label_ranking_loss,
        response_method=("decision_function", "predict_proba", "predict"),
        greater_is_better=False,
    ),
    "neg_coverage_error": MultiLabelScorer(
        sklearn.metrics.coverage_error,
        response_method=("decision_function", "predict_proba", "predict"),
        sign=-1,
    ),
    "ndgc": MultiLabelScorer(
        sklearn.metrics.ndcg_score,
        response_method=("decision_function", "predict_proba", "predict"),
        sign=1,
    ),
    "matthews_corrcoef_micro": MultiLabelScorer(
        sklearn.metrics.matthews_corrcoef,
        average="micro",
        response_method="predict",
        sign=1,
    ),
    "matthews_corrcoef_macro": MultiLabelScorer(
        sklearn.metrics.matthews_corrcoef,
        average="macro",
        response_method="predict",
        sign=1,
    ),
    "matthews_corrcoef_weighted": MultiLabelScorer(
        sklearn.metrics.matthews_corrcoef,
        average="weighted",
        response_method="predict",
        sign=1,
    ),
    "matthews_corrcoef_samples": MultiLabelScorer(
        sklearn.metrics.matthews_corrcoef,
        average="samples",
        response_method="predict",
        sign=1,
    ),
}

micro_scorers = {
    "f1": MultiLabelScorer(
        get_scorer("f1_micro"),
        average="micro",
    ),
    "precision": MultiLabelScorer(
        get_scorer("precision_micro"),
        average="micro",
    ),
    "recall": MultiLabelScorer(
        get_scorer("recall_micro"),
        average="micro",
    ),
    "jaccard": MultiLabelScorer(
        get_scorer("jaccard_micro"),
        average="micro",
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
    "ndgc": MultiLabelScorer(
        sklearn.metrics.ndcg_score,
        response_method=("decision_function", "predict_proba", "predict"),
        average=None,
        sign=1,
    ),
}

level_scorers = {}
for metric, scorer in micro_scorers.items():
    for average in (
        ("micro", "macro", "weighted", "samples")
        if scorer._average == "micro" else (None,)
    ):
        # Set version of scorer with corresponding average
        new_scorer = copy.deepcopy(scorer)
        new_scorer._average = average
        scorer_name = metric
        if average is not None:
            scorer_name += "_" + average

        level_scorers[scorer_name] = new_scorer

        # Set out-of-bag version of scorer (will only work on training data)
        level_scorers[scorer_name + "_oob"] = OOBScorer(new_scorer)


def parse_multilabel_proba(y):
    return np.hstack([label[:, 1, np.newaxis] for label in y])


def get_level_scores(cascade, level, *args, **kwargs):
    """Get scores for each level of a cascade."""
    raise NotImplementedError
    return getattr(cascade, "level_scores_", [])
