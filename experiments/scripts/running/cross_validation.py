"""
The :mod:`sklearn.model_selection._validation` module includes classes and
functions to validate the model.
"""

# Author: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#         Gael Varoquaux <gael.varoquaux@normalesup.org>
#         Olivier Grisel <olivier.grisel@ensta.org>
#         Raghav RV <rvraghav93@gmail.com>
#         Michal Karbownik <michakarbownik@gmail.com>
# License: BSD 3 clause


import time
import warnings
from numbers import Real
from pathlib import Path

import numpy as np
from joblib import logger

from sklearn.base import clone, is_classifier
from sklearn.metrics import check_scoring, get_scorer_names
from sklearn.metrics._scorer import _check_multimetric_scoring
from sklearn.utils import indexable
from sklearn.utils._param_validation import (
    HasMethods,
    Integral,
    StrOptions,
    validate_params,
)
from sklearn.utils.metaestimators import _safe_split
from sklearn.utils.parallel import Parallel, delayed
from sklearn.utils.validation import _num_samples
from sklearn.model_selection._split import check_cv
from sklearn.model_selection._validation import (
    _warn_or_raise_about_fit_failures,
    _insert_error_scores,
    _aggregate_score_dicts,
    _normalize_score_results,
    _score,
)
from skmultilearn.model_selection import IterativeStratification

# HACK
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from model_building.cascade_learn.cascade import Cascade


def make_iterative_stratification(**kwargs):
    """CV factory to use in YAML config files."""
    # NOTE: simply calling __init__ will not work, PyYAML expects module-level
    # functions.
    return IterativeStratification(**kwargs)


@validate_params(
    {
        "estimator": [HasMethods("fit")],
        "X": ["array-like", "sparse matrix"],
        "y": ["array-like", None],
        "groups": ["array-like", None],
        "scoring": [
            StrOptions(set(get_scorer_names())),
            callable,
            list,
            tuple,
            dict,
            None,
        ],
        "cv": ["cv_object"],
        "n_jobs": [Integral, None],
        "verbose": ["verbose"],
        "fit_params": [dict, None],
        "pre_dispatch": [Integral, str],
        "return_train_score": ["boolean"],
        "return_estimator": ["boolean"],
        "return_indices": ["boolean"],
        "error_score": [StrOptions({"raise"}), Real],
        "return_fitted_params": [
            None, "boolean", callable, list, tuple, set, str,
        ],
    },
    prefer_skip_nested_validation=False,  # estimator is not validated yet
)
def cross_validate_cascade_levels(
    estimator,
    X,
    y=None,
    *,
    groups=None,
    scoring=None,
    cv=None,
    n_jobs=None,
    verbose=0,
    fit_params=None,
    pre_dispatch="2*n_jobs",
    return_train_score=False,
    return_estimator=False,
    return_indices=False,
    return_fitted_params=False,
    error_score=np.nan,
):
    """Use warm-start to yield scores for all levels of a cascade."""
    X, y, groups = indexable(X, y, groups)

    cv = check_cv(cv, y, classifier=is_classifier(estimator))

    if callable(scoring):
        scorers = scoring
    elif scoring is None or isinstance(scoring, str):
        scorers = check_scoring(estimator, scoring)
    else:
        scorers = _check_multimetric_scoring(estimator, scoring)

    indices = cv.split(X, y, groups)
    if return_indices:
        # materialize the indices since we need to store them in the returned dict
        indices = list(indices)

    # We clone the estimator to make sure that all the folds are
    # independent, and that it is pickle-able.
    parallel = Parallel(n_jobs=n_jobs, verbose=verbose, pre_dispatch=pre_dispatch)
    results = parallel(
        delayed(_custom_fit_and_score)(
            estimator=clone(estimator),
            X=X,
            y=y,
            scorer=scorers,
            train=train,
            test=test,
            verbose=verbose,
            parameters=None,
            fit_params=fit_params,
            return_train_score=return_train_score,
            return_times=True,
            return_estimator=return_estimator,
            error_score=error_score,
            return_fitted_params=return_fitted_params,
        )
        for train, test in indices
    )

    _warn_or_raise_about_fit_failures(results, error_score)

    # For callable scoring, the return type is only know after calling. If the
    # return type is a dictionary, the error scores can now be inserted with
    # the correct key.
    if callable(scoring):
        _insert_error_scores(results, error_score)

    results = _aggregate_score_dicts(results)

    ret = {}
    ret["fit_time"] = results["fit_time"]
    ret["score_time"] = results["score_time"]

    if return_estimator:
        ret["estimator"] = results["estimator"]

    if return_indices:
        ret["indices"] = {}
        ret["indices"]["train"], ret["indices"]["test"] = zip(*indices)

    test_scores_dict = _normalize_score_results(results["test_scores"])
    if return_train_score:
        train_scores_dict = _normalize_score_results(results["train_scores"])

    for name in test_scores_dict:
        ret["test_%s" % name] = test_scores_dict[name]
        if return_train_score:
            key = "train_%s" % name
            ret[key] = train_scores_dict[name]
    
    if "fitted_params" in results:
        ret["fitted_params"] = _normalize_score_results(results["fitted_params"])

    return ret
 

def _custom_fit_and_score(
    estimator,
    X,
    y,
    scorer,
    train,
    test,
    verbose,
    parameters,
    fit_params,
    return_train_score=False,
    return_parameters=False,  # XXX
    return_n_test_samples=False,
    return_times=False,
    return_estimator=False,  # XXX
    split_progress=None,
    candidate_progress=None,
    error_score=np.nan,
    return_fitted_params=False,
):
    outer_start_time = time.time()
    fit_params = fit_params or {}
    progress_msg = ""

    if verbose > 2:
        if split_progress is not None:
            progress_msg = f" {split_progress[0]+1}/{split_progress[1]}"
        if candidate_progress and verbose > 9:
            progress_msg += f"; {candidate_progress[0]+1}/{candidate_progress[1]}"

    if verbose > 1:
        if parameters is None:
            params_msg = ""
        else:
            sorted_keys = sorted(parameters)  # Ensure deterministic o/p
            params_msg = ", ".join(f"{k}={parameters[k]}" for k in sorted_keys)
    if verbose > 9:
        start_msg = f"[CV{progress_msg}] START {params_msg}"
        print(f"{start_msg}{(80 - len(start_msg)) * '.'}")

    X_train, y_train = _safe_split(estimator, X, y, train)
    X_test, y_test = _safe_split(estimator, X, y, test, train)

    estimator = clone(estimator)

    # FIXME: this is a hack to get the cascade from the estimator when a dropper
    # is used with it in a pipeline. We should perform dropping here, as a
    # cross_validate parameter.
    # Or just get the max_levels parameter without considering specific classes.
    if isinstance(estimator, Cascade):
        cascade = estimator
    else:  # Get the cascade from the estimator
        cascades = [
            est for est in estimator.get_params().items()
            if isinstance(est[1], Cascade)
        ]
        if len(cascades) == 0:
            raise ValueError(f"Cascade not found in estimator: {estimator}")
        if len(cascades) > 1:
            warnings.warn(
                "Multiple cascades found in estimator params:"
                f" {[name for name, _ in cascades]}. Using the first one: {cascades[0][0]}."
            )

        cascade = cascades[0][1]

    cascade.set_params(warm_start=True)
    result = {}
    test_scores = {}
    train_scores = {}

    if return_n_test_samples:
        result["n_test_samples"] = _num_samples(X_test)

    for level in range(cascade.max_levels + 1):
        start_time = time.time()

        cascade.set_params(max_levels=level)
        estimator.fit(X_train, y_train, **fit_params)

        fit_time = time.time() - start_time
        test_scores |= {
            f"level{level}__" + name: score
            for name, score in _score(
                estimator, X_test, y_test, scorer, {}, error_score,
            ).items()
        }
        score_time = time.time() - start_time - fit_time
        if return_train_score:
            train_scores |= {
                f"level{level}__" + name: score
                for name, score in _score(
                    estimator, X_train, y_train, scorer, {}, error_score,
                ).items()
            }
        
        if return_times:
            result[f"level{level}__fit_time"] = fit_time
            result[f"level{level}__score_time"] = score_time
        
        if verbose > 1:
            total_time = score_time + fit_time
            end_msg = f"[CV{progress_msg}] END "
            result_msg = params_msg + (";" if params_msg else "")
            if verbose > 2:
                if isinstance(test_scores, dict):
                    for scorer_name in sorted(test_scores):
                        result_msg += f" {scorer_name}: ("
                        if return_train_score:
                            scorer_scores = train_scores[scorer_name]
                            result_msg += f"train={scorer_scores:.3f}, "
                        result_msg += f"test={test_scores[scorer_name]:.3f})"
                else:
                    result_msg += ", score="
                    if return_train_score:
                        result_msg += f"(train={train_scores:.3f}, test={test_scores:.3f})"
                    else:
                        result_msg += f"{test_scores:.3f}"
            result_msg += f" total time={logger.short_format_time(total_time)}"

            # Right align the result_msg
            end_msg += "." * (80 - len(end_msg) - len(result_msg))
            end_msg += result_msg
            print(end_msg)
        
    result["fit_error"] = None  # TODO: capture error
    if return_times:
        result["fit_time"] = time.time() - outer_start_time - score_time
        result["score_time"] = score_time  # Only last scoring

    result["test_scores"] = test_scores
    if return_train_score:
        result["train_scores"] = train_scores
    if return_fitted_params:
        result["fitted_params"] = _get_fitted_params(
            estimator,
            filter=(None if return_fitted_params == True else return_fitted_params),
            sep="."
        )
    
    return result


def _get_fitted_params(estimator, filter=None, sep=".", deep=True):
    params = getattr(estimator, "__dict__", {})

    if hasattr(estimator, "get_params"):
        # Some parameters are only recovered with deep=True, even though we
        # remove "__" containing parameters just after.
        params |= estimator.get_params(deep=True)

    params = {k: v for k, v in params.items() if "__" not in k}
    # FIXME: last_level_ is cascade specific
    params = {
        k: v for k, v in params.items()
        if k not in ("last_level", "last_level_")
    }

    if filter is None:
        filter_func = lambda x: True
    elif callable(filter):
        filter_func = filter
    elif isinstance(filter, (list, tuple, set)):
        filter_func = lambda x: x in filter
    elif isinstance(filter, str):
        filter_func = lambda x: x == filter
    else:
        raise ValueError(f"Invalid filter: {filter}")
    
    result = {k: v for k, v in params.items() if filter_func(k)}

    if not deep:
        return result

    for k, v in params.items():
        # HACK: recurse one more level to access fitted transformers at
        # ColumnTransformer.transformers_, since, unlike Pipeline, it does not
        # substitute the self.transformers with their fitted versions to be accessed by
        # get_params(deep=True).
        if k == "transformers_":
            for transformer_name, transformer, _ in v:
                for tk, tv in _get_fitted_params(
                    transformer, filter, sep, deep,
                ).items():
                    result[k + sep + transformer_name + sep + tk] = tv
        else:
            for k2, v2 in _get_fitted_params(v, filter, sep, deep).items():
                result[k + sep + k2] = v2

    return result
