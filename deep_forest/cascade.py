from numbers import Integral, Real

import joblib
import numpy as np
from scipy import sparse
from sklearn.base import (
    TransformerMixin,
    BaseEstimator,
    MetaEstimatorMixin,
    clone,
    check_is_fitted,
    _fit_context,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import check_scoring, get_scorer_names
from sklearn.metrics._scorer import _check_multimetric_scoring
from sklearn.utils import check_random_state
from sklearn.utils._estimator_html_repr import _VisualBlock
from sklearn.utils._tags import _safe_tags
from sklearn.utils._param_validation import HasMethods, Interval, StrOptions, Hidden
from sklearn.utils.metaestimators import available_if, _BaseComposition
from sklearn.pipeline import (
    _fit_transform_one,
    _final_estimator_has,
    _print_elapsed_time,
)
from sklearn.compose import ColumnTransformer
from imblearn.pipeline import Pipeline, _fit_resample_one


class AlternatingLevel(TransformerMixin, _BaseComposition):
    # _parameter_constraints = {
    #     **ColumnTransformer._parameter_constraints,
    #     "last_output_indices": [dict, None],
    # }
    # del _parameter_constraints["remainder"]
    _required_parameters = ["transformers"]

    _parameter_constraints: dict = {
        "transformers": [list, Hidden(tuple)],
        "sparse_threshold": [Interval(Real, 0, 1, closed="both")],
        "n_jobs": [Integral, None],
        "transformer_weights": [dict, None],
        "verbose": ["verbose"],
        "verbose_feature_names_out": ["boolean"],
    }

    def __init__(
        self,
        transformers,
        *,
        sparse_threshold=0.3,
        n_jobs=None,
        transformer_weights=None,
        verbose=False,
        verbose_feature_names_out=True,
        last_output_indices=None,
    ):
        self.transformers = transformers
        self.sparse_threshold = sparse_threshold
        self.n_jobs = n_jobs
        self.transformer_weights = transformer_weights
        self.verbose = verbose
        self.verbose_feature_names_out = verbose_feature_names_out
        self.last_output_indices = last_output_indices
    
    def get_params(self, deep=True):
        return self._get_params("transformers", deep=deep)

    def set_params(self, **kwargs):
        return self._set_params("transformers", **kwargs)
    
    @property
    def _last_output_indices(self):
        # If no outputs exist from the last level
        if self.last_output_indices is None:
            return {name: slice(0, 0) for name, _ in self.transformers}
        return self.last_output_indices

    @property
    def output_indices_(self):
        check_is_fitted(self)
        return self.column_transformer_.output_indices_
    
    def _get_column_indices(self):
        slices = []
        for name, _ in self.transformers:
            new_slice = np.ones(self.n_features_in_, dtype=bool)
            new_slice[self._last_output_indices[name]] = False
            slices.append(new_slice)
        return slices

    def _make_column_transformer(self):
        return ColumnTransformer(
            transformers=[
                (*t, cols)
                for t, cols in zip(self.transformers, self._get_column_indices())
            ],
            sparse_threshold=self.sparse_threshold,
            n_jobs=self.n_jobs,
            transformer_weights=self.transformer_weights,
            verbose=self.verbose,
            verbose_feature_names_out=self.verbose_feature_names_out,
        )

    @_fit_context(
        # Transformers are not validated yet
        prefer_skip_nested_validation=False,
    )
    def fit(self, X, y=None):
        self._check_n_features(X, reset=True)
        self.column_transformer_ = self._make_column_transformer().fit(X, y)
        return self

    @_fit_context(
        # Transformers are not validated yet
        prefer_skip_nested_validation=False,
    )
    def fit_transform(self, X, y=None):
        self._check_n_features(X, reset=True)
        self.column_transformer_ = self._make_column_transformer()
        return self.column_transformer_.fit_transform(X, y)
    
    def transform(self, X):
        check_is_fitted(self)
        return self.column_transformer_.transform(X)
    
    def _sk_visual_block_(self):
        names, transformers = zip(*self.transformers)
        name_details = self._get_column_indices()
        return _VisualBlock(
            "parallel", transformers, names=names, name_details=name_details
        )


class Cascade(Pipeline):
    _parameter_constraints = {
        "level": [
            HasMethods(["fit", "transform"]),
            HasMethods(["fit_transform", "transform"]),
            # HasMethods(["fit_resample"]),  # TODO: Decide.
        ],
        "final_estimator": [
            HasMethods(["fit"]),
            StrOptions({"passthrough"}),
            None,
        ],
        "memory": [None, str, HasMethods(["cache"])],
        "verbose": ["boolean"],
        "max_levels": [Interval(Integral, 1, None, closed="left")],
        "min_levels": [Interval(Integral, 0, None, closed="left")],
        "scoring": [
            None,
            callable,
            StrOptions(set(get_scorer_names())),
            list,
            tuple,
            dict,
        ],
        "min_score": [Interval(Real, 0, None, closed="left")],
        "min_improvement": [None, Interval(Real, None, None, closed="both")],
        "min_relative_improvement": [None, Interval(Real, None, None, closed="both")],
        "max_unimproving_levels": [None, Interval(Integral, 0, None, closed="left")],
        "keep_original_features": ["boolean"],
        "validation_size": [
            None,
            Interval(Integral, 0, None, closed="left"),
            Interval(Real, 0, 1, closed="neither"),
            tuple,
        ],
        "inter_level_sampler": [HasMethods(["fit_resample"]), None],
        "random_state": ["random_state"],
    }

    def __init__(
        self,
        *,
        level,
        final_estimator,
        memory=None,
        verbose=False,
        max_levels=10,
        min_levels=1,
        scoring=None,
        min_score=0.0,
        min_improvement=None,
        min_relative_improvement=None,
        max_unimproving_levels=None,
        keep_original_features=True,
        validation_size=0.2,
        inter_level_sampler=None,
        trim_to_best_score=False,
        refit=True,
        random_state=None,
    ):
        self.final_estimator = final_estimator
        self.level = level
        self.memory = memory
        self.verbose = verbose
        self.max_levels = max_levels
        self.min_levels = min_levels
        self.scoring = scoring
        self.min_score = min_score
        self.min_improvement = min_improvement
        self.min_relative_improvement = min_relative_improvement
        self.max_unimproving_levels = max_unimproving_levels
        self.keep_original_features = keep_original_features
        self.validation_size = validation_size
        self.inter_level_sampler = inter_level_sampler
        self.trim_to_best_score = trim_to_best_score
        self.refit = refit
        self.random_state = random_state

        # FIXME: steps must exist for _more_tags() to work, but this contradicts
        # sklearn's principle of not validating params in __init___
        _fe = self.final_estimator
        self.steps = [
            (
                "final_estimator_",
                _fe if _fe in ("passthrough", None) else clone(_fe),
            )
        ]

    def _more_tags(self):
        return {"pairwise": _safe_tags(self.final_estimator, "pairwise")}

    def _combine_features(self, original_X, new_X):
        if self.keep_original_features:
            Xs = [original_X, new_X]
            if any(sparse.issparse(f) for f in Xs):
                return sparse.hstack(Xs, format="csr")
            return np.hstack(Xs)
        return new_X

    def _validate_stop_criteria(self):
        if self.min_levels > self.max_levels:
            raise ValueError(
                f"{self.min_levels=} must be less than or equal to {self.max_levels=}"
            )

        self.stop_criteria_are_set_ = (
            self.min_score > 0.0
            or self.min_improvement is not None
            or self.min_relative_improvement is not None
        )
        self.collect_scores_ = (
            self.stop_criteria_are_set_
            or self.trim_to_best_score
            # One may set scoring only for inspection purposes:
            or self.scoring is not None
        )

        if self.stop_criteria_are_set_ and isinstance(self.scorer_, dict):
            raise ValueError(
                "Setting numeric stop criteria is not allowed for multimetric scorers."
            )

        if self.collect_scores_ and self.scorer_ is None:
            raise ValueError(
                (
                    (
                        "self.min_score, self.min_improvement, and/or"
                        " self.min_relative_improvement are set,"
                    )
                    if self.stop_criteria_are_set_
                    else (f"{self.trim_to_best_score=}")
                )
                + (
                    f" but no scoring function was provided ({self.scoring=}).)"
                    ' One can set scoring="passthrough" to use the'
                    " final estimator's score method."
                )
            )

        if self.collect_scores_:
            self.level_scores_ = []

    def _validate_scorer(self):
        # Validate scoring stopping criterion
        if self.min_improvement is None:
            self.min_improvement_ = -np.inf
        else:
            self.min_improvement_ = self.min_improvement

        if self.min_relative_improvement is None:
            self.min_relative_improvement_ = -np.inf
        else:
            self.min_relative_improvement_ = self.min_relative_improvement

        if self.max_unimproving_levels is None:
            self.max_unimproving_levels_ = np.inf
        else:
            self.max_unimproving_levels_ = self.max_unimproving_levels

        if self.final_estimator in ("passthrough", None):
            self.scorer_ = None
        elif callable(self.scoring):
            self.scorer_ = self.scoring
        elif self.scoring is None or isinstance(self.scoring, str):
            self.scorer_ = check_scoring(self.final_estimator, self.scoring, allow_none=True)
        else:
            self.scorer_ = _check_multimetric_scoring(self.final_estimator, self.scoring)

        if self.trim_to_best_score and self.scorer_ is None:
            raise ValueError(
                "trim_to_best_score=True requires a 'self.scoring' to be set."
            )
        if (
            isinstance(self.scorer_, dict)
            and (self.trim_to_best_score not in (False, None, *self.scorer_))
        ):
            raise ValueError(
                f"For multimetric scorers, if {self.trim_to_best_score=} is specified"
                " it needs to be a string denoting one of its metrics."
                f" Received metrics {tuple(self.scorer_.keys())}"
            )

    def _stop_criterion(self, X, X_val, y, y_val):
        """Return True if the cascade should stop training."""
        # TODO: OOB scores to avoid validation set
        if not self.collect_scores_:
            return False

        with _print_elapsed_time(
            self.__class__.__name__,
            "Fit and score final estimator"
        ):
            estimator = clone(self.final_estimator).fit(X, y)

        X_val = self._apply_transformers(X_val)

        if isinstance(self.scorer_, dict):
            score = {
                name: scorer(estimator, X_val, y_val)
                for name, scorer in self.scorer_.items()
            }
        else:
            score = self.scorer_(estimator, X_val, y_val)

        # Store the scores for each level
        self.level_scores_.append(score)

        if not self.stop_criteria_are_set_:
            return False

        # If this is the first score, we can't compare it to the previous one
        if not hasattr(self, "_last_score"):
            self._last_score = score
            print(f"First score: {score:.4f}")
            return np.abs(score) > self.min_score

        delta = score - self._last_score
        relative_delta = delta / np.abs(self._last_score)

        improved = (
            delta >= self.min_improvement_
            or relative_delta >= self.min_relative_improvement_
        )
        if improved or not hasattr(self, "_n_unimproving_levels"):
            self._n_unimproving_levels = 0
        if not improved:
            self._n_unimproving_levels += 1

        stop = self.n_levels_ >= self.min_levels and (
            np.abs(score) > self.min_score
            or self._n_unimproving_levels > self.max_unimproving_levels_
        )
        print(
            f"Score: {score:.4f} Delta: {delta:.4f}"
            f" Relative delta: {relative_delta:.4f}"
            f" Unimproving levels: {self._n_unimproving_levels}"
        )

        self._last_score = score
        return stop

    def _check_fit_params(self, **fit_params):
        fit_params_steps = {
            "final_estimator": {},
            "level": {},
            "sampler": {},
        }
        for pname, pval in fit_params.items():
            step_param = "__" in pname and pname.split("__", 1)
            if not step_param or step_param[0] not in fit_params_steps:
                raise ValueError(
                    f"Cascade.fit does not accept the {pname} parameter. "
                    "You can pass parameters to the levels, samplers or final "
                    "estimator of the Cascade using the name__parameter "
                    "format, e.g. `Cascade.fit(X, y, level__sample_weight"
                    "=sample_weight, final_estimator__sample_weight"
                    "=sample_weight)`"
                )
            step, param = step_param
            fit_params_steps[step][param] = pval
        return fit_params_steps

    def _log_message(self, step_idx, step_name):
        if not self.verbose:
            return None
        if step_name == "final_estimator":
            return "Fitting final estimator"
        elif step_name == "sampler":
            return f"(level {step_idx + 1} of {self.max_levels}) " "Resampling data"
        elif step_name == "level":
            return (
                f"(level {step_idx + 1} of {self.max_levels}) "
                "Fitting level transformer"
            )
        raise ValueError(f"Unknown step name: {step_name}")

    def _validate_names(self, names):
        """Skip name validation, since names are generated automatically."""
        pass

    @_fit_context(
        # Level and final estimators are not validated yet
        prefer_skip_nested_validation=False,
    )
    def fit(self, X, y=None, **fit_params):
        """Fit the model.

        Fit all the transforms/samplers one after the other and
        transform/sample the data, then fit the transformed/sampled
        data using the final estimator.

        Parameters
        ----------
        X : iterable
            Training data. Must fulfill input requirements of first step of the
            pipeline.

        y : iterable, default=None
            Training targets. Must fulfill label requirements for all steps of
            the pipeline.

        **fit_params : dict of str -> object
            Parameters passed to the ``fit`` method of each step, where
            each parameter name is prefixed such that parameter ``p`` for step
            ``s`` has key ``s__p``.

        Returns
        -------
        self : Pipeline
            This estimator.
        """
        # We override this method only to correct log printing and docstring.
        self._validate_steps()
        self._validate_scorer()
        self._validate_stop_criteria()

        fit_params_steps = self._check_fit_params(**fit_params)
        Xt, yt = self._fit(X, y, **fit_params_steps)

        if self.refit and self.collect_scores_:
            print("Refitting...")
            Xt, yt = self._fit(X, y, final_fit=True, **fit_params_steps)

        if self._final_estimator != "passthrough":
            with _print_elapsed_time(
                self.__class__.__name__,
                self._log_message(None, "final_estimator"),
            ):
                fit_params_last_step = fit_params_steps["final_estimator"]
                self._final_estimator.fit(Xt, yt, **fit_params_last_step)

        return self

    def _fit(self, X, y=None, final_fit=False, **fit_params):
        # Setup the memory
        if self.memory is None or isinstance(self.memory, str):
            memory = joblib.Memory(location=self.memory, verbose=0)
        else:
            memory = self.memory

        fit_transform_one_cached = memory.cache(_fit_transform_one)
        fit_resample_one_cached = memory.cache(_fit_resample_one)

        if final_fit or not self.collect_scores_:
            X_val, y_val = None, None
        else:
            self.random_state_ = check_random_state(self.random_state)
            if isinstance(self.validation_size, tuple):
                # FIXME: no need for refit in this case
                # TODO: document self.validation_size as tuple of (X_val, y_val)
                # TODO: use some sklearn function to get scores and validation.
                # E.g. GridSearchCV, cross_val_score, etc. A search wrapper would
                # even make validation not a job of Cascade.
                X_val, y_val = self.validation_size
            else:
                X, X_val, y, y_val = train_test_split(
                    X,
                    y,
                    test_size=self.validation_size,
                    random_state=self.random_state_,
                )

        max_levels = self.n_levels_ if final_fit else self.max_levels
        original_X = X

        # Initialize steps with the final estimator
        # FIXME: also done in __init__
        self.steps = [
            (
                "final_estimator_",
                (
                    clone(self.final_estimator)
                    if isinstance(self.final_estimator, BaseEstimator)
                    else self.final_estimator
                ),
            ),
        ]
        self.n_levels_ = 0
        last_level = None

        # Collect scores without any level, only the final estimator.
        if not final_fit:
            self._stop_criterion(X, X_val, y, y_val)

        for level_count in range(max_levels):
            cloned_transformer = clone(self.level)

            if last_level is not None:
                # Make each transformer in the current level unaware of its own outputs
                # of the last level.
                if isinstance(cloned_transformer, AlternatingLevel):
                    cloned_transformer.set_params(
                        last_output_indices=last_level.output_indices_,
                    )
                # Find AlternatingLevel objects recursively
                for param_name, param in last_level.get_params().items():
                    if isinstance(param, AlternatingLevel):
                        cloned_transformer.set_params(
                            **{param_name + "__last_output_indices": param.output_indices_}
                        )

            # Fit or load from cache the current transformer
            if hasattr(cloned_transformer, "transform") or hasattr(
                cloned_transformer, "fit_transform"
            ):
                new_X, fitted_transformer = fit_transform_one_cached(
                    cloned_transformer,
                    X,
                    y,
                    None,
                    message_clsname=self.__class__.__name__,
                    message=self._log_message(level_count, "level"),
                    **fit_params["level"],
                )
            # TODO: Should we assume self.level is always a transformer?
            elif hasattr(cloned_transformer, "fit_resample"):
                new_X, y, fitted_transformer = fit_resample_one_cached(
                    cloned_transformer,
                    X,
                    y,
                    message_clsname=self.__class__.__name__,
                    message=self._log_message(level_count, "level"),
                    **fit_params["level"],
                )
            else:
                raise RuntimeError

            X = self._combine_features(original_X, new_X)
            self.steps.insert(-1, (f"level{level_count}", fitted_transformer))
            # TODO: inter-level steps, mutliple steps
            X, y = self._resample_data(X, y, **fit_params["sampler"])

            # TODO: make as property (useful for warm-start)
            self.n_levels_ = level_count + 1

            if not final_fit and self._stop_criterion(X, X_val, y, y_val):
                break

            last_level = fitted_transformer

        if not final_fit:
            self._trim_levels()

        return X, y

    def _trim_levels(self):
        if not self.trim_to_best_score:
            return

        # TODO: Consider self.min_levels
        self.n_levels_ = np.argmax(self.level_scores_)

        # self.level_scores_[0] corresponds to the empty cascade, only the final
        # estimator.
        self.trim_index_ = self.n_levels_ - 1
        if self.inter_level_sampler is None and self.trim_index_ >= 0:
            self.trim_index_ *= 2

        # TODO: document trimmed_steps_
        self.trimmed_steps_ = self.steps[self.trim_index_ + 1 : -1]
        self.steps[self.trim_index_ + 1 : -1] = []

        if self.verbose:
            print(
                f"Trimmed to {self.n_levels_} levels."
                f" Level scores: {self.level_scores_}"
            )

    # FIXME: maybe not needed anymore, it can simply be in a pipeline (or list of
    # steps) passed to self.level. AlternatingLevel takes care of a lot.
    def _resample_data(self, X, y, **fit_params):
        # TODO: cache
        if self.inter_level_sampler is None:
            return X, y

        level_count = (len(self.steps) - 1) // 2

        # Not strictly necessary, since samplers are stateless
        sampler = clone(self.inter_level_sampler)

        with _print_elapsed_time(
            self.__class__.__name__,
            self._log_message(level_count, "sampler"),
        ):
            X, y = sampler.fit_resample(X, y, **fit_params)

        # To enable easy refit and visulization of the cascade
        self.steps.insert(-1, (f"sampler{level_count}", sampler))

        return X, y

    def _apply_transformers(self, X):
        Xt = X
        for _, name, transform in self._iter(with_final=False):
            Xt = self._combine_features(X, transform.transform(Xt))
        return Xt

    @available_if(_final_estimator_has("predict"))
    def predict(self, X, **predict_params):
        """Transform the data, and apply `predict` with the final estimator.

        Call `transform` of each transformer in the pipeline. The transformed
        data are finally passed to the final estimator that calls `predict`
        method. Only valid if the final estimator implements `predict`.

        Parameters
        ----------
        X : iterable
            Data to predict on. Must fulfill input requirements of first step
            of the pipeline.

        **predict_params : dict of string -> object
            Parameters to the ``predict`` called at the end of all
            transformations in the pipeline. Note that while this may be
            used to return uncertainties from some models with return_std
            or return_cov, uncertainties that are generated by the
            transformations in the pipeline are not propagated to the
            final estimator.

            .. versionadded:: 0.20

        Returns
        -------
        y_pred : ndarray
            Result of calling `predict` on the final estimator.
        """
        Xt = self._apply_transformers(X)
        return self._final_estimator.predict(Xt, **predict_params)

    @available_if(_final_estimator_has("predict_proba"))
    def predict_proba(self, X, **predict_proba_params):
        """Transform the data, and apply `predict_proba` with the final estimator.

        Call `transform` of each transformer in the pipeline. The transformed
        data are finally passed to the final estimator that calls
        `predict_proba` method. Only valid if the final estimator implements
        `predict_proba`.

        Parameters
        ----------
        X : iterable
            Data to predict on. Must fulfill input requirements of first step
            of the pipeline.

        **predict_proba_params : dict of string -> object
            Parameters to the `predict_proba` called at the end of all
            transformations in the pipeline.

        Returns
        -------
        y_proba : ndarray of shape (n_samples, n_classes)
            Result of calling `predict_proba` on the final estimator.
        """
        Xt = self._apply_transformers(X)
        return self._final_estimator.predict_proba(Xt, **predict_proba_params)

    @available_if(_final_estimator_has("decision_function"))
    def decision_function(self, X):
        """Transform the data, and apply `decision_function` with the final estimator.

        Call `transform` of each transformer in the pipeline. The transformed
        data are finally passed to the final estimator that calls
        `decision_function` method. Only valid if the final estimator
        implements `decision_function`.

        Parameters
        ----------
        X : iterable
            Data to predict on. Must fulfill input requirements of first step
            of the pipeline.

        Returns
        -------
        y_score : ndarray of shape (n_samples, n_classes)
            Result of calling `decision_function` on the final estimator.
        """
        Xt = self._apply_transformers(X)
        return self._final_estimator.decision_function(Xt)

    @available_if(_final_estimator_has("score_samples"))
    def score_samples(self, X):
        """Transform the data, and apply `score_samples` with the final estimator.

        Call `transform` of each transformer in the pipeline. The transformed
        data are finally passed to the final estimator that calls
        `score_samples` method. Only valid if the final estimator implements
        `score_samples`.

        Parameters
        ----------
        X : iterable
            Data to predict on. Must fulfill input requirements of first step
            of the pipeline.

        Returns
        -------
        y_score : ndarray of shape (n_samples,)
            Result of calling `score_samples` on the final estimator.
        """
        Xt = self._apply_transformers(X)
        return self._final_estimator.score_samples(Xt)

    @available_if(_final_estimator_has("predict_log_proba"))
    def predict_log_proba(self, X, **predict_log_proba_params):
        """Transform the data, and apply `predict_log_proba` with the final estimator.

        Call `transform` of each transformer in the pipeline. The transformed
        data are finally passed to the final estimator that calls
        `predict_log_proba` method. Only valid if the final estimator
        implements `predict_log_proba`.

        Parameters
        ----------
        X : iterable
            Data to predict on. Must fulfill input requirements of first step
            of the pipeline.

        **predict_log_proba_params : dict of string -> object
            Parameters to the ``predict_log_proba`` called at the end of all
            transformations in the pipeline.

        Returns
        -------
        y_log_proba : ndarray of shape (n_samples, n_classes)
            Result of calling `predict_log_proba` on the final estimator.
        """
        Xt = self._apply_transformers(X)
        return self._final_estimator.predict_log_proba(Xt, **predict_log_proba_params)

    def _can_transform(self):
        return self._final_estimator == "passthrough" or hasattr(
            self._final_estimator, "transform"
        )

    @available_if(_can_transform)
    def transform(self, X):
        """Transform the data, and apply `transform` with the final estimator.

        Call `transform` of each transformer in the pipeline. The transformed
        data are finally passed to the final estimator that calls
        `transform` method. Only valid if the final estimator
        implements `transform`.

        This also works where final estimator is `None` in which case all prior
        transformations are applied.

        Parameters
        ----------
        X : iterable
            Data to transform. Must fulfill input requirements of first step
            of the pipeline.

        Returns
        -------
        Xt : ndarray of shape (n_samples, n_transformed_features)
            Transformed data.
        """
        Xt = self._apply_transformers(X)
        return self._final_estimator.transform(Xt)

    def _can_inverse_transform(self):
        return False

    @available_if(_final_estimator_has("score"))
    def score(self, X, y=None, sample_weight=None):
        """Transform the data, and apply `score` with the final estimator.

        Call `transform` of each transformer in the pipeline. The transformed
        data are finally passed to the final estimator that calls
        `score` method. Only valid if the final estimator implements `score`.

        Parameters
        ----------
        X : iterable
            Data to predict on. Must fulfill input requirements of first step
            of the pipeline.

        y : iterable, default=None
            Targets used for scoring. Must fulfill label requirements for all
            steps of the pipeline.

        sample_weight : array-like, default=None
            If not None, this argument is passed as ``sample_weight`` keyword
            argument to the ``score`` method of the final estimator.

        Returns
        -------
        score : float
            Result of calling `score` on the final estimator.
        """
        Xt = self._apply_transformers(X)
        score_params = {}
        if sample_weight is not None:
            score_params["sample_weight"] = sample_weight
        return self._final_estimator.score(Xt, y, **score_params)
