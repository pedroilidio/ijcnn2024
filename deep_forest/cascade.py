from numbers import Integral, Real
import joblib
import numpy as np
from scipy import sparse
from sklearn.base import (
    TransformerMixin,
    BaseEstimator,
    MetaEstimatorMixin,
    clone,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import check_scoring, get_scorer_names
from sklearn.utils import check_random_state
from sklearn.utils._tags import _safe_tags
from sklearn.utils._param_validation import HasMethods, Interval, StrOptions
from sklearn.utils.metaestimators import available_if
from sklearn.pipeline import (
    _fit_transform_one,
    _final_estimator_has,
    _print_elapsed_time,
)
from sklearn.compose import ColumnTransformer
from imblearn.pipeline import Pipeline, _fit_resample_one


class Cascade(Pipeline):
    _parameter_constraints = {
        "level": [
            HasMethods(["fit", "transform"]),
            HasMethods(["fit_transform", "transform"]),
            list,
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
        "scoring": [None, callable, StrOptions(set(get_scorer_names()))],
        "min_score": [Interval(Real, 0, None, closed="left")],
        "min_improvement": [None, Interval(Real, None, None, closed="both")],
        "min_relative_improvement": [None, Interval(Real, None, None, closed="both")],
        "max_unimproving_levels": [None, Interval(Integral, 0, None, closed="left")],
        "keep_original_features": ["boolean"],
        "validation_size": [
            Interval(Integral, 0, None, closed="left"),
            Interval(Real, 0, 1, closed="neither"),
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
        # FIXME: steps must exist for _more_tags() to work
        self.steps = [
            ("final_estimator_", clone(self.final_estimator)),
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
        self.collect_scores_ = self.stop_criteria_are_set_ or self.trim_to_best_score

        if not self.collect_scores_ and self.scoring is not None:
            raise ValueError(
                f"self.scoring other than None (received {self.scoring}) requires"
                " at least one of 'min_score', 'min_improvement',"
                " or 'min_relative_improvement' parameters to be set or"
                " 'trim_to_best_score' to be True. Otherwise, there is no"
                " use for the scoring function and it must be set to None."
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
        # TODO: check validation_size

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

        self.scorer_ = check_scoring(
            self.final_estimator,
            scoring=self.scoring,
            allow_none=True,
        )
        if self.scorer_ is None and self.trim_to_best_score:
            raise ValueError(
                "trim_to_best_score=True requires a 'self.scoring' to be set."
            )

    def _stop_criterion(self, X, X_val, y, y_val):
        """Return True if the cascade should stop training."""
        # TODO: OOB scores to avoid validation set
        # if self.scorer_ is None:
        if not self.collect_scores_:
            return False

        estimator = clone(self.final_estimator).fit(X, y)
        X_val = self._apply_transformers(X_val)
        score = self.scorer_(estimator, X_val, y_val)

        if self.trim_to_best_score:
            # Store the score for each level
            if not hasattr(self, "level_scores_"):
                self.level_scores_ = []
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

    def _validate_steps(self):
        """Complement _validate_params in the case where self.level is a list."""
        if isinstance(self.level, list):
            for name_transformer in self.level:
                if (
                    not isinstance(name_transformer, tuple)
                    or len(name_transformer) != 2
                    or not isinstance(name_transformer[0], str)
                ):
                    raise TypeError(
                        "self.level should be either a transformer or a list of"
                        " tuples in the format (transformer_name, transformer)."
                        f" Got {name_transformer} instead."
                    )

                transformer = name_transformer[1]

                if not hasattr(transformer, "transform") or (
                    not hasattr(transformer, "fit")
                    and not hasattr(transformer, "fit_transform")
                ):
                    raise TypeError(
                        "If self.level is a list, all the transformers must implement"
                        " transform and either fit or fit_transform."
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
        self._validate_params()
        self._validate_steps()
        self._validate_scorer()
        self._validate_stop_criteria()

        fit_params_steps = self._check_fit_params(**fit_params)
        Xt, yt = self._fit(X, y, **fit_params_steps)

        if self.refit and self.collect_scores_:
            print("Refitting...")
            Xt, yt = self._fit(X, y, final_fit=True, **fit_params_steps)

        with _print_elapsed_time(
            self.__class__.__name__,
            self._log_message(None, "final_estimator"),
        ):
            if self._final_estimator != "passthrough":
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
            X, X_val, y, y_val = train_test_split(
                X,
                y,
                test_size=self.validation_size,
                random_state=self.random_state_,
            )

        max_levels = self.n_levels_ if final_fit else self.max_levels
        original_X = X

        if isinstance(self.level, list):
            # TODO: Make a transformer wrapper to do the column swapping,
            # passing always a single trasnformer to self.level
            # Update: Probably not possible, since every level must swap columns
            base_level = ColumnTransformer(
                [(name, transformer, slice(None)) for name, transformer in self.level]
            )
            # No outputs exist from the last level
            last_level_slices = {name: slice(0, 0) for name, _ in self.level}

        else:
            base_level = self.level

        # Initialize steps with the final estimator
        self.steps = [
            ("final_estimator_", clone(self.final_estimator)),
        ]
        self.n_levels_ = 0

        for level_count in range(max_levels):
            cloned_transformer = clone(base_level)

            if isinstance(self.level, list):
                # Make each transformer in the current level unaware of its own outputs
                # of the last level.
                new_transformers = []
                for i, item in enumerate(cloned_transformer.transformers):
                    name, transformer, old_slice = item
                    new_slice = np.ones(X.shape[1], dtype=bool)
                    new_slice[last_level_slices[name]] = False
                    new_transformers.append((name, transformer, new_slice))

                cloned_transformer.transformers = new_transformers

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

            # TODO: make as property
            self.n_levels_ = level_count + 1

            if self._stop_criterion(X, X_val, y, y_val):
                break

            if isinstance(self.level, list):
                last_level_slices = fitted_transformer.output_indices_

        if not final_fit:
            self._trim_levels()

        return X, y

    def _trim_levels(self):
        if not self.trim_to_best_score:
            return

        # TODO: Consider self.min_levels
        # TODO: level_scores_[0] could be the case of no levels, only the final
        # estimator.
        self.n_levels_ = np.argmax(self.level_scores_) + 1
        self.trim_index_ = self.n_levels_
        if self.inter_level_sampler is None:
            self.trim_index_ *= 2

        self.steps[self.trim_index_ + 1 : -1] = []

        if self.verbose:
            print(
                f"Trimmed to {self.n_levels_} levels."
                f" Level scores: {self.level_scores_}"
            )

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
