# TODO: use metadata routing on predict, fit_transform, etc.
from abc import ABCMeta, abstractmethod
from numbers import Integral, Number, Real
from typing import Self
from warnings import warn

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
from sklearn.utils import Bunch
from sklearn.utils._estimator_html_repr import _VisualBlock
from sklearn.utils._tags import _safe_tags
from sklearn.utils._param_validation import HasMethods, Interval, StrOptions, Hidden
from sklearn.utils.metaestimators import available_if, _BaseComposition
from sklearn.pipeline import (
    _fit_transform_one,
    _final_estimator_has,
    _print_elapsed_time,
)
from sklearn.utils._metadata_requests import (
    MetadataRouter,
    MethodMapping,
    process_routing,
    _routing_enabled,
    _raise_for_params,
    METHODS,
)
from sklearn.compose import ColumnTransformer
from imblearn.pipeline import Pipeline, _fit_resample_one, check_memory


class BaseLevel(TransformerMixin, BaseEstimator, metaclass=ABCMeta):
    """Base class for levels in a cascade.
    """
    @abstractmethod
    def fit(self, X, y, last_level: Self | None = None, **fit_params):
        pass

    @abstractmethod
    def transform(self, X, y=None):
        pass


class AlternatingLevel(BaseLevel, _BaseComposition):
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
        "last_level": [BaseLevel, None]
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
        last_level: Self | None = None
    ):
        self.transformers = transformers
        self.sparse_threshold = sparse_threshold
        self.n_jobs = n_jobs
        self.transformer_weights = transformer_weights
        self.verbose = verbose
        self.verbose_feature_names_out = verbose_feature_names_out
        self.last_level = last_level
    
    def get_params(self, deep=True):
        return self._get_params("transformers", deep=deep)

    def set_params(self, **kwargs):
        return self._set_params("transformers", **kwargs)
    
    @property
    def output_indices_(self):
        check_is_fitted(self)
        return self.column_transformer_.output_indices_

    @property
    def transformers_(self):
        check_is_fitted(self)
        return self.column_transformer_.transformers_
    
    def _get_column_indices(self):
        if self.last_level_ is None:
            # FIXME: sklearn error? slice(None) should also work.
            # return [slice(None)] * len(self.transformers)
            return [np.ones(self.n_features_in_, dtype=bool)] * len(self.transformers)

        last_output_indices = self.last_level_.output_indices_

        # Make each transformer in the current level unaware of its own outputs
        # of the last level.
        slices = []
        for name, _ in self.transformers:
            new_slice = np.ones(self.n_features_in_, dtype=bool)
            new_slice[last_output_indices[name]] = False
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
    # TODO: configure last_level as metadata routing
    def fit(self, X, y=None, last_level=None, **fit_params):
        self._validate_params()
        self._check_n_features(X, reset=True)
        self.last_level_ = self.last_level or last_level
        self.column_transformer_ = (
            self._make_column_transformer().fit(X, y, **fit_params)
        )
        return self

    @_fit_context(
        # Transformers are not validated yet
        prefer_skip_nested_validation=False,
    )
    def fit_transform(self, X, y=None, last_level=None, **fit_params):
        self._validate_params()
        self._check_n_features(X, reset=True)
        self.last_level_ = self.last_level or last_level
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


# TODO: Nested SequentialLevel still not working.
class SequentialLevel(Pipeline, BaseLevel):
    def __init__(
        self, steps, *, memory=None, verbose=False, last_level: Self | None = None,
    ):
        self.steps = steps
        self.memory = memory
        self.verbose = verbose
        self.last_level = last_level
    
    def transform(self, X, **params):
        """Transform the data with transformer steps.

        This class simply enables a Pipeline with a sampler (or predictor) as
        final estimator to still be used as a transformer, by calling
        `transform` of each transformer in the pipeline. If also a transformer,
        the final estimator's `transform` method will also be called, but this
        method remains valid even otherwise (unlike the original
        imblearn.Pipeline.transform).
        
        While this would be equivalent to the concept of setting the final
        estimator to `None` or `"passthrough"`, the original
        `imblearn.Pipeline.fit_resample()` does not returns the resampled `y` in
        this case, which hinders its application with `Cascades`. 

        Parameters
        ----------
        X : iterable
            Data to transform. Must fulfill input requirements of first step
            of the pipeline.

        **params : dict of str -> object
            Parameters requested and accepted by steps. Each step must have
            requested certain metadata for these parameters to be forwarded to
            them.

            Only available if `enable_metadata_routing=True`. See
            :ref:`Metadata Routing User Guide <metadata_routing>` for more
            details.

        Returns
        -------
        Xt : ndarray of shape (n_samples, n_transformed_features)
            Transformed data.
        """
        _raise_for_params(params, self, "transform")

        # not branching here since params is only available if
        # enable_metadata_routing=True
        routed_params = process_routing(self, "transform", **params)
        Xt = X
        for _, name, transform in self._iter():
            Xt = transform.transform(Xt, **routed_params[name].transform)
        return Xt


class Cascade(Pipeline):
    _parameter_constraints = {
        "level": [
            HasMethods(["fit", "transform"]),
            HasMethods(["fit_transform", "transform"]),
            HasMethods(["fit_resample"]),
        ],
        "final_estimator": [
            HasMethods(["fit"]),
            StrOptions({"passthrough"}),
            None,
        ],
        "memory": [None, str, HasMethods(["cache"])],
        "verbose": ["verbose"],
        "max_levels": [Interval(Integral, 0, None, closed="left")],
        "keep_original_features": ["boolean"],
        "warm_start": ["boolean"],
    }

    def __init__(
        self,
        level,
        final_estimator,
        *,
        memory=None,
        verbose=False,
        max_levels=10,
        keep_original_features=True,
        warm_start=False,
    ):
        self.final_estimator = final_estimator
        self.level = level
        self.memory = memory
        self.verbose = verbose
        self.max_levels = max_levels
        self.keep_original_features = keep_original_features
        self.warm_start = warm_start

    # def _more_tags(self):
    #     return {"pairwise": _safe_tags(self.final_estimator, "pairwise")}
    
    def _more_tags(self):
        tags = {
            "_xfail_checks": {
                "check_dont_overwrite_parameters": (
                    "Pipeline changes the `steps` parameter, which it shouldn't."
                    "Therefore this test is x-fail until we fix this."
                ),
                "check_estimators_overwrite_params": (
                    "Pipeline changes the `steps` parameter, which it shouldn't."
                    "Therefore this test is x-fail until we fix this."
                ),
            }
        }

        try:
            tags["pairwise"] = _safe_tags(self.level, "pairwise")
        except (ValueError, AttributeError, TypeError):
            # This happens when the `steps` is not a list of (name, estimator)
            # tuples and `fit` is not called yet to validate the steps.
            pass

        try:
            tags["multioutput"] = _safe_tags(self.final_estimator, "multioutput")
        except (ValueError, AttributeError, TypeError):
            # This happens when the `steps` is not a list of (name, estimator)
            # tuples and `fit` is not called yet to validate the steps.
            pass

        return tags

    def _combine_features(self, *, new_X, original_X):
        if self.keep_original_features:
            Xs = [new_X, original_X]
            if any(sparse.issparse(f) for f in Xs):
                return sparse.hstack(Xs, format="csr")
            return np.hstack(Xs)
        return new_X

    def _log_message(self, step_idx=None, step_name=None, message=None):
        if not self.verbose:
            return None
        if message is not None:
            return message
        elif step_name == "final_estimator":
            return "Fitting final estimator"
        elif step_name == "level":
            return (
                f"(level {step_idx + 1} of {self.max_levels}) "
                "Fitting level transformer"
            )
        raise ValueError(f"Unknown step name: {step_name}")

    @property
    def _estimator_type(self):
        return self.final_estimator_._estimator_type

    def _validate_names(self, names):
        """Skip name validation, since names are generated automatically."""
        pass

    @property
    def steps(self):  # Compatibility with Pipeline
        # HACK: compatibility for Pipeline to get _more_tags
        if not hasattr(self, "levels_"):  # Not yet fitted
            return [
                ("level", self.level),
                ("final_estimator", self.final_estimator),
            ]
        return (
            [(f"level{i + 1}", level) for i, level in enumerate(self.levels_)]
            + [("final_estimator", self.final_estimator_)]
        )

    @property
    def _final_estimator(self):  # Compatibility with Pipeline
        # check_is_fitted(self)
        return self.final_estimator_

    def get_metadata_routing(self):
        """Get metadata routing of this object.

        Please check :ref:`User Guide <metadata_routing>` on how the routing
        mechanism works.

        Returns
        -------
        routing : MetadataRouter
            A :class:`~utils.metadata_routing.MetadataRouter` encapsulating
            routing information.
        """
        router = MetadataRouter(owner=self.__class__.__name__)
        method_mapping = MethodMapping()

        # fit, fit_predict, and fit_transform call fit_transform if it
        # exists, or else fit and transform
        if hasattr(self.level, "fit_transform"):
            (
                method_mapping.add(caller="fit", callee="fit_transform")
                .add(caller="fit_transform", callee="fit_transform")
                .add(caller="fit_predict", callee="fit_transform")
                .add(caller="fit_resample", callee="fit_transform")
            )
        else:
            (
                method_mapping.add(caller="fit", callee="fit")
                .add(caller="fit", callee="transform")
                .add(caller="fit_transform", callee="fit")
                .add(caller="fit_transform", callee="transform")
                .add(caller="fit_predict", callee="fit")
                .add(caller="fit_predict", callee="transform")
                .add(caller="fit_resample", callee="fit")
                .add(caller="fit_resample", callee="transform")
            )

        (
            method_mapping.add(caller="predict", callee="transform")
            .add(caller="predict", callee="transform")
            .add(caller="predict_proba", callee="transform")
            .add(caller="decision_function", callee="transform")
            .add(caller="predict_log_proba", callee="transform")
            .add(caller="transform", callee="transform")
            .add(caller="inverse_transform", callee="inverse_transform")
            .add(caller="score", callee="transform")
            .add(caller="fit_resample", callee="transform")
        )

        router.add(method_mapping=method_mapping, level=self.level)

        if self.final_estimator is None or self.final_estimator == "passthrough":
            return router

        # then we add the last step
        method_mapping = MethodMapping()

        if hasattr(self.final_estimator, "fit_transform"):
            (
                method_mapping.add(caller="fit_transform", callee="fit_transform").add(
                    caller="fit_resample", callee="fit_transform"
                )
            )
        else:
            (
                method_mapping.add(caller="fit", callee="fit")
                .add(caller="fit", callee="transform")
                .add(caller="fit_resample", callee="fit")
                .add(caller="fit_resample", callee="transform")
            )
        (
            method_mapping.add(caller="fit", callee="fit")
            .add(caller="predict", callee="predict")
            .add(caller="fit_predict", callee="fit_predict")
            .add(caller="predict_proba", callee="predict_proba")
            .add(caller="decision_function", callee="decision_function")
            .add(caller="predict_log_proba", callee="predict_log_proba")
            .add(caller="transform", callee="transform")
            .add(caller="inverse_transform", callee="inverse_transform")
            .add(caller="score", callee="score")
            .add(caller="fit_resample", callee="fit_resample")
        )

        router.add(
            method_mapping=method_mapping,
            final_estimator=self.final_estimator,
        )
        return router

    @_fit_context(
        # Level and final estimators are not validated yet
        prefer_skip_nested_validation=False,
    )
    def fit(self, X, y=None, **params):
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
        routed_params = self._check_method_params(method="fit", props=params)
        Xt, yt = self._fit(X, y, routed_params)

        with _print_elapsed_time(
            self.__class__.__name__,
            self._log_message(None, "final_estimator"),
        ):
            if self.final_estimator != "passthrough":
                last_step_params = routed_params["final_estimator"]
                self.final_estimator_ = clone(self.final_estimator)
                self.final_estimator_.fit(Xt, yt, **last_step_params["fit"])

        return self

    def _fit(self, X, y=None, routed_params=None):
        memory = check_memory(self.memory)

        if hasattr(self.level, "fit_resample"):
            fit_transform_or_resample_one_cached = memory.cache(_fit_resample_one)
        elif hasattr(self.level, "transform") or hasattr(self.level, "fit_transform"):
            fit_transform_or_resample_one_cached = memory.cache(_fit_transform_one)
        else:
            raise RuntimeError  # Should not happen given correct param validation.

        if not self.warm_start or not hasattr(self, "levels_"):
            # Free allocated memory, if any
            self.levels_ = []

        n_more_levels = self.max_levels - len(self.levels_)

        if n_more_levels < 0:
            raise ValueError(
                "max_levels=%d must be larger or equal to "
                "len(levels_)=%d when warm_start==True"
                % (self.max_levels, len(self.levels_))
            )
        elif n_more_levels == 0:
            if self.max_levels == 0:
                warn("max_levels=0, so only the final estimator will be fit.")
            else:  # max_levels == len(levels_)
                warn(
                    "Warm-start fitting without increasing max_levels does not "
                    "fit new levels."
                )

        if self.levels_:  # Warm-start
            # FIXME: it seems cumbersome to store the last y_resampled_
            # while disregarding the current y, but the alternative would be
            # to fit all samplers again.
            Xt, y = self._apply_transformers(X), self.last_y_resampled_
        else:
            Xt = X

        for _ in range(n_more_levels):
            level = clone(self.level)
            n_levels = len(self.levels_)

            # Set the last_level param for objects that support it.
            if self.levels_:
                # TODO: use metadata routing and fit params instead of estimator
                # __init__ parameters.
                last_level_params = self.levels_[-1].get_params(deep=True)
                level.set_params(
                    **{
                        param: last_level_params[param.removesuffix("__last_level")]
                        for param in level.get_params().keys()
                        if param.endswith("__last_level")
                    }
                )
                if hasattr(level, "last_level"):
                    level.set_params(last_level=self.levels_[-1])

            # Fit or load from cache the current level
            result = fit_transform_or_resample_one_cached(
                level,
                Xt,
                y,
                message_clsname=self.__class__.__name__,
                message=self._log_message(n_levels, "level"),
                params=routed_params["level"],
                **({"weight": None} if not hasattr(level, "fit_resample") else {}),
            )
            if hasattr(self.level, "fit_resample"):
                new_X, y, fitted_level = result
            else:
                new_X, fitted_level = result

            self.levels_.append(fitted_level)
            Xt = self._combine_features(original_X=X, new_X=new_X)
        
        if self.warm_start:
            self.last_y_resampled_ = y

        return Xt, y

    def _apply_transformers(self, X):
        Xt = X
        for _, name, transform in self._iter(with_final=False, filter_resample=False):
            # NOTE: we set filter_resample=False to enable calling transforms 
            # when self.level is a SequentialLevel object (or imblearn Pipeline with last
            # estimator set to "passthrough"). Otherwise, the step would be skipped
            # since it has a "fit_resample" method, even though it can also
            # transform.
            if hasattr(transform, "transform"):
                Xt = self._combine_features(
                    original_X=X, new_X=transform.transform(Xt),
                )
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

        Returns
        -------
        y_pred : ndarray
            Result of calling `predict` on the final estimator.
        """
        Xt = self._apply_transformers(X)
        return self.final_estimator_.predict(Xt, **predict_params)

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
        return self.final_estimator_.predict_proba(Xt, **predict_proba_params)

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
        return self.final_estimator_.decision_function(Xt)

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
        return self.final_estimator_.score_samples(Xt)

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
        return self.final_estimator_.predict_log_proba(Xt, **predict_log_proba_params)

    def _can_transform(self):
        return self.final_estimator_ == "passthrough" or hasattr(
            self.final_estimator_, "transform"
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
        return self.final_estimator_.transform(Xt)

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
        return self.final_estimator_.score(Xt, y, **score_params)

    def _check_method_params(self, method, props, **kwargs):
        if _routing_enabled():
            routed_params = process_routing(self, method, **props, **kwargs)
            return routed_params
        else:
            fit_params_steps = Bunch(
                level=Bunch(**{method: {} for method in METHODS}),
                final_estimator=Bunch(**{method: {} for method in METHODS}),
            )
            for pname, pval in props.items():
                if "__" not in pname:
                    raise ValueError(
                        "Pipeline.fit does not accept the {} parameter. "
                        "You can pass parameters to specific steps of your "
                        "pipeline using the stepname__parameter format, e.g. "
                        "`Pipeline.fit(X, y, logisticregression__sample_weight"
                        "=sample_weight)`.".format(pname)
                    )
                step, param = pname.split("__", 1)
                fit_params_steps[step]["fit"][param] = pval
                # without metadata routing, fit_transform and fit_predict
                # get all the same params and pass it to the last fit.
                fit_params_steps[step]["fit_transform"][param] = pval
                fit_params_steps[step]["fit_predict"][param] = pval
            return fit_params_steps
