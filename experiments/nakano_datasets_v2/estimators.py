"""Employ tree embeddings and weak-label inputting in deep forest models.
"""
import copy
from numbers import Real, Integral
import functools
import os

import joblib
import numpy as np
import sklearn.metrics
from sklearn.base import (
    BaseEstimator,
    MetaEstimatorMixin,
    TransformerMixin,
    clone,
    _fit_context,
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
from skmultilearn.dataset import load_dataset
from skmultilearn.model_selection import IterativeStratification

from deep_forest.tree_embedder import ForestEmbedder
from deep_forest.cascade import Cascade, AlternatingLevel
from deep_forest.weak_labels import WeakLabelImputer, PositiveUnlabeledImputer
from deep_forest.estimator_adapters import (
    ProbaTransformer,
    EstimatorAsTransformer,
    MultiOutputVotingClassifier,
    MultiOutputVotingRegressor,
    TreeEmbedderWithOutput,
)
from nakano_datasets_v2 import scoring


average_precision_micro_oob_scorer = scoring.level_scorers["average_precision_micro_oob"]

RSTATE = 0  # check_random_state(0)
NJOBS = 14
MEMORY = joblib.Memory(location="cache", verbose=10)
# NOTE: the paper undersamples for the whole forest, we perform undersampling
# for each tree (NOW FIXED).
MAX_EMBEDDING_SAMPLES = 0.5
# Maximum fraction of samples in a tree node for it to be used in the embeddings
MAX_NODE_SIZE = 0.95
N_COMPONENTS = 0.8
VERBOSE = 10
FOREST_PARAMS = dict(
    n_estimators=150,
    max_features="sqrt",
    min_samples_leaf=5,
    max_depth=None,
    n_jobs=NJOBS,
    verbose=True,
    random_state=RSTATE,
)
CASCADE_PARAMS = dict(
    max_levels=10,
    verbose=VERBOSE,
    random_state=RSTATE,
    memory=MEMORY,
    # scoring="mean_squared_error",
    scoring=average_precision_micro_oob_scorer,
    validation_size="train",
    trim_to_best_score=True,
)

for var in [
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "BLIS_NUM_THREADS",
]:
    value = NJOBS
    print(f"Setting environment variable {var}=\"{value}\"")
    os.environ[var] = str(value)


def make_iterative_stratification(**kwargs):
    """CV factory to use in YAML config files."""
    # NOTE: simply calling __init__ will not work, PyYAML expects module-level
    # functions.
    return IterativeStratification(**kwargs)


class Densifier(
    BaseEstimator,
    TransformerMixin,
):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X.toarray()


class UndersampledTransformer(
    BaseEstimator,
    MetaEstimatorMixin,
):
    _parameter_constraints = {
        "estimator": [HasMethods(["fit", "transform"])],
        "n_samples": [
            Interval(Real, 0.0, 1.0, closed="neither"),
            Interval(Integral, 1, None, closed="left"),
        ],
        "random_state": ["random_state"],
    }
    def __init__(self, estimator, n_samples, random_state=None) -> None:
        self.estimator = estimator
        self.n_samples = n_samples
        self.random_state = random_state

    @_fit_context(prefer_skip_nested_validation=False)
    def fit(self, X, y):
        self._validate_params()

        if isinstance(self.n_samples, float):
            self.n_samples_ = int(np.ceil(X.shape[0]) * self.n_samples)

        self.estimator_ = clone(self.estimator)
        random_state = check_random_state(self.random_state)

        indices_to_keep = random_state.choice(X.shape[0], self.n_samples_, replace=False)
        self.estimator_.fit(X[indices_to_keep], y[indices_to_keep])
        return self
    
    def transform(self, X, y=None):
        return self.estimator_.transform(X)


# [...] we discard the nodes in which are present in more than [max_node_size%]
# of the instances, removing non-meaningful nodes, such as the ones located
# close to the root and the root itself.

# [...] the remaining nodes are weighted according to Eq. (1) where nodes that
# contain many instances are credited a lower weight, and the opposite is valid
# for more representative nodes.

rf_embedder = (#UndersampledTransformer(
    ForestEmbedder(
        RandomForestRegressor(
            **FOREST_PARAMS,
            max_samples=MAX_EMBEDDING_SAMPLES,
            bootstrap=True,  # Default for RF
        ),
        method="path",
        node_weights="log_node_size",  # Eq. (1)
        max_node_size=MAX_NODE_SIZE,
    )#,
    #n_samples=MAX_EMBEDDING_SAMPLES
)

xt_embedder = (#UndersampledTransformer(
    ForestEmbedder(
        ExtraTreesRegressor(
            **FOREST_PARAMS,
            max_samples=MAX_EMBEDDING_SAMPLES,
            bootstrap=True,
        ),
        method="path",
        node_weights="log_node_size",  # Eq. (1)
        max_node_size=0.8,
    )#,
    #n_samples=MAX_EMBEDDING_SAMPLES
)


final_estimator = MultiOutputVotingClassifier(
    estimators=[
        (
            "rf",
            RandomForestClassifier(
                **FOREST_PARAMS,
                oob_score=True,
                max_samples=0.9,
                bootstrap=True,  # Default for RF
            ),
        ),
        (
            "xt",
            ExtraTreesClassifier(
                **FOREST_PARAMS,
                oob_score=True,
                max_samples=0.9,
                bootstrap=True,
            ),
        ),
    ],
)


# For the number of PCA components in our method, we have optimized the number
# of components c"sqrt"nsidering a percentage of the total number of decision path
# features{1 component, 1%, 5%, 20%, 40%, 60%, 80%, 95%} multiplied by min(N,
# |c|), being N the number of instances in the dataset and |c| the number of
# nodes in the ensemble. [we fixed that percentage to N_COMPONENTS]

alternating_level_embedding = AlternatingLevel([
    ("xt_embedder", Pipeline([
        ("xt", xt_embedder),
        ("densifier", Densifier()),
        (
            "pca",
            PCA(n_components=N_COMPONENTS, random_state=RSTATE),
            # LOBPCGTruncatedSVD(
            #     n_components=N_COMPONENTS,
            #     max_components=800,
            #     random_state=RSTATE,
            # ),
        ),
    ])),
    ("rf_embedder", Pipeline([
        ("rf", rf_embedder),
        ("densifier", Densifier()),
        (
            "pca",
            PCA(n_components=N_COMPONENTS, random_state=RSTATE),
            # LOBPCGTruncatedSVD(
            #     n_components=N_COMPONENTS,
            #     max_components=800,
            #     random_state=RSTATE,
            # ),
        ),
    ])),
])


alternating_level_proba = AlternatingLevel(
    [
        (
            "rf",
            EstimatorAsTransformer(
                RandomForestRegressor(**FOREST_PARAMS)
            ),
        ),
        (
            "xt",
            EstimatorAsTransformer(
                ExtraTreesRegressor(**FOREST_PARAMS)
            ),
        ),
    ]
)

alternating_level_embedding_proba = AlternatingLevel([
    ("xt", TreeEmbedderWithOutput(
            xt_embedder,
            post_transformer=Pipeline([
                ("densifier", Densifier()),
                ("pca", PCA(n_components=N_COMPONENTS, random_state=RSTATE)),
            ]),
            # post_transformer=PCA(n_components=N_COMPONENTS, random_state=RSTATE),
            # post_transformer=LOBPCGTruncatedSVD(
            #     n_components=N_COMPONENTS,
            #     max_components=800,
            #     random_state=RSTATE,
            # ),
        ),
    ),
    ("rf", TreeEmbedderWithOutput(
            rf_embedder,
            post_transformer=Pipeline([
                ("densifier", Densifier()),
                ("pca", PCA(n_components=N_COMPONENTS, random_state=RSTATE)),
            ]),
            # post_transformer=PCA(n_components=N_COMPONENTS, random_state=RSTATE),
            # LOBPCGTruncatedSVD(
            #     n_components=N_COMPONENTS,
            #     max_components=800,
            #     random_state=RSTATE,
            # ),
        ),
    ),
])

weak_label_imputer = PositiveUnlabeledImputer(
    ExtraTreesClassifier(  # TODO: use regressor
        **FOREST_PARAMS,
        bootstrap=True,
        max_samples=0.9,
        oob_score=True,
        # This favors positives too much when weight_proba=True
        #   class_weight="balanced",  
    ),
    # threshold=0.95,  # if max_depth=None
    threshold=0.8,
    use_oob_proba=True,
    weight_proba=True,
    verbose=True,
)

# ===========================================================================
# Cascade estimators
# ===========================================================================

cascade_proba = clone(Cascade(
    level=alternating_level_proba,
    final_estimator=final_estimator,
    **CASCADE_PARAMS,
))

cascade_tree_embedder = clone(Cascade(
    level=alternating_level_embedding,
    final_estimator=final_estimator,
    **CASCADE_PARAMS,
))

cascade_tree_embedder_pvalue = clone(cascade_tree_embedder).set_params(
    level__rf_embedder__rf__max_pvalue=0.05,
    level__xt_embedder__xt__max_pvalue=0.05,
    # level__rf_embedder__rf__estimator__max_pvalue=0.05,
    # level__xt_embedder__xt__estimator__max_pvalue=0.05,
)

cascade_tree_embedder_proba = clone(Cascade(
    level=alternating_level_embedding_proba,
    final_estimator=final_estimator,
    **CASCADE_PARAMS,
))

cascade_weak_label_proba = clone(Cascade(
    level=alternating_level_proba,
    final_estimator=final_estimator,
    inter_level_sampler=weak_label_imputer,
    **CASCADE_PARAMS,
))

cascade_weak_label_tree_embedder = clone(Cascade(
    level=alternating_level_embedding,
    final_estimator=final_estimator,
    inter_level_sampler=weak_label_imputer,
    **CASCADE_PARAMS,
))

cascade_weak_label_tree_embedder_proba = clone(Cascade(
    level=alternating_level_embedding_proba,
    final_estimator=final_estimator,
    inter_level_sampler=weak_label_imputer,
    **CASCADE_PARAMS,
))

cascade_weak_label_tree_embedder_pvalue = clone(
    cascade_weak_label_tree_embedder,
).set_params(
    level__rf_embedder__rf__max_pvalue=0.05,
    level__xt_embedder__xt__max_pvalue=0.05,
    # level__rf_embedder__rf__estimator__max_pvalue=0.05,
    # level__xt_embedder__xt__estimator__max_pvalue=0.05,
)

estimators_dict = {
    "cascade_proba": cascade_proba,
    "cascade_tree_embedder": cascade_tree_embedder,
    "cascade_tree_embedder_pvalue": cascade_tree_embedder_pvalue,
    "cascade_tree_embedder_proba": cascade_tree_embedder_proba,
    "cascade_weak_label_proba": cascade_weak_label_proba,
    "cascade_weak_label_tree_embedder": cascade_weak_label_tree_embedder,
    "cascade_weak_label_tree_embedder_proba": cascade_weak_label_tree_embedder,
    "cascade_weak_label_tree_embedder_pvalue": cascade_weak_label_tree_embedder_pvalue,
}

if __name__ == "__main__":
    # X, y, _, _ = load_dataset("mediamill", "undivided")
    X, y, _, _ = load_dataset("emotions", "undivided")
    # X, y = load_iris(return_X_y=True)
    # X, y, _, _ = load_dataset("yeast", "undivided")
    X, y = X.toarray(), y.toarray()
    # y[:, 0] = 0  # Simulate missing label
    # breakpoint()
    # cascade = clone(cascade_tree_embedder).set_params(max_levels=2)
    cascade = clone(final_estimator)
    # cascade = positive_dropper.wrap_estimator(
    #     cascade_weak_label_proba,
    #     drop=0.25,
    #     random_state=RSTATE,
    # )
    cascade = cascade.fit(X, y)
    for scoring_name, scorer in scoring.scoring_metrics.items():
        print(scoring_name, sklearn.metrics.check_scoring(cascade, scorer)(cascade, X, y))

    joblib.dump(cascade, "cascade.joblib")
    with open("cascade.html", "w") as f:
        f.write(estimator_html_repr(cascade))
