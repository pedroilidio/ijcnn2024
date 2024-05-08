"""Employ tree embeddings and weak-label inputting in deep forest models.
"""
import copy
from numbers import Real, Integral
import functools
import os

import scipy.sparse
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
from imblearn.pipeline import Pipeline as ImblearnPipeline
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
    GridSearchCV,
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
from deep_forest.cascade import Cascade, AlternatingLevel, SequentialLevel
from deep_forest import weak_labels
from deep_forest.estimator_adapters import (
    ProbaTransformer,
    RegressorAsBinaryClassifier,
    EstimatorAsTransformer,
    MultiOutputVotingClassifier,
    MultiOutputVotingRegressor,
    TreeEmbedderWithOutput,
    CVClassifier,
    CVRegressor,
    UnanimityClassifier,
)

from nakano_datasets_v2 import scoring
from positive_dropper import PositiveDropper
import label_complement


RSTATE = 0  # check_random_state(0)
NJOBS = 14
# MEMORY = joblib.Memory(location="cache", verbose=10)
MEMORY = None
# NOTE: the paper undersamples for the whole forest, we perform undersampling
# for each tree (NOW FIXED).
MAX_EMBEDDING_SAMPLES = 0.5
# Maximum fraction of samples in a tree node for it to be used in the embeddings
MAX_NODE_SIZE = 0.95
N_COMPONENTS = 0.8
# N_COMPONENTS = "mle"  # Use Minka's (2000) MLE to determine the number of components
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
    memory=MEMORY,
    keep_original_features=True,
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


class Densifier(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        if scipy.sparse.issparse(X):
            return X.toarray()
        return X


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

xt_embedder_pca = Pipeline([
    ("embedder", xt_embedder),
    ("densifier", Densifier()),
    ("pca", PCA(n_components=N_COMPONENTS, random_state=RSTATE)),
])

rf_embedder_pca = Pipeline([
    ("embedder", rf_embedder),
    ("densifier", Densifier()),
    ("pca", PCA(n_components=N_COMPONENTS, random_state=RSTATE)),
])

xt_proba_transformer = EstimatorAsTransformer(
    ExtraTreesRegressor(
        **FOREST_PARAMS,
        # oob_score=True,  # Only necessary for final estimator
        max_samples=0.5,
        bootstrap=True,  # Default for RF
    ),
)
rf_proba_transformer = EstimatorAsTransformer(
    RandomForestRegressor(
        **FOREST_PARAMS,
        # oob_score=True,  # Only necessary for final estimator
        max_samples=0.5,
        bootstrap=True,
    ),
)

final_estimator = RegressorAsBinaryClassifier(
    MultiOutputVotingRegressor(
        estimators=[
            (
                "rf",
                RandomForestRegressor(
                    **FOREST_PARAMS,
                    oob_score=True,
                    max_samples=0.5,
                    bootstrap=True,  # Default for RF
                ),
            ),
            (
                "xt",
                ExtraTreesRegressor(
                    **FOREST_PARAMS,
                    oob_score=True,
                    max_samples=0.5,
                    bootstrap=True,
                ),
            ),
        ],
    )
)


# Copies final_estimator.
fixed_level_proba = FeatureUnion([
    ("rf", rf_proba_transformer),
    ("xt", xt_proba_transformer),
])

alternating_level_embedding = AlternatingLevel([
    ("rf", rf_embedder_pca),
    ("xt", xt_embedder_pca),
])

alternating_level_proba = AlternatingLevel([
    ("xt", xt_proba_transformer),
    ("rf", rf_proba_transformer),
])

alternating_level_embedding_proba = AlternatingLevel([
    ("rf", FeatureUnion([("embedder", rf_embedder_pca), ("proba", rf_proba_transformer)])),
    ("xt", FeatureUnion([("embedder", xt_embedder_pca), ("proba", xt_proba_transformer)])),
])

# TODO: faster but different from Nakano et al. (2023)
# alternating_level_embedding_proba = AlternatingLevel([
#     ("xt", TreeEmbedderWithOutput(
#             xt_embedder,
#             post_transformer=Pipeline([
#                 ("densifier", Densifier()),
#                 ("pca", PCA(n_components=N_COMPONENTS, random_state=RSTATE)),
#             ]),
#         ),
#     ),
#     ("rf", TreeEmbedderWithOutput(
#             rf_embedder,
#             post_transformer=Pipeline([
#                 ("densifier", Densifier()),
#                 ("pca", PCA(n_components=N_COMPONENTS, random_state=RSTATE)),
#             ]),
#         ),
#     ),
# ])

imputer_estimator = MultiOutputVotingRegressor(
    estimators=[
        (
            "rf",
            RandomForestRegressor(
                **FOREST_PARAMS,
                oob_score=True,
                max_samples=0.5,
                bootstrap=True,  # Default for RF
            ),
        ),
        (
            "xt",
            ExtraTreesRegressor(
                **FOREST_PARAMS,
                oob_score=True,
                max_samples=0.5,
                bootstrap=True,
            ),
        ),
    ],
)

scar_imputer = weak_labels.SCARImputer(
    label_freq_percentile=0.95,
    verbose=True,
    estimator=imputer_estimator,
)

lc_imputer = weak_labels.LabelComplementImputer(
    label_freq_percentile=0.95,
    verbose=True,
    estimator=imputer_estimator,
    weight_proba=False,
)

wang_imputer = label_complement.LabelComplementImputer(
    estimator=UnanimityClassifier(
        estimators=[
            ("rf", RandomForestClassifier(**FOREST_PARAMS)),
            ("et", ExtraTreesClassifier(**FOREST_PARAMS)),
        ],
        threshold=0.4,
    ),
    verbose=True,
    tice_params=dict(max_bepp=5, max_splits=500, min_set_size=5),  # random_state=RSTATE),
    cv_params=dict(cv=5),
)

# zhou_level = FeatureUnion(  # Too slow.
#     [
#         ("rf", EstimatorAsTransformer(CVRegressor(RandomForestRegressor(**FOREST_PARAMS), cv=5))),
#         ("xt", EstimatorAsTransformer(CVRegressor(ExtraTreesRegressor(**FOREST_PARAMS), cv=5))),
#     ]
# )


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

cascade_tree_embedder_chi2 = clone(cascade_tree_embedder).set_params(
    level__rf__embedder__max_pvalue=0.05,
    level__xt__embedder__max_pvalue=0.05,
)

cascade_tree_embedder_proba = clone(Cascade(
    level=alternating_level_embedding_proba,
    final_estimator=final_estimator,
    **CASCADE_PARAMS,
))

# TODO: Change name "alternating_forests"
cascade_lc_proba = clone(Cascade(
    level=SequentialLevel([
        ("alternating_forests", alternating_level_proba),
        ("label_imputer", lc_imputer),
    ]),
    final_estimator=final_estimator,
    **CASCADE_PARAMS,
))

cascade_lc_tree_embedder = clone(Cascade(
    level=SequentialLevel([
        ("alternating_forests", alternating_level_embedding),
        ("label_imputer", lc_imputer),
    ]),
    final_estimator=final_estimator,
    **CASCADE_PARAMS,
))

cascade_lc_tree_embedder_proba = clone(Cascade(
    level=SequentialLevel([
        ("alternating_forests", alternating_level_embedding_proba),
        ("label_imputer", lc_imputer),
    ]),
    final_estimator=final_estimator,
    **CASCADE_PARAMS,
))

cascade_lc_tree_embedder_chi2 = clone(
    cascade_lc_tree_embedder,
).set_params(
    level__alternating_forests__rf__embedder__max_pvalue=0.05,
    level__alternating_forests__xt__embedder__max_pvalue=0.05,
)

cascade_scar_proba = clone(Cascade(
    level=SequentialLevel([
        ("alternating_forests", alternating_level_proba),
        ("label_imputer", scar_imputer),
    ]),
    final_estimator=final_estimator,
    **CASCADE_PARAMS,
))

cascade_scar_tree_embedder = clone(Cascade(
    level=SequentialLevel([
        ("alternating_forests", alternating_level_embedding),
        ("label_imputer", scar_imputer),
    ]),
    final_estimator=final_estimator,
    **CASCADE_PARAMS,
))

cascade_scar_tree_embedder_proba = clone(Cascade(
    level=SequentialLevel([
        ("alternating_forests", alternating_level_embedding_proba),
        ("label_imputer", scar_imputer),
    ]),
    final_estimator=final_estimator,
    **CASCADE_PARAMS,
))

cascade_scar_tree_embedder_chi2 = clone(
    cascade_scar_tree_embedder,
).set_params(
    level__alternating_forests__rf__embedder__max_pvalue=0.05,
    level__alternating_forests__xt__embedder__max_pvalue=0.05,
)

cascade_wang = clone(Cascade(
    level=SequentialLevel([
        # ("transformer", zhou_level),
        ("transformer", fixed_level_proba),
        ("label_imputer", wang_imputer),
    ]),
    final_estimator=final_estimator,
    **CASCADE_PARAMS,
))

cascade_zhou = clone(Cascade(
    # level=zhou_level,
    level=fixed_level_proba,
    final_estimator=final_estimator,
    **CASCADE_PARAMS,
))

estimators_dict = {
    "cascade_wang": cascade_proba,
    "cascade_proba": cascade_proba,
    "cascade_tree_embedder": cascade_tree_embedder,
    "cascade_tree_embedder_chi2": cascade_tree_embedder_chi2,
    "cascade_tree_embedder_proba": cascade_tree_embedder_proba,
    "cascade_lc_proba": cascade_lc_proba,
    "cascade_lc_tree_embedder": cascade_lc_tree_embedder,
    "cascade_lc_tree_embedder_proba": cascade_lc_tree_embedder,
    "cascade_lc_tree_embedder_chi2": cascade_lc_tree_embedder_chi2,
    "cascade_scar_proba": cascade_scar_proba,
    "cascade_scar_tree_embedder": cascade_scar_tree_embedder,
    "cascade_scar_tree_embedder_proba": cascade_scar_tree_embedder,
    "cascade_scar_tree_embedder_chi2": cascade_scar_tree_embedder_chi2,
}

if __name__ == "__main__":
    # X, y, _, _ = load_dataset("mediamill", "undivided")
    X, y, _, _ = load_dataset("emotions", "undivided")
    # X, y = load_iris(return_X_y=True)
    # X, y, _, _ = load_dataset("yeast", "undivided")
    X, y = X.toarray(), y.toarray()
    y = np.atleast_2d(y)
    # y[:, 0] = 0  # Simulate missing label
    # breakpoint()
    # cascade = clone(cascade_scar_tree_embedder_proba).set_params(
    cascade = clone(cascade_lc_tree_embedder_proba).set_params(
        max_levels=3,
        memory=None,
    )
    cascade.set_params(**{
        k: 1 for k, v in cascade.get_params().items()
        if k.endswith("n_jobs")
    })

    cascade = ImblearnPipeline([
        ("dropper", PositiveDropper(0.25)),
        ("cascade", cascade),
    ])
    # cascade = clone(final_estimator)
    # cascade = positive_dropper.wrap_estimator(
    #     cascade_weak_label_proba,
    #     drop=0.25,
    #     random_state=RSTATE,
    # )
    cascade = cascade.fit(X, y)

    # scorers = scoring.level_scorers.keys()  # All scorers.
    scorers = [
        "label_ranking_average_precision_score",
        "average_precision_micro",
        "tp_micro_oob",
        "tp_micro",
        "tp_micro_masked",
        "precision_micro",
        "precision_micro_masked",
        "precision_macro_masked",
        "precision_weighted_masked",
        "precision_samples_masked",
    ]
    for scorer_name in scorers:
        scorer = scoring.all_scorers[scorer_name]
        print(scorer_name, sklearn.metrics.check_scoring(cascade, scorer)(cascade, X, y))

    joblib.dump(cascade, "cascade.joblib")
    with open("cascade.html", "w") as f:
        f.write(estimator_html_repr(cascade))
