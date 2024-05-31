"""Employ tree embeddings and weak-label inputting in deep forest models.
"""
import os
from multiprocessing import cpu_count

import scipy.sparse
from sklearn.base import (
    BaseEstimator,
    TransformerMixin,
    clone,
)
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    RandomForestRegressor,
    ExtraTreesRegressor,
)

from . import label_complement
from .cascade_learn.tree_embedder import ForestEmbedder
from .cascade_learn.cascade import Cascade, AlternatingLevel, SequentialLevel
from .cascade_learn import weak_labels
from .cascade_learn.estimator_adapters import (
    RegressorAsBinaryClassifier,
    EstimatorAsTransformer,
    MultiOutputVotingRegressor,
    UnanimityClassifier,
)


RSTATE = 0  # check_random_state(0)
NJOBS = (cpu_count() // 5) or 1  # 1/5 of the available cores, for each CV fold
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


class Densifier(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if scipy.sparse.issparse(X):
            return X.toarray()
        return X


rf_embedder = (
    ForestEmbedder(
        RandomForestRegressor(
            **FOREST_PARAMS,
            max_samples=MAX_EMBEDDING_SAMPLES,
            bootstrap=True,  # Default for RF
        ),
        method="path",
        node_weights="log_node_size",  # Eq. (1)
        max_node_size=MAX_NODE_SIZE,
    )
)

xt_embedder = (
    ForestEmbedder(
        ExtraTreesRegressor(
            **FOREST_PARAMS,
            max_samples=MAX_EMBEDDING_SAMPLES,
            bootstrap=True,
        ),
        method="path",
        node_weights="log_node_size",  # Eq. (1)
        max_node_size=0.8,
    )
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

cafe = clone(Cascade(
    level=alternating_level_embedding,
    final_estimator=final_estimator,
    **CASCADE_PARAMS,
))

cafe_os = clone(Cascade(
    level=alternating_level_embedding_proba,
    final_estimator=final_estimator,
    **CASCADE_PARAMS,
))

slcforest = clone(Cascade(
    level=SequentialLevel([
        ("alternating_forests", alternating_level_proba),
        ("label_imputer", lc_imputer),
    ]),
    final_estimator=final_estimator,
    **CASCADE_PARAMS,
))

cafe_slc = clone(Cascade(
    level=SequentialLevel([
        ("alternating_forests", alternating_level_embedding_proba),
        ("label_imputer", lc_imputer),
    ]),
    final_estimator=final_estimator,
    **CASCADE_PARAMS,
))

flaforest = clone(Cascade(
    level=SequentialLevel([
        ("alternating_forests", alternating_level_proba),
        ("label_imputer", scar_imputer),
    ]),
    final_estimator=final_estimator,
    **CASCADE_PARAMS,
))

cafe_fla = clone(Cascade(
    level=SequentialLevel([
        ("alternating_forests", alternating_level_embedding_proba),
        ("label_imputer", scar_imputer),
    ]),
    final_estimator=final_estimator,
    **CASCADE_PARAMS,
))

lcforest = clone(Cascade(
    level=SequentialLevel([
        # ("transformer", zhou_level),
        ("transformer", fixed_level_proba),
        ("label_imputer", wang_imputer),
    ]),
    final_estimator=final_estimator,
    **CASCADE_PARAMS,
))

gcforest = clone(Cascade(
    # level=zhou_level,
    level=fixed_level_proba,
    final_estimator=final_estimator,
    **CASCADE_PARAMS,
))

estimators_dict = {
    "lcforest": lcforest,
    "gcforest": gcforest,
    "cafe": cafe,
    "cafe_os": cafe_os,
    "cafe_slc": cafe_slc,
    "flaforest": flaforest,
    "cafe_fla": cafe_fla,
    "final_estimator": final_estimator,
}

# TODO: make test from it
# if __name__ == "__main__":
#     # X, y, _, _ = load_dataset("mediamill", "undivided")
#     X, y, _, _ = load_dataset("emotions", "undivided")
#     # X, y = load_iris(return_X_y=True)
#     # X, y, _, _ = load_dataset("yeast", "undivided")
#     X, y = X.toarray(), y.toarray()
#     y = np.atleast_2d(y)
#     # y[:, 0] = 0  # Simulate missing label
#     # breakpoint()
#     # cascade = clone(cafe_fla).set_params(
#     cascade = clone(cafe_slc).set_params(
#         max_levels=3,
#         memory=None,
#     )
#     cascade.set_params(**{
#         k: 1 for k, v in cascade.get_params().items()
#         if k.endswith("n_jobs")
#     })
#
#     cascade = ImblearnPipeline([
#         ("dropper", PositiveDropper(0.25)),
#         ("cascade", cascade),
#     ])
#     # cascade = clone(final_estimator)
#     # cascade = positive_dropper.wrap_estimator(
#     #     cascade_weak_label_proba,
#     #     drop=0.25,
#     #     random_state=RSTATE,
#     # )
#     cascade = cascade.fit(X, y)
#
#     # scorers = scoring.level_scorers.keys()  # All scorers.
#     scorers = [
#         "label_ranking_average_precision_score",
#         "average_precision_micro",
#         "tp_micro_oob",
#         "tp_micro",
#         "tp_micro_masked",
#         "precision_micro",
#         "precision_micro_masked",
#         "precision_macro_masked",
#         "precision_weighted_masked",
#         "precision_samples_masked",
#     ]
#     for scorer_name in scorers:
#         scorer = scoring.all_scorers[scorer_name]
#         print(scorer_name, sklearn.metrics.check_scoring(cascade, scorer)(cascade, X, y))
#
#     joblib.dump(cascade, "cascade.joblib")
#     with open("cascade.html", "w") as f:
#         f.write(estimator_html_repr(cascade))
#