"""Employ tree embeddings and weak-label inputting in deep forest models.
"""
from numbers import Real, Integral
import joblib
import numpy as np
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
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import (
    KFold,
    RepeatedKFold,
    RepeatedStratifiedKFold,
)
from sklearn.utils import check_random_state, estimator_html_repr
from sklearn.utils._param_validation import (
    Interval,
)
from skmultilearn.dataset import load_dataset

from deep_forest.tree_embedder import (
    ForestEmbedder,
)
from deep_forest.cascade import Cascade, AlternatingLevel
from deep_forest.weak_labels import WeakLabelImputer
from deep_forest.estimator_adapters import ClassifierTransformer, MultiOutputVotingClassifier
from skmultilearn.model_selection import IterativeStratification
from _lobpcg_truncated_svd import LOBPCGTruncatedSVD
import positive_dropper

RSTATE = check_random_state(0)


def make_iterative_stratification(**kwargs):
    """CV factory to use in YAML config files."""
    # NOTE: simply calling __init__ will not work, PyYAML expects module-level
    # functions.
    return IterativeStratification(**kwargs)


rf_embedder = ForestEmbedder(
    RandomForestClassifier(
        n_estimators=100,
        max_features="sqrt",
        max_depth=10,
        random_state=RSTATE,
    ),
    method="path",
    node_weights="log_node_size",
    max_node_size=0.8,
)

xt_embedder = ForestEmbedder(
    ExtraTreesClassifier(
        n_estimators=100,
        max_features=1,
        max_depth=10,
        random_state=RSTATE,
    ),
    method="path",
    node_weights="log_node_size",
    max_node_size=0.8,
)

final_estimator = MultiOutputVotingClassifier(
    estimators=[
        (
            "rf",
            RandomForestClassifier(
                n_estimators=100,
                max_features="sqrt",
                random_state=RSTATE,
            ),
        ),
        (
            "xt",
            ExtraTreesClassifier(
                n_estimators=100,
                max_features=1,
                random_state=RSTATE,
            ),
        ),
    ],
)

# level_estimator = Pipeline(
#     [
#         (
#             "forest_embedders",
#             FeatureUnion([("xt", xt_embedder), ("rf", rf_embedder)]),
#         ),
#         (
#             "pca",
#             MySparsePCA(n_components=0.80, random_state=RSTATE),
#         ),
#     ]
# )

alternating_level_estimator = AlternatingLevel([
    ("xt_embedder", Pipeline([
        ("xt", xt_embedder),
        (
            "pca",
            # NMF(n_components=100, init="random", random_state=RSTATE, verbose=1, max_iter=5),
            # TruncatedSVD(
            LOBPCGTruncatedSVD(
                n_components=100,
                random_state=RSTATE,
            ),
        ),
    ])),
    ("rf_embedder", Pipeline([
        ("rf", rf_embedder),
        (
            "pca",
            # NMF(n_components=100, init="random", random_state=RSTATE, verbose=1, max_iter=5),
            # TruncatedSVD(
            LOBPCGTruncatedSVD(
                n_components=100,
                random_state=RSTATE,
            ),
        ),
    ])),
])

weak_label_imputer = WeakLabelImputer(
    ExtraTreesClassifier(
        n_estimators=100,
        bootstrap=True,
        max_samples=0.9,
        oob_score=True,
        random_state=RSTATE,
    ),
    threshold=0.5,
    use_oob_proba=True,
    weight_proba=True,
    verbose=True,
)

cascade_proba = Cascade(
    level=ClassifierTransformer(final_estimator),
    final_estimator=final_estimator,
    max_levels=10,
    verbose=True,
    random_state=RSTATE,
)

cascade_tree_embedder = Cascade(
    level=alternating_level_estimator,
    final_estimator=final_estimator,
    max_levels=10,
    verbose=True,
    random_state=RSTATE,
)

cascade_weak_label_proba = Cascade(
    level=ClassifierTransformer(final_estimator),
    final_estimator=final_estimator,
    inter_level_sampler=weak_label_imputer,
    max_levels=10,
    verbose=True,
    random_state=RSTATE,
)

cascade_weak_label_tree_embedder = Cascade(
    level=alternating_level_estimator,
    final_estimator=final_estimator,
    inter_level_sampler=weak_label_imputer,
    max_levels=10,
    verbose=True,
    random_state=RSTATE,
)

cascade_weak_label_tree_embedder_pvalue = Cascade(
    level=alternating_level_estimator,
    final_estimator=final_estimator,
    inter_level_sampler=weak_label_imputer,
    max_levels=10,
    verbose=True,
    random_state=RSTATE,
).set_params(
    level__rf_embedder__rf__max_pvalue=0.05,
    level__xt_embedder__xt__max_pvalue=0.05,
)

estimators_dict = {
    # "cascade_original": cascade_original,
    "cascade_tree_embedder": cascade_tree_embedder,
    # "cascade_weak_label_tree_embedder": cascade_weak_label_tree_embedder,
}

if __name__ == "__main__":
    # X, y, _, _ = load_dataset("mediamill", "undivided")
    # X, y = load_iris(return_X_y=True)
    X, y, _, _ = load_dataset("yeast", "undivided")
    X, y = X.toarray(), y.toarray()[:, :1]
    # cascade = cascade_original.fit(X, y)
    # cascade = cascade_tree_embedder.fit(X, y)
    cascade = positive_dropper.wrap_estimator(
        cascade_weak_label_tree_embedder_pvalue,
        drop=0.5,
        random_state=RSTATE,
    )
    cascade = cascade.fit(X, y)

    joblib.dump(cascade, "cascade.joblib")
    with open("cascade.html", "w") as f:
        f.write(estimator_html_repr(cascade))
