"""Employ tree embeddings and weak-label inputting in deep forest models.
"""
from numbers import Real, Integral
from functools import partial
import os

import joblib
import numpy as np
import sklearn.metrics
from sklearn.base import clone
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
from sklearn.utils._param_validation import (
    Interval,
)
from skmultilearn.dataset import load_dataset

from deep_forest.tree_embedder import ForestEmbedder
from deep_forest.cascade import Cascade, AlternatingLevel
from deep_forest.weak_labels import WeakLabelImputer, PositiveUnlabeledImputer
from deep_forest.estimator_adapters import (
    ProbaTransformer,
    EstimatorAsTransformer,
    MultiOutputVotingClassifier,
    TreeEmbedderWithOutput,
)
from skmultilearn.model_selection import IterativeStratification
from _lobpcg_truncated_svd import LOBPCGTruncatedSVD
import positive_dropper

N_TREES = 150
# NOTE: the paper undersamples for the whole forest, we perform undersampling
# for each tree.
MAX_EMBEDDING_SAMPLES = 0.5
MAX_DEPTH = 10
N_COMPONENTS = 0.1
MAX_LEVELS = 10
VERBOSE = 10
RSTATE = 0  # check_random_state(0)
NJOBS = 14
MEMORY = None

for var in [
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "BLIS_NUM_THREADS",
]:
    value = NJOBS
    print(f"Setting environment variable {var}=\"{value}\"")
    os.environ[var] = str(value)

scoring_metrics = {
    "precision_samples": "precision_samples",
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
    "accuracy": "accuracy",
    "label_ranking_average_precision_score": sklearn.metrics.make_scorer(
        sklearn.metrics.label_ranking_average_precision_score,
        needs_threshold=True,
        greater_is_better=True,
    ),
    "label_ranking_loss": sklearn.metrics.make_scorer(
        sklearn.metrics.label_ranking_loss,
        needs_threshold=True,
        greater_is_better=True,
    ),
    # Not multi-label
    # - roc_auc_ovr
    # - roc_auc_ovr_weighted
    # - roc_auc_ovo
    # - roc_auc_ovo_weighted
    # - average_precision
    # - neg_log_loss
    # - neg_brier_score
    # - balanced_accuracy
    # - matthews_corrcoef
}


def get_scoring_metrics():
    return scoring_metrics


def get_level_scores(cascade, level, *args, **kwargs):
    """Get scores for each level of a cascade."""
    return getattr(cascade, "level_scores_", [])


def get_get_level_scores():
    return partial(get_level_scores)


def make_iterative_stratification(**kwargs):
    """CV factory to use in YAML config files."""
    # NOTE: simply calling __init__ will not work, PyYAML expects module-level
    # functions.
    return IterativeStratification(**kwargs)


# [...] we discard the nodes in which are present in more than [max_node_size%]
# of the instances, removing non-meaningful nodes, such as the ones located
# close to the root and the root itself.

# [...] the remaining nodes are weighted according to Eq. (1) where nodes that
# contain many instances are credited a lower weight, and the opposite is valid
# for more representative nodes.

rf_embedder = ForestEmbedder(
    RandomForestRegressor(
        n_estimators=N_TREES,
        max_features="sqrt",
        max_depth=MAX_DEPTH,
        max_samples=MAX_EMBEDDING_SAMPLES,
        bootstrap=MAX_EMBEDDING_SAMPLES is not None,
        n_jobs=NJOBS,  # Less depth, less memory
        verbose=VERBOSE,
        random_state=RSTATE,
    ),
    method="path",
    node_weights="log_node_size",  # Eq. (1)
    max_node_size=0.8,
)

xt_embedder = ForestEmbedder(
    ExtraTreesRegressor(
        n_estimators=N_TREES,
        max_features=1,
        max_depth=MAX_DEPTH,
        max_samples=MAX_EMBEDDING_SAMPLES,
        bootstrap=MAX_EMBEDDING_SAMPLES is not None,
        n_jobs=NJOBS,  # Less depth, less memory
        verbose=VERBOSE,
        random_state=RSTATE,
    ),
    method="path",
    node_weights="log_node_size",  # Eq. (1)
    max_node_size=0.8,
)

final_estimator = MultiOutputVotingClassifier(
    estimators=[
        (
            "rf",
            RandomForestClassifier(
                n_estimators=N_TREES,
                max_features="sqrt",
                random_state=RSTATE,
                n_jobs=NJOBS,
                verbose=VERBOSE,
            ),
        ),
        (
            "xt",
            ExtraTreesClassifier(
                n_estimators=N_TREES,
                max_features=1,
                random_state=RSTATE,
                n_jobs=NJOBS,
                verbose=VERBOSE,
            ),
        ),
    ],
)


# For the number of PCA components in our method, we have optimized the number
# of components considering a percentage of the total number of decision path
# features{1 component, 1%, 5%, 20%, 40%, 60%, 80%, 95%} multiplied by min(N,
# |c|), being N the number of instances in the dataset and |c| the number of
# nodes in the ensemble. [we fixed that percentage to N_COMPONENTS]

alternating_level_embedding = AlternatingLevel([
    ("xt_embedder", Pipeline([
        ("xt", xt_embedder),
        (
            "pca",
            LOBPCGTruncatedSVD(
                n_components=N_COMPONENTS,
                max_components=800,
                random_state=RSTATE,
            ),
        ),
    ])),
    ("rf_embedder", Pipeline([
        ("rf", rf_embedder),
        (
            "pca",
            LOBPCGTruncatedSVD(
                n_components=N_COMPONENTS,
                max_components=800,
                random_state=RSTATE,
            ),
        ),
    ])),
])


alternating_level_proba = AlternatingLevel(
    [
        (
            "rf",
            EstimatorAsTransformer(
                RandomForestRegressor(
                    max_features="sqrt",
                    n_estimators=N_TREES,
                    random_state=RSTATE,
                    max_depth=MAX_DEPTH,
                    n_jobs=NJOBS,
                    verbose=VERBOSE,
            )
            ),
        ),
        (
            "xt",
            EstimatorAsTransformer(
                ExtraTreesRegressor(
                    max_features=1,
                    n_estimators=N_TREES,
                    random_state=RSTATE,
                    max_depth=MAX_DEPTH,
                    n_jobs=NJOBS,
                    verbose=VERBOSE,
                )
            ),
        ),
    ]
)

alternating_level_embedding_proba = AlternatingLevel([
    ("xt", TreeEmbedderWithOutput(
            xt_embedder,
            post_transformer=LOBPCGTruncatedSVD(
                n_components=N_COMPONENTS,
                max_components=800,
                random_state=RSTATE,
            ),
        ),
    ),
    ("rf", TreeEmbedderWithOutput(
            rf_embedder,
            post_transformer=LOBPCGTruncatedSVD(
                n_components=N_COMPONENTS,
                max_components=800,
                random_state=RSTATE,
            ),
        ),
    ),
])

weak_label_imputer = PositiveUnlabeledImputer(
    ExtraTreesClassifier(  # TODO: use regressor
        n_estimators=N_TREES,
        bootstrap=True,
        max_samples=0.9,
        oob_score=True,
        random_state=RSTATE,
        n_jobs=NJOBS,
        max_depth=MAX_DEPTH,
        verbose=VERBOSE,
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

cascade_proba = Cascade(
    level=alternating_level_proba,
    final_estimator=final_estimator,
    max_levels=MAX_LEVELS,
    verbose=True,
    random_state=RSTATE,
    memory=MEMORY,
)

cascade_tree_embedder = Cascade(
    level=alternating_level_embedding,
    final_estimator=final_estimator,
    max_levels=MAX_LEVELS,
    verbose=True,
    random_state=RSTATE,
    memory=MEMORY,
)

cascade_tree_embedder_pvalue = clone(cascade_tree_embedder).set_params(
    level__rf_embedder__rf__max_pvalue=0.05,
    level__xt_embedder__xt__max_pvalue=0.05,
)

cascade_tree_embedder_proba = Cascade(
    level=alternating_level_embedding_proba,
    final_estimator=final_estimator,
    max_levels=MAX_LEVELS,
    verbose=True,
    random_state=RSTATE,
    memory=MEMORY,
)

cascade_weak_label_proba = Cascade(
    level=alternating_level_proba,
    final_estimator=final_estimator,
    inter_level_sampler=weak_label_imputer,
    max_levels=MAX_LEVELS,
    verbose=True,
    random_state=RSTATE,
    memory=MEMORY,
)

cascade_weak_label_tree_embedder = Cascade(
    level=alternating_level_embedding,
    final_estimator=final_estimator,
    inter_level_sampler=weak_label_imputer,
    max_levels=MAX_LEVELS,
    verbose=True,
    random_state=RSTATE,
    memory=MEMORY,
)

cascade_weak_label_tree_embedder_pvalue = clone(
    cascade_weak_label_tree_embedder,
).set_params(
    level__rf_embedder__rf__max_pvalue=0.05,
    level__xt_embedder__xt__max_pvalue=0.05,
)

estimators_dict = {
    "cascade_proba": cascade_proba,
    "cascade_tree_embedder": cascade_tree_embedder,
    "cascade_tree_embedder_pvalue": cascade_tree_embedder_pvalue,
    "cascade_tree_embedder_proba": cascade_tree_embedder_proba,
    "cascade_weak_label_proba": cascade_weak_label_proba,
    "cascade_weak_label_tree_embedder": cascade_weak_label_tree_embedder,
    "cascade_weak_label_tree_embedder_pvalue": cascade_weak_label_tree_embedder_pvalue,
}

if __name__ == "__main__":
    breakpoint()
    # X, y, _, _ = load_dataset("mediamill", "undivided")
    X, y, _, _ = load_dataset("delicious", "undivided")
    # X, y = load_iris(return_X_y=True)
    # X, y, _, _ = load_dataset("yeast", "undivided")
    X, y = X.toarray(), y.toarray()
    # breakpoint()
    # cascade = cascade_original.fit(X, y)
    # cascade = cascade_tree_embedder.fit(X, y)
    cascade = positive_dropper.wrap_estimator(
        cascade_weak_label_proba,
        drop=0.25,
        random_state=RSTATE,
    )
    cascade = cascade.fit(X, y)

    joblib.dump(cascade, "cascade.joblib")
    with open("cascade.html", "w") as f:
        f.write(estimator_html_repr(cascade))
