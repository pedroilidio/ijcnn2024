"""Employ tree embeddings and weak-label inputting in deep forest models.
"""
import joblib
from sklearn.datasets import load_iris
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.decomposition import PCA
from sklearn.ensemble import (
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

from deep_forest.tree_embedder import (
    ForestEmbedder,
)
from deep_forest.cascade import Cascade
from deep_forest.weak_labels import WeakLabelImputer

RSTATE = check_random_state(0)

rf_embedder = ForestEmbedder(
    RandomForestClassifier(
        n_estimators=100,
        max_features="sqrt",
        random_state=RSTATE,
    ),
    method="all_nodes",
    node_weights="neg_log_frac",
    max_node_size=0.8,
)

xt_embedder = ForestEmbedder(
    ExtraTreesClassifier(
        n_estimators=100,
        max_features=1,
        random_state=RSTATE,
    ),
    method="all_nodes",
    node_weights="neg_log_frac",
    max_node_size=0.8,
)

final_estimator = StackingClassifier(
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
    cv=KFold(n_splits=3, shuffle=True, random_state=RSTATE),
)

level_estimator = Pipeline(
    [
        (
            "forest_embedders",
            FeatureUnion([("xt", xt_embedder), ("rf", rf_embedder)]),
        ),
        (
            "pca",
            PCA(n_components=0.8, random_state=RSTATE),
        ),
    ]
)

weak_label_imputer = WeakLabelImputer(final_estimator, threshold=0.8)

cascade_forest = Cascade(
    level=level_estimator,
    final_estimator=final_estimator,
    # scorer="neg_mean_squared_error",
    # stopping_score=-0.0001,
    # min_improvement=0.00001,
    max_levels=10,
    verbose=True,
    random_state=RSTATE,
)

weak_label_cascade_forest = Cascade(
    level=level_estimator,
    inter_level_sampler=weak_label_imputer,
    final_estimator=final_estimator,
    max_levels=10,
    verbose=True,
    random_state=RSTATE,
    memory=None,
)

if __name__ == "__main__":
    X, y = load_iris(return_X_y=True)
    # cascade = cascade_forest.fit(X, y)
    cascade = weak_label_cascade_forest.fit(X, y)

    joblib.dump(cascade, "cascade.joblib")
    with open("cascade.html", "w") as f:
        f.write(estimator_html_repr(cascade))
