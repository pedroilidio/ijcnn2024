"""Original deep forest model based on [1]_ and [2]_.

.. [1] Zhou, Z. H., & Feng, J. (2017, August). Deep Forest: Towards An
Alternative to Deep Neural Networks. In IJCAI (pp. 3553-3559).

.. [2] Zhou, Z. H., & Feng, J. (2019). Deep forest. National science review,
6(1), 74-86.
"""
import joblib
from sklearn.datasets import load_iris
from sklearn.pipeline import FeatureUnion
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold
from sklearn.utils import check_random_state
from deep_forest.estimator_adapters import ProbaTransformer
from deep_forest.cascade import Cascade

RSTATE = check_random_state(0)

# Slighly modified relative to the original papers. Instead of averaging four
# random forest and four extra trees each submitted to 3-fold CV, we combine
# the predictions of a random forest and extra trees each submitted to
# 4-times repeated 3-fold CV.
cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=4, random_state=RSTATE)

random_forest_transformer = ProbaTransformer(
    CalibratedClassifierCV(
        RandomForestClassifier(
            n_estimators=500,
            max_features="sqrt",
            random_state=RSTATE,
        ),
        cv=cv,
        ensemble=True,
    )
)
extra_trees_transformer = ProbaTransformer(
    CalibratedClassifierCV(
        ExtraTreesClassifier(
            n_estimators=500,
            max_features=1,
            random_state=RSTATE,
        ),
        cv=cv,
        ensemble=True,
    )
)

level_estimator = FeatureUnion(
    [
        ("xt", extra_trees_transformer),
        ("rf", random_forest_transformer),
    ]
)

cascade_forest = Cascade(
    level=level_estimator,
    final_estimator=RandomForestClassifier(
        n_estimators=500,
        max_features="sqrt",
        random_state=RSTATE,
    ),
    scorer="neg_mean_squared_error",
    stopping_score=-0.0001,
    # min_improvement=0.00001,
    max_levels=10,
    verbose=True,
    random_state=RSTATE,
)

if __name__ == "__main__":
    X, y = load_iris(return_X_y=True)
    cascade_forest.fit(X, y)
    joblib.dump(cascade_forest, "cascade_forest.joblib")
