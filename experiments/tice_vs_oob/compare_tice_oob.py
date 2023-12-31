from pathlib import Path
import sys

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import sklearn.preprocessing

from deep_forest.weak_labels import SCARImputer
from deep_forest import tice
sys.path.insert(0, str(Path(__file__).parent.parent))
from positive_dropper import PositiveDropper


def main():
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=3,
        n_redundant=0,
        n_repeated=0,
        n_classes=2,
        n_clusters_per_class=2,
        weights=None,
        # flip_y=0.01,
        class_sep=1.0,
        hypercube=True,
        shift=0.0,
        scale=1.0,
        shuffle=True,
        random_state=0,
    )
    X = sklearn.preprocessing.MinMaxScaler().fit_transform(X)
    y = y.reshape(-1, 1)

    c1, cc1 = tice.estimate_label_frequency_lower_bound(X, y, 10, most_promising_only=True)

    oob_imputer = SCARImputer(
        estimator=RandomForestClassifier(
            n_estimators=100,
            max_depth=3,
            random_state=0,
            oob_score=True,
            bootstrap=True,
            max_samples=0.5,
        ),
        verbose=True,
    )

    Xt, yt = oob_imputer.fit_resample(X, y)
    c2 = oob_imputer.label_frequency_estimates_

    breakpoint()


if __name__ == "__main__":
    main()