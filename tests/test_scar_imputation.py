import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import sklearn.preprocessing

from deep_forest import weak_labels
from deep_forest import tice

@pytest.fixture
def data():
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=3,
        n_redundant=0,
        n_repeated=0,
        n_classes=2,
        n_clusters_per_class=2,
        weights=None,
        flip_y=0.01,
        class_sep=1.0,
        hypercube=True,
        shift=0.0,
        scale=1.0,
        shuffle=True,
        random_state=0,
    )
    X = sklearn.preprocessing.MinMaxScaler().fit_transform(X)
    y = y.reshape(-1, 1)
    return X, y


@pytest.mark.parametrize(
    "imputer",
    [weak_labels.SCARImputer, weak_labels.LabelComplementImputer],
    ids=["scar", "label_complement"],
)
def test_scar_imputer(data, imputer):
    X, y = data

    oob_imputer = imputer(
        estimator=RandomForestClassifier(
            n_estimators=100,
            max_depth=3,
            random_state=0,
            oob_score=True,
            bootstrap=True,
            max_samples=0.5,
        ),
        verbose=True,
        label_freq_percentile=0.95,
    )

    Xt, yt = oob_imputer.fit_resample(X, y)

    assert (yt[y == 1] == 1).all()
    assert (yt[y == 0] == 1).any()
