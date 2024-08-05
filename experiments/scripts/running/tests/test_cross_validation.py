import pytest
from pathlib import Path
import sys

import numpy as np
import sklearn.metrics
from sklearn.base import clone
from imblearn.pipeline import Pipeline as ImblearnPipeline

from experiments.runner_scripts.cross_validation import cross_validate_cascade_levels
import experiments.scripts.model_building.estimators as estimators
import experiments.scripts.running.scoring as scoring

from .scripts.running.data_loaders import load_nakano

sys.path.insert(0, str(Path(__file__).parents[2]))
from model_building.positive_dropper import PositiveDropper


@pytest.fixture
def random_state():
    return 0


@pytest.fixture
def data():
    Xy = load_nakano("datasets/MLC/VirusGO.csv", min_positives=30)
    return Xy["X"], Xy["y"]


@pytest.fixture
def cascade():
    estimator = clone(estimators.cafe_slc).set_params(
        max_levels=2,
        memory=None,
    )
    estimator.set_params(**{
        k: 1 for k, v in estimator.get_params().items()
        if k.endswith("n_jobs")
    })
    return estimator


@pytest.fixture
def cascade_with_dropper(random_state, cascade):
    return ImblearnPipeline([
        ("dropper", PositiveDropper(0.25, random_state=random_state)),
        ("cascade", cascade),
    ])


def test_cascade(data, cascade_with_dropper):
    X, y = data
    cascade = cascade_with_dropper.fit(X, y)

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


def test_cross_validation_cascade_levels(data, cascade_with_dropper):
    X, y = data
    X = X[:1000]
    y = y[:1000]

    fitted_params = [
        "n_components_",
        "label_frequency_estimates_",
    ]
    results = cross_validate_cascade_levels(
        estimator=cascade_with_dropper,
        X=X,
        y=y,
        cv=estimators.make_iterative_stratification(n_splits=2),
        scoring=scoring.all_scorers,
        error_score="raise",
        return_estimator=False,
        return_train_score=True,
        return_fitted_params=fitted_params,
    )

    assert "fitted_params" in results.keys()
    assert results["fitted_params"]
    for key in results["fitted_params"].keys():
        assert any(key.endswith(param) for param in fitted_params)
        assert "last_level" not in key

    for score_name, score in results.items():
        if (
            score_name.startswith("train_")
            and (score_name.endswith("_masked") or score_name.endswith("_oob"))
        ):
            assert not np.isnan(score).any()
