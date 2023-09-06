from pathlib import Path
import warnings

import numpy as np
from yaml import dump
from sklearn.model_selection import cross_validate, StratifiedKFold
from skmultilearn.dataset import load_dataset
from skmultilearn.model_selection import IterativeStratification

from estimators import estimators_dict


def load_dataset_wrapper(dataset, min_positives: int | float = False):
    X, y, _, _ = load_dataset(dataset, "undivided")
    X, y = X.toarray(), y.toarray()

    if isinstance(min_positives, float):
        min_positives =  int(np.ceil(min_positives * y.shape[0]))
    if min_positives:
        warnings.warn(
            f"Label columns {np.where(y.sum(axis=0) > min_positives)[0].tolist()}"
            f" of dataset {dataset!r} (which had {y.shape[0]} labels in total) have"
            f" less than {min_positives} positives. Dropping them during loading."
        )
        y = y[:, y.sum(axis=0) >= min_positives]

        if y.shape[1] == 0:
            raise ValueError(
                f"None of the label columns of dataset {dataset!r} had more than"
                f" {min_positives} positives."
            )

    return dict(X=X, y=y)

datasets = [
    'Corel5k',
    'genbase',
    'bibtex',
    'birds',
    'delicious',
    'emotions',
    'enron',
    'mediamill',
    'medical',
    'rcv1subset1',
    # 'rcv1subset2',
    # 'rcv1subset3',
    # 'rcv1subset4',
    # 'rcv1subset5',
    'scene',
    'tmc2007_500',
    'yeast',
]

scoring = [
    "accuracy",
    "f1_micro",
    "f1_macro",
    "f1_weighted",
    "average_precision",
    "roc_auc",
    "matthews_corrcoef",
    "balanced_accuracy",
]


def main():
    outdir = Path("results")
    outdir.mkdir(exist_ok=True)

    for dataset in datasets:
        print(f"Loading {dataset}...")
        X, y, _, _ = load_dataset(dataset, "undivided")
        X, y = X.toarray(), y.toarray()
        print(f"X shape: {X.shape}")
        print(f"y shape: {y.shape}")
        print(f"y density: {y.mean()}")
        print()
        assert set(np.unique(y)) == {0, 1}

        for name, estimator in estimators_dict.items():
            print(f"Running {name}...")
            cv_results = cross_validate(
                estimator,
                X,
                y,
                cv=IterativeStratification(n_splits=5, order=1),
                scoring=scoring,
                verbose=3,
                n_jobs=5,
            )
            print(f"cv_results: {cv_results}")
            with (outdir / f"{name}.yml").open('w') as f:
                dump(cv_results, f)
            print()


if __name__ == "__main__":
    main()