import numpy as np
from pathlib import Path
from yaml import dump
from sklearn.model_selection import cross_validate, StratifiedKFold
from skmultilearn.dataset import load_dataset
from estimators import estimators_dict
from skmultilearn.model_selection import IterativeStratification


def load_dataset_wrapper(dataset):
    X, y, _, _ = load_dataset(dataset, "undivided")
    return dict(X=X.toarray(), y=y.toarray())


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