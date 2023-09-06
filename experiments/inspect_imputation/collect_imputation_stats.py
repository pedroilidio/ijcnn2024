import argparse
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state
from sklearn.metrics import confusion_matrix
from skmultilearn.dataset import load_dataset
from deep_forest.cascade import Cascade

sys.path.insert(0, str(Path(__file__).parent.parent))
from estimators import estimators_dict
from run_cascades import load_dataset_wrapper

DEF_PATH_LOG = Path(__file__).with_suffix(".log")
DEF_LOG_LEVEL = "INFO"
DEF_OUT_DIR = Path("results")
DEF_N_JOBS = 1
DEF_K = 5 # number of folds for cross-validation
RSTATE = 0  # seed for random state
SCORING = [
    "f1_macro",
    "f1_micro",
    "f1_weighted",
    "f1_samples",
    "precision_macro",
    "precision_micro",
    "precision_weighted",
    "precision_samples",
    "recall_macro",
    "recall_micro",
    "recall_weighted",
    "recall_samples",
    "jaccard_macro",
    "jaccard_micro",
    "jaccard_weighted",
    "jaccard_samples",
    # "average_precision",
    # "neg_brier_score",
    "accuracy",
]


def load_dataset_wrapper(dataset):
    X, y, _, _ = load_dataset(dataset, "undivided")
    return dict(X=X.toarray(), y=y.toarray())


# @validate_params
# {
#     "drop": [Interval(Real, 0.0, 1.0, closed="left")],
#     "random_state": ["random_state"],
# }
def positive_unlabeled_cross_validate(
    cascade,
    *,
    k,
    X,
    y,
    random_state,
    n_jobs,
):
    """Cross-validate imputers of a cascade masking positive labels.

    Positives are divided into k folds. In each fold, the selected positive
    labels are masked and the cascade is fitted on the masked labels. The
    resampled labels in each level are collected and scoring on the masked
    labels is computed.
    """
    if set(np.unique(y)) != {0.0, 1.0}:
        raise ValueError("Only binary or multi-label classification is supported.")

    random_state = check_random_state(random_state)
    validation_size = 0.2

    if y.ndim == 1:
        y = y.reshape(-1, 1)
    
    X, X_val, y, y_val = train_test_split(
        X,
        y,
        test_size=validation_size,
        random_state=random_state,
    )

    folds = []
    for j, y_col in enumerate(y.T):
        positive_indices = np.nonzero(y_col)[0] 
        if len(positive_indices) < k:
            raise ValueError(
                f"Number of positive samples ({len(positive_indices)}) in y column "
                f"{j} is less than number of folds ({k})."
            )
        random_state.shuffle(positive_indices)
        folds.append(np.array_split(positive_indices, k))
    
    folds = list(zip(*folds)) # folds.shape = (k, n_outputs, n_positives / k)

    def _fit_fold(fold_i, fold):
        y_sample = y.copy()
        for j in range(y_sample.shape[1]):
            y_sample[fold[j], j] = 0.0

        cascade_ = _CascadeSaveResamples(
            **cascade.get_params(deep=False) | dict(
                random_state=random_state,
                # final_estimator="passthrough",
                max_levels=10,
                # validation_size=0.2,
                validation_size=(X_val, y_val),
                refit=False,
                scoring=SCORING,
            )
        )
        cascade_.fit(X, y_sample)
        eval_indices = np.where(y_sample == 0.0)

        results = []
        for level, (y_resampled, level_scores) in enumerate(
            zip(cascade_.y_resampled_, cascade_.level_scores_)
        ):
            tn, fp, fn, tp = confusion_matrix(
                y[eval_indices],
                y_resampled[eval_indices],
            ).ravel()
            results.append(
                level_scores | dict(
                    fold=fold_i, level=level, tn=tn, fp=fp, fn=fn, tp=tp,
                ),
            )
        return results

    results = joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(_fit_fold)(fold_i, fold)
        for fold_i, fold in enumerate(folds)
    )
    results = sum(results, [])
    return pd.DataFrame.from_records(results)


class _CascadeSaveResamples(Cascade):
    def _resample_data(self, X, y, **fit_params):
        Xt, yt = super()._resample_data(X, y, **fit_params)
        self.y_resampled_ = getattr(self, "y_resampled_", []) + [yt]
        return Xt, yt
    
    # def _validate_scorer(self):
    #     super()._validate_scorer()

    #     def set_estimator_classes(estimator):
    #         estimator.classes_ = self.classes_[0]
    #         return estimator

    #     if isinstance(self.scorer_, dict):
    #         # Convert multioutput to single output
    #         self.scorer_ = {
    #             name: lambda est, y, y_pred: scorer(x.reshape(-1, 1))
    #             for name, scorer in self.scorer_.items()
    #         }


def collect_imputation_stats(
    *,
    out_dir: Path = DEF_OUT_DIR,
    raise_errors: bool = False,
    n_jobs: int = DEF_N_JOBS,
    k: int = DEF_K,
    random_state: int = RSTATE,
):
    out_dir.mkdir(exist_ok=True, parents=True)

    datasets = [
        'genbase',  # too sparse
        'medical',  # too sparse  # ref
        'scene',  # ref
        'tmc2007_500',  # ref
        'yeast',  # ref
        'emotions',
        'Corel5k',  # too sparse
        'bibtex',
        'birds',  # perfect for testing: 1.4 s per level
        'delicious',  # 2.4 min per level
        'enron',
        'mediamill',
        'rcv1subset1',
        'rcv1subset2',
        'rcv1subset3',
        'rcv1subset4',
        'rcv1subset5',
    ]

    for dataset in datasets:
        logging.info(f"Loading {dataset}...")
        data = load_dataset_wrapper(dataset, min_positives=15)
        X, y = data["X"], data["y"]
        logging.info(f"X shape: {X.shape}; y shape: {y.shape}; y density: {y.mean()}")
        assert set(np.unique(y)) == {0, 1}

        for name, estimator in estimators_dict.items():
            out_path = out_dir / f"{dataset}__{name}.tsv"

            if out_path.exists():
                logging.info(f"Skipping {name=} {dataset=} (already exists).")
                continue
            if estimator.inter_level_sampler is None:
                logging.warning(f"Skipping {name} (no inter-level sampler).")
                continue

            logging.info(f"Running {name} on dataset {dataset}...")
            try:
                results_df = positive_unlabeled_cross_validate(
                    estimator,
                    k=k,
                    X=X,
                    y=y,
                    n_jobs=n_jobs,
                    random_state=random_state,
                )
                results_df.to_csv(out_path, sep="\t", index=False)
                logging.info("Results saved.")
            except Exception as e:
                logging.warning(f"Skipping {dataset=} {name=}: {e}")
                if raise_errors:
                    raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEF_OUT_DIR)
    parser.add_argument("--log-file", type=Path, default=DEF_PATH_LOG)
    parser.add_argument("--log-level", type=str, default=DEF_LOG_LEVEL)
    parser.add_argument("--n-jobs", type=int, default=DEF_N_JOBS)
    parser.add_argument("--k", type=int, default=DEF_K)
    parser.add_argument("--random-state", type=int, default=RSTATE)
    parser.add_argument("--raise-errors", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="* [%(levelname)s] %(asctime)s: %(message)s",
        handlers=[logging.FileHandler(args.log_file), logging.StreamHandler()],
    )

    collect_imputation_stats(
        out_dir=args.out_dir,
        raise_errors=args.raise_errors,
        n_jobs=args.n_jobs,
        k=args.k,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()