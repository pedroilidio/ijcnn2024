"""Manually fixes differences in estimator namings in previous versions of the runs."""
import argparse
from pathlib import Path
from warnings import warn
from time import time

import pandas as pd
import numpy as np
from tqdm import tqdm


def rename_result_columns_as_initially(df):
    metric = df.columns.get_level_values("metric")
    scorer = df.columns.get_level_values("scorer")
    splitted_scorer = scorer.to_series().str.split("_", n=1, expand=True)
    train_test, suffix = splitted_scorer[0], splitted_scorer[1]
    df.columns = (
        "results."
        + train_test
        + "_"
        + metric.astype(str)
        + ("_" + suffix).fillna("")
    )


def rename_estimators(input_path: Path):
    print(f"Renaming estimators from {input_path}...")
    data = pd.read_table(input_path)
    output_path = input_path.with_stem(input_path.stem + "_renamed")

    wrapper_renaming = {
        np.nan: "",
        "drop10": "__10",
        "drop20": "__20",
        "drop30": "__30",
        "drop40": "__40",
        "drop50": "__50",
        "drop60": "__60",
        "drop70": "__70",
        "drop80": "__80",
        "drop90": "__90",
    }

    if "wrapper.name" in data:
        # Rename estimators to include wrapper name
        data["suffix"] = (
            data["wrapper.name"].map(wrapper_renaming).fillna(data["wrapper.name"])
        )
        data["estimator.name"] += data["suffix"]

    level = data.columns.str.extract(r"level(\d+)")[0]
    data.columns = data.columns.str.replace(r"level(\d+)(__)?(\.)?", "", regex=True)

    results = data.set_index(data.columns[level.isna().values].tolist())

    results.columns = pd.MultiIndex.from_arrays(
        (results.columns, level[~level.isna()].astype(int)),
        names=("scorer", "level"),
    )
    results = results.stack("level")
    results = results.reset_index()

    run_id = ["cv.fold", "dataset.name", "estimator.name"]
    level_run_id = ["cv.fold", "dataset.name", "estimator.name", "level"]

    dup_idx = results.duplicated(level_run_id)
    
    if dup_idx.any():
        warn(f"Removing duplicate rows: \n {results.loc[dup_idx, level_run_id]}")
        results = results.sort_values("start").drop_duplicates(
            level_run_id,
            keep="last",
        )
    
    score_columns_mask = results.columns.str.match(r"results\.(test|train).*")

    # Internal scores (considering the dropped labels) are not computed when no
    # dropper is used. We thus copy the external scores to the internal scores
    # in these cases, since no dropping is actually performed.
    internal_scores = results.columns[
        results.columns.str.endswith("_internal")
    ]
    no_drop = results["wrapper.name"].isna()  # wrapper is the positive dropper

    results.loc[no_drop, internal_scores] = results.loc[
        no_drop,
        internal_scores.str.removesuffix("_internal")
    ].values  # ignore column names

    best_level = (
        results.loc[results.level != 0]  # Ignore level 0 (only the final estimator)
        # .set_index(results.columns[~score_columns_mask].tolist())
        .set_index(level_run_id)
        .groupby(level=run_id)
        .max(numeric_only=True)  # Max between levels
        .reset_index()
        .assign(level="best")
    )

    expected_best = (
        results.loc[results.level != 0]  # Ignore level 0 (only the final estimator)
        .set_index(results.columns[~score_columns_mask].tolist())
        .droplevel("level")
        # .stack("scorer", future_stack=True)
    )
    # expected_best = expected_best.droplevel(
    #     [l for l in expected_best.index.names if l.startswith("results.")]
    # )
    
    # Categorical index for faster groupby iteration
    expected_best = expected_best.droplevel(
        [l for l in expected_best.index.names if l not in run_id]
    )
    expected_best = expected_best.reset_index()
    expected_best.index = pd.MultiIndex.from_frame(
        expected_best.loc[:, run_id].astype("category")
    )
    expected_best = expected_best.drop(run_id, axis=1)

    metric = (
        expected_best.columns.get_level_values("scorer")
        .str.removeprefix("results.")
        .str.removeprefix("train_")
        .str.removeprefix("test_")
        .str.removesuffix("_internal")
        .str.removesuffix("_masked")
        .str.removesuffix("_oob")
        .rename("metric")
        # .to_series()
        # .set_axis(expected_best.index)
    )

    old_columns = expected_best.columns

    # Example:
    # results.test_roc_auc_micro_oob_internal -> (roc_auc_micro, test_oob_internal)
    cat_columns = pd.DataFrame.from_records(
        (
            (m, s.replace("_" + m, ""))
            for m, s in zip(metric, old_columns.str.replace("results.", ""))
        ),
        columns=("metric", "scorer"),
    ).astype("category")

    expected_best.columns = pd.MultiIndex.from_frame(cat_columns)

    # expected_best = expected_best.stack("metric")
    expected_best = expected_best.sort_index(axis=1).sort_index(axis=0)

    def get_expected_best(g, col):
        # result = (
        #     g.T.groupby(level="metric", group_keys=False, sort=False, observed=False)
        #     .apply(lambda g2: g2.droplevel("metric").T.nlargest(1, col).T)
        #     # .apply(lambda g2: g2.T.nlargest(1, (g2.name, col)).T)
        # ).T
        result = (
            g.stack("metric")
            .groupby(level="metric", group_keys=False, sort=False, observed=False)
            .apply(lambda g2: g2.nlargest(1, col))
            .unstack("metric")
            #.reorder_levels(["metric", "scorer"], axis=1)
        )
        # assert result.shape[0] == 1
        return result

    # Appending groups like so is slow:
    # expected_best = (
    #     # expected_best.sort_index(level=["cv.fold", "dataset.name", "estimator.name"])
    #     expected_best.sort_index()
    #     .groupby(
    #         level=["cv.fold", "dataset.name", "estimator.name"],
    #         group_keys=False,
    #         sort=False,
    #     )
    #     .apply(get_expected_best)
    #     .set_axis(old_columns, axis=1)
    #     .reset_index()
    #     # .droplevel("metric", axis=1)
    #     .assign(level="expected_best")
    # )
    
    print("Selecting internal best level...")
    grouped = expected_best.groupby(
        level=run_id,  # + ["metric"],  # If metric was stacked
        group_keys=False,
        sort=False,
        observed=False,
    )
    groups_oob = []
    groups_train = []

    for _, group in tqdm(grouped, total=grouped.ngroups):
        groups_oob.append(get_expected_best(group, "train_oob_internal"))
        groups_train.append(get_expected_best(group, "train_internal"))
        # Stack per group is faster than:
        # groups_oob.append(group.nlargest(1, "train_oob_internal"))
        # groups_train.append(group.nlargest(1, "train_internal"))

    print("Concatenating groups with best OOB scores...")
    best_oob = pd.concat(groups_oob)#, ignore_index=True)
    # best_oob = best_oob.stack("metric") #
    rename_result_columns_as_initially(best_oob)
    best_oob = best_oob.reset_index().assign(level="best_oob")
    
    print("Concatenating groups with best training scores...")
    best_train = pd.concat(groups_train)#, ignore_index=True)
    # best_train = best_train.stack("metric") #
    rename_result_columns_as_initially(best_train)
    best_train = best_train.reset_index().assign(level="best_train")

    results["original_estimator_name"] = results["estimator.name"]
    results["estimator.name"] += "__level" + results["level"].astype(str)

    # Assert that best level scores are higher than expected best scores
    # TODO: should not be necessary.
    wrong_bo = (
        best_oob.set_index(run_id)
        > best_level.set_index(run_id).loc[:, best_oob.columns]
    )
    if wrong_bo.any().any():
        warn(
            f"Level with best OOB scores are better than best level for"
            f" {wrong_bo.sum().sum()} runs:"
            f"\n{best_oob.loc[wrong_bo.any(axis=1), run_id]}"
        )

    wrong_bt = (
        best_train.set_index(run_id)
        > best_level.set_index(run_id).loc[:, best_train.columns]
    )
    if wrong_bt.any().any():
        warn(
            f"Level with best training scores are better than best level for"
            f" {wrong_bt.sum().sum()} runs:"
            f"\n{best_train.loc[wrong_bt.any(axis=1), run_id]}"
        )

    # Add suffixes to distinguish the best level, best OOB and best training
    # as different estimators
    best_level["original_estimator_name"] = best_level["estimator.name"]
    best_oob["original_estimator_name"] = best_oob["estimator.name"]
    best_train["original_estimator_name"] = best_train["estimator.name"]

    # best_level["estimator.name"] += "__best"
    # best_oob["estimator.name"] += "__best_oob"
    # best_train["estimator.name"] += "__best_train"

    # If one intends to keep categories, use:
    #     best_level["estimator.name"].cat.rename_categories(
    #         lambda x: x + "__best"
    #     )
    best_level["estimator.name"] = (
        best_level["estimator.name"].astype(str) + "__best"
    )
    best_oob["estimator.name"] = (
        best_oob["estimator.name"].astype(str) + "__best_oob"
    )
    best_train["estimator.name"] = (
        best_train["estimator.name"].astype(str) + "__best_train"
    )

    results = pd.concat(
        [results, best_level, best_oob, best_train],
        ignore_index=True,
    )

    # Final abbreviations
    results["estimator.name"] = (
        results["estimator.name"]
        .str.replace("level", "")
        .str.replace("best_oob", "bo")
        .str.replace("best_train", "bt")
        .str.replace("best", "b")
        .str.replace("cascade_", "")
        .str.replace("tree_embedder", "te")
        .str.replace("proba", "os")
    )

    results.to_csv(output_path, sep="\t", index=False)
    print(f"Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Rename estimators in a runs table.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "results",
        type=Path,
        help="Path to the runs table to rename.",
    )
    args = parser.parse_args()

    rename_estimators(args.results)


if __name__ == "__main__":
    main()
