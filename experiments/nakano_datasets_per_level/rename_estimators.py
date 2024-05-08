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

    # ========================================================================
    # Apply LCForest (Wang et al., 2020) prunning strategy.
    # ========================================================================
    # Stop cascade when imputation is finished for all labels.
    print("Select LCForest levels according to the original imputation criterion...")

    # Select cascade_wang, cascade_wang__30, and so on
    lc_level = results.loc[results["estimator.name"].str.startswith("cascade_wang")]
    lc_level = lc_level.loc[lc_level["level"] != 0]  # No c estimation yet

    # Imputation finishes when no labels are enabled
    imputation_finished = (
        # TODO: We should use "next_enable_imputation_" instead of "enable_imputation_"
        lc_level.loc[:, lc_level.columns.str.endswith("enable_imputation_")]
        # (NaN, A) -> A, (B, NaN) -> B
        .apply(lambda s: s.loc[s.first_valid_index()], axis=1)
        # "[True, False, True]" -> "1 0 1"
        .str.strip("[]")
        .str.replace("True", "1")
        .str.replace("False", "0")
        .str.replace(",", " ")
        # All labels are disabled when all values are 0
        .apply(lambda s: not np.fromstring(s, dtype=bool).any())
        # .apply(np.fromstring, dtype=bool)
    )
    # imputation_finished = imputation_finished.apply(lambda s: not s.any())
    print("Number of early stopped LCForest runs:", imputation_finished.sum())

    level_is_max = (
        lc_level
        .groupby(run_id, group_keys=False, sort=False, as_index=False)
        .apply(lambda g: g["level"] == g["level"].max())
    )

    # Select first level to finish imputation or the last level if none finished
    lc_level = (
        lc_level.loc[level_is_max | imputation_finished]
        .sort_values("level")
        .drop_duplicates(run_id, keep="first")
        .assign(level="finished_imputation")
    )

    # ========================================================================

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
        # .droplevel("level")  # XXX Uncommented is working.
        # .stack("scorer", future_stack=True)
    )
    # expected_best = expected_best.droplevel(
    #     [l for l in expected_best.index.names if l.startswith("results.")]
    # )
    
    # Categorical index for faster groupby iteration
    expected_best = expected_best.droplevel(
        # [l for l in expected_best.index.names if l not in run_id]  # XXX
        [l for l in expected_best.index.names if l not in level_run_id]
    )
    idx_names = expected_best.index.names
    expected_best = expected_best.reset_index()
    expected_best.index = pd.MultiIndex.from_frame(
        expected_best.loc[:, idx_names].astype("category")
    )
    expected_best = expected_best.drop(idx_names, axis=1)

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
            # .apply(lambda g2: g2.nlargest(1, col))  # XXX Working.
            .apply(lambda g2: (
                g2.nlargest(1, col)
                .droplevel("level")
                # .reset_index(level="level", names=("", "original_level"),
            ))
            .unstack("metric")
            #.reorder_levels(["metric", "scorer"], axis=1)
        )
        assert result.shape[0] == 1  # XXX
        return result

    def get_max_ascending_score(g, col):
        # Apply gcForest (Zhou et al., 2019) prunning strategy.
        # Stop cascade when a drop in OOB score is detected (originally CV).
        assert g.index.is_monotonic_increasing
        result = (
            g.stack("metric")
            .groupby(level="metric", group_keys=False, sort=False, observed=False)
            # Get the row before the first row where the score decreases
            # If no decrease is detected, get the last row (argmax returns 0, so -1)
            .apply(lambda g2: g2.iloc[[(g2[col].diff() < 0).argmax() - 1]].droplevel("level"))
            .unstack("metric")
        )
        assert result.shape[0] == 1
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
    groups_max_ascending_oob = []

    for _, group in tqdm(grouped, total=grouped.ngroups):
        groups_oob.append(get_expected_best(group, "train_oob_internal"))
        groups_train.append(get_expected_best(group, "train_internal"))
        groups_max_ascending_oob.append(
            get_max_ascending_score(group, "train_oob_internal")
        )
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

    print("Concatenating groups with max ascending OOB scores...")
    max_ascending_oob = pd.concat(groups_max_ascending_oob)#, ignore_index=True)
    # max_ascending_oob = max_ascending_oob.stack("metric") #
    rename_result_columns_as_initially(max_ascending_oob)
    max_ascending_oob = max_ascending_oob.reset_index().assign(level="max_ascending_oob")

    # ========================================================================

    results["original_estimator_name"] = results["estimator.name"]
    results["estimator.name"] += "__level" + results["level"].astype(str)

    lc_level["original_estimator_name"] = lc_level["estimator.name"]
    lc_level["estimator.name"] += "__finished_imputation"

    # # Assert that best level scores are higher than expected best scores
    # # TODO: should not be necessary.
    # wrong_bo = (
    #     best_oob.set_index(run_id)
    #     > best_level.set_index(run_id).loc[:, best_oob.columns]
    # )
    # if wrong_bo.any().any():
    #     warn(
    #         f"Level with best OOB scores are better than best level for"
    #         f" {wrong_bo.sum().sum()} runs:"
    #         f"\n{best_oob.loc[wrong_bo.any(axis=1), run_id]}"
    #     )

    # wrong_bt = (
    #     best_train.set_index(run_id)
    #     > best_level.set_index(run_id).loc[:, best_train.columns]
    # )
    # if wrong_bt.any().any():
    #     warn(
    #         f"Level with best training scores are better than best level for"
    #         f" {wrong_bt.sum().sum()} runs:"
    #         f"\n{best_train.loc[wrong_bt.any(axis=1), run_id]}"
    #     )

    # Add suffixes to distinguish the best level, best OOB and best training
    # as different estimators
    best_level["original_estimator_name"] = best_level["estimator.name"]
    best_oob["original_estimator_name"] = best_oob["estimator.name"]
    best_train["original_estimator_name"] = best_train["estimator.name"]
    max_ascending_oob["original_estimator_name"] = max_ascending_oob["estimator.name"]

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
    max_ascending_oob["estimator.name"] = (
        max_ascending_oob["estimator.name"].astype(str) + "__max_ascending_oob"
    )

    results = pd.concat(
        [results, lc_level, best_level, best_oob, best_train, max_ascending_oob],
        ignore_index=True,
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
