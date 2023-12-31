"""Manually fixes differences in estimator namings in previous versions of the runs."""
import argparse
from pathlib import Path
from warnings import warn

import pandas as pd
import numpy as np


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
    estimator_renaming = {
        # "bxt_gmo__25": "bxt_gmo",
        # "brf_gmo__75": "brf_gmo",
        # "adss_bxt_gso": "ss_bxt_gso__ad_fixed",
        # "md_ss_bxt_gso": "ss_bxt_gso__md_fixed",
        # "md_ds_bxt_gso": "ss_bxt_gso__md_size",
        # "ss_bxt_gso": "ss_bxt_gso__mse_fixed",
        # "rs_bxt_gso": "ss_bxt_gso__mse_random",
        # "ds_bxt_gso": "ss_bxt_gso__mse_density",
    }

    # Rename estimators
    # data["estimator.name"] = (
    #     data["estimator.name"].map(estimator_renaming).fillna(data["estimator.name"])
    # )

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
        names=("metric", "level"),
    )
    results = results.stack("level")
    results = results.reset_index()

    run_identifiers = ["cv.fold", "dataset.name", "estimator.name", "level"]
    dup_idx = results.duplicated(run_identifiers)
    
    if dup_idx.any():
        warn(f"Removing duplicate rows: \n {results.loc[dup_idx, run_identifiers]}")
        results = results.sort_values("start").drop_duplicates(
            run_identifiers,
            keep="last",
        )
    
    score_columns_mask = results.columns.str.match(r"results\.(test|train).*")

    best_level_results = (
        results
        .set_index(results.columns[~score_columns_mask].tolist())
        .groupby(level=["cv.fold", "dataset.name", "estimator.name"])
        .max()  # Max between levels
        .reset_index()
        .assign(level="best")
    )

    def get_expected_best(g):
        return g.T.groupby(
            (
                g.columns
                .str.removeprefix("results.")
                .str.removeprefix("train_")
                .str.removeprefix("test_")
                .str.removesuffix("_oob")
                .str.removesuffix("_masked")
            ),
        ).apply(
            lambda g2: g2.T.sort_values(f"results.train_{g2.name}_oob").iloc[-1]
        )

    expected_best_level_results = (
        results
        .set_index(results.columns[~score_columns_mask].tolist())
        .groupby(level=["cv.fold", "dataset.name", "estimator.name"])
        .apply(get_expected_best)
        .reset_index()
        .assign(level="expected_best")
    )
    new_columns = expected_best_level_results.columns.get_level_values(1).values
    new_columns[new_columns == ''] = (
        expected_best_level_results.columns.get_level_values(0)[new_columns == '']
    )
    expected_best_level_results.columns = new_columns

    results['original_estimator_name'] = results['estimator.name']
    best_level_results['original_estimator_name'] = (
        best_level_results['estimator.name']
    )
    expected_best_level_results['original_estimator_name'] = (
        expected_best_level_results['estimator.name']
    )

    best_level_results['estimator.name'] += '__best'
    expected_best_level_results['estimator.name'] += '__expected_best'

    results = pd.concat([results, best_level_results, expected_best_level_results])

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
