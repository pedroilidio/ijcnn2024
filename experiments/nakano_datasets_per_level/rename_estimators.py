"""Manually fixes differences in estimator namings in previous versions of the runs."""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np


def rename_estimators(input_path: Path, level_scores_path: Path):
    print(f"Renaming estimators from {input_path}...")
    data = pd.read_table(input_path)
    level_scores_data = pd.read_table(level_scores_path)

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

    # Rename estimators
    # Split max_levels from estimator name (e.g. my_estimator__8 -> my_estimator)
    data[["estimator.name", "estimator.level"]] = (
        data["estimator.name"].str.extract("(\w+)__(\d+)")
    )
    data["estimator.level"] = data["estimator.level"].astype(int)

    if "wrapper.name" not in data.columns:
        data["wrapper.name"] = np.nan

    print("Estimators internally scored:", *level_scores_data["estimator.name"].unique())

    level_scores_data["estimator.name"] = (
        level_scores_data["estimator.name"].str.removesuffix("__10")
    )

    # Combine with internal Cascade scores for each level
    data = data.merge(
        right=level_scores_data,
        on=[
            "estimator.name",
            "estimator.level",
            "wrapper.name",
            "dataset.name",
            "cv.fold",
        ],
        how="outer"
    )

    print("Estimators:", *data["estimator.name"].unique())

    # Rename estimators to include wrapper name
    data["suffix"] = (
        data["wrapper.name"].map(wrapper_renaming).fillna(data["wrapper.name"])
    )
    data["estimator.name"] += data["suffix"]
    data.to_csv(output_path, sep="\t", index=False)

    print("Estimators with wrappers:", *data["estimator.name"].unique())

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
    parser.add_argument(
        "level_scores",
        type=Path,
        help="Path to the output of 'parse_level_scores.py'.",
    )
    args = parser.parse_args()

    rename_estimators(args.results, args.level_scores)


if __name__ == "__main__":
    main()
