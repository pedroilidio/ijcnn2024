from argparse import ArgumentParser
from itertools import zip_longest
from pathlib import Path
import warnings
import yaml
import pandas as pd


def main(estimators: list[str] | None = None):
    if estimators:
        print(f"Keeping only the following estimators: {estimators}")

    selected_cols = [
        "end",
        "hash",
        "start",
        "dataset.name",
        "estimator.call",
        "estimator.name",
        # 'estimator.params.min_samples_leaf',
        # 'estimator.params.n_estimators',
        # 'cv.params.scoring',
    ]

    rows = []
    for p in Path("runs").glob("*"):
        with p.open() as f:
            try:
                run_data = yaml.unsafe_load(f)
            except yaml.constructor.ConstructorError as e:
                warnings.warn(f"Could not load run {p}: {e}")
                continue
            if "results" not in run_data:
                continue
            if estimators and run_data["estimator"]["name"] not in estimators:
                continue

        row = pd.json_normalize(run_data, sep=".", max_level=2)
        row = row.loc[
            :, row.columns.isin(selected_cols) | row.columns.str.startswith("results")
        ]
        rows.append(row)

    df = pd.concat(rows)

    #  Keep only the most recent run from each estimator on each dataset
    df = df.sort_values("start").drop_duplicates(  # TODO: sort within groups
        ["dataset.name", "estimator.name"], keep="last"
    )
    # Use MultiIndex instead of column names such as "dataset.name"
    df.columns = pd.MultiIndex.from_arrays(
        zip_longest(*df.columns.str.split("."), fillvalue="")
    )
    df = df.explode(df.loc[:, ("results", slice(None))].columns.to_list())
    df.insert(0, ("cv", "fold"), df.groupby("hash").cumcount())

    df.to_csv("results.tsv", index=False, sep="\t")

    return df


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "estimators",
        nargs="*",
        help=(
            "Estimators to include in the table. If not specified, all"
            " estimators are included."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(**vars(parse_args()))
