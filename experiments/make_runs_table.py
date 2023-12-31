import argparse
from pathlib import Path
import re
import warnings

import yaml
import pandas as pd
from tqdm import tqdm


def sanitize_yaml(yaml_string):
    # indent includes the newline
    pattern = re.compile(r"(?P<indent>\s*)(?P<key>.*?)!!python/(?P<call>\S+)(.*)")
    return pattern.sub(
        r'\g<indent>\g<key>\g<indent>  call: "\g<call>"',
        yaml_string,
    )


def make_runs_table(
    *,
    estimators: list[str] | None = None,
    outpath: Path | str = "results.tsv",
    run_locations: list[Path] | None = None,
):
    if estimators:
        print(f"Keeping only the following estimators: {estimators}")

    # Gather file paths for each run
    run_paths = []
    for p in run_locations:
        if p.is_dir():
            run_paths.extend(p.glob("**/*.yaml"))
            run_paths.extend(p.glob("**/*.yml"))
        else:
            run_paths.append(p)

    print("Gathering data from the following locations:", *run_locations, sep="\n  * ")

    rows = []
    for p in tqdm(run_paths):
        with p.open() as f:
            try:
                # Remove slow and unsafe !!python directives (no need to load objects)
                # content = sanitize_yaml(f.read())  # TODO
                content = f.read()
                run_data = yaml.unsafe_load(content)
            except yaml.constructor.ConstructorError as e:
                warnings.warn(f"Could not load run {p}: {e}")
                continue
            if "results" not in run_data:
                continue
            if estimators and run_data["estimator"]["name"] not in estimators:
                continue

        row = pd.json_normalize(run_data, sep=".", max_level=2)
        rows.append(row)

    print("Building table...")
    df = pd.concat(rows)

    # Keep only the most recent run from each estimator on each dataset
    print("Discarding old runs...")
    df = (
        df
        .groupby("hash")
        .apply(lambda g: g.sort_values("start"))  # Sort runs within each hash
        .groupby(level=0)  # Group by hash again
        .last()  # Keep only the most recent run
    )

    # # Use MultiIndex instead of column names such as "dataset.name"
    # df.columns = pd.MultiIndex.from_arrays(
    #     zip_longest(*df.columns.str.split("."), fillvalue="")
    # )
    result_cols = df.columns[df.columns.str.startswith("results.")].to_list()
    df = df.dropna(subset=result_cols, axis=0, how="all")
    df = df.dropna(axis=1, how="all")
    result_cols = df.columns.intersection(result_cols).to_list()

    # Make sure all lengths match before exploding
    # (missing folds are filled with lists of None)
    def set_iterable_nan(col):
        if not col.isna().any():
            return col
        n_folds = int(col.dropna().str.len().max())
        new_col = col.copy()
        # List of lists because pandas cannot set a list to each position directly.
        new_col.loc[new_col.isna()] = [[None] * n_folds] * new_col.isna().sum()
        return new_col

    df.loc[:, result_cols] = df[result_cols].apply(set_iterable_nan)

    df = df.explode(result_cols)
    df["cv.fold"] = df.groupby(level=0).cumcount()
    # drop=true because 'hash' also a column already.
    df = df.reset_index(drop=True).sort_index(axis=1)

    # Compute combined time to fit and score
    df.loc[:, "results.fit_score_time"] = (
        df["results.score_time"] + df["results.fit_time"]
    )

    df.to_csv(outpath, index=False, sep="\t")
    print(f"Saved to {outpath}")

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Build a table of results from the runs yaml outputs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--estimators",
        nargs="+",
        help=(
            "Estimators to include in the table. If not specified, all"
            " estimators are included."
        ),
    )
    parser.add_argument(
        "--out",
        "-o",
        type=Path,
        default=Path("results.tsv"),
        help="Path to save the table to.",
    )
    parser.add_argument(
        "--runs",
        nargs="+",
        type=Path,
        default=[Path("runs")],
        help=(
            "Path(s) to the runs' output yaml files (or to directories"
            " containing them)."
        ),
    )
    args = parser.parse_args()

    make_runs_table(
        estimators=args.estimators,
        outpath=args.out,
        run_locations=args.runs,
    )


if __name__ == "__main__":
    main()
