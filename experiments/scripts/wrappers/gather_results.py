#!/usr/bin/env python3
"""Compile results from runs into a single table.

There are three steps to this script:

    1. Gather results from the runs into a single table.
    2. Reformat the table, renaming the estimators in the table to account for
       the multiple levels of the cascades.
    3. Rename the estimators so that the names are nicely formatted for the
       published plots.
"""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[2]))
from scripts.results_processing.make_runs_table import make_runs_table
from scripts.results_processing.rename_estimators import rename_estimators
from scripts.results_processing.final_rename import final_rename

OUTDIR = "results"
RUNS = (Path("runs"),)


def gather_results(
    *,
    outdir: str | Path = OUTDIR,
    runs: list[Path] = RUNS,
):
    """Compile results from runs into a single table.

    There are three steps to this script:

        1. Gather results from the runs into a single table.
        2. Reformat the table, renaming the estimators in the table to account
           for the multiple levels of the cascades.
        3. Rename the estimators so that the names are nicely formatted for the
           published plots.

    Parameters
    ----------
    outdir : str or Path
        Output directory for the result tables.
    runs : list of Path
        Run files or directories containing the runs.
    """
    runs = list(runs)
    outdir = Path(outdir)
    first_outpath = outdir / "results.tsv"

    # The three steps mentioned in the module's docstring
    make_runs_table(outpath=first_outpath, run_locations=runs)
    path_renamed = rename_estimators(first_outpath)
    path_renamed_final = final_rename(path_renamed)

    return path_renamed_final


def parse_args():
    parser = argparse.ArgumentParser(description="Gather results from runs.")
    parser.add_argument(
        "--outdir",
        "-o",
        type=str,
        default=OUTDIR,
        help="Output directory for the result tables.",
    )
    parser.add_argument(
        "--runs",
        "-r",
        type=Path,
        nargs="+",
        default=list(RUNS),
        help="Run files or directories containing the runs.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    gather_results(outdir=args.outdir, runs=args.runs)


if __name__ == "__main__":
    main()
