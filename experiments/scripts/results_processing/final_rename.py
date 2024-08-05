from argparse import ArgumentParser
from pathlib import Path

import pandas as pd

MAPPINGS = {
    r"cafe(__\d+)?__level0": r"RF+ET\1",
    r"gcforest(__\d+)?__max_ascending_oob": r"gcForest\1",
    r"lcforest(__\d+)?__finished_imputation": r"LCForest\1",
    r"slcforest(__\d+)?__best_oob": r"SLCForest\1",
    r"flaforest(__\d+)?__best_oob": r"FLAForest\1",
    r"cafe_slc(__\d+)?__best_oob": r"CaFE-SLC\1",
    r"cafe_fla(__\d+)?__best_oob": r"CaFE-FLA\1",
    r"cafe(__\d+)?__best_train": r"CaFE\1",
    r"cafe_os(__\d+)?__best_train": r"CaFE-OS\1",
}


def final_rename(path_results: Path, path_out: Path | None):
    path_out = path_out or path_results.with_stem(path_results.stem + '_final')

    print(f"Renaming estimators in {path_results} and saving to {path_out}.")

    print("Loading data...")
    df = pd.read_table(path_results)

    print("Renaming...")
    df.loc[:, "processed_name"] = df["estimator.name"]
    df.loc[:, "estimator.name"] = df.processed_name.replace(MAPPINGS, regex=True)

    print("Saving...")
    df.to_csv(path_out, sep='\t', index=False)

    print("Done.")
    return path_out


def main():
    argparser = ArgumentParser()
    argparser.add_argument("results", type=Path)
    argparser.add_argument("--out", "-o", default=None, type=Path)
    args = argparser.parse_args()
    final_rename(args.results, args.out)


if __name__ == '__main__':
    main()
