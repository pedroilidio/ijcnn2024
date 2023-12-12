import sys
import warnings
from pathlib import Path
import argparse

import joblib
import pandas as pd
from imblearn.pipeline import Pipeline
from tqdm import tqdm
import yaml

from deep_forest.cascade import Cascade

# Allow joblib to find nakano_datasets_v2
sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_level_scores(dir_models: Path, output_path: Path):
    score_records = []

    for path_model in tqdm(list(dir_models.glob("*.joblib"))):
        models = joblib.load(path_model)

        with open(path_model.with_suffix(".yml"), "r") as file_metadata:
            metadata = yaml.unsafe_load(file_metadata)

        score_records.extend(
            {
                "estimator": metadata["estimator"]["name"],
                "dataset": metadata["dataset"]["name"],
                "level": level,
                "fold": fold,
            } | level_scores
            for fold, model in enumerate(models)
            for level, level_scores  in enumerate(model.level_scores_)
        )

    df_scores = pd.DataFrame.from_records(score_records)
    df_scores.to_csv(output_path, index=False, sep="\t")
    print(f"Saved level scores to '{output_path}'.")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Parse level scores from cascade forest models."
            "This script takes a directory path containing the cascade forest"
            " models and outputs the parsed level scores to a specified file. Each"
            " model's level scores are extracted and saved in a tab-separated"
            " value (tsv) file format."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "dir_models", type=Path,
        help=(
            "Directory path containing the cascade forest models stored as"
            " '.joblib' files"
        ),
    )
    parser.add_argument(
        "--out", "-o", type=Path, default="level_scores.tsv",
        help="Output file path for the parsed level scores",
    )
    args = parser.parse_args()

    parse_level_scores(args.dir_models, args.out)


if __name__ == "__main__":
    main()
