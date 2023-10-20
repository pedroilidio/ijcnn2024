from argparse import ArgumentParser
from pathlib import Path
import pandas as pd
import yaml
from tqdm import tqdm
from run_experiments import load_dataset, deep_update


def collect_dataset_metadata(config_file, output_file, unsafe_yaml=False):
    print(f"Loading datasets from {config_file}.")
    yaml_loader = yaml.unsafe_load if unsafe_yaml else yaml.safe_load
    with open(config_file, "r") as f:
        config = yaml_loader(f)

    datasets_load_info = deep_update(
        config["defaults"]["aliases"]["dataset"],
        config["aliases"]["dataset"],
    )
    datasets = tqdm(
        map(load_dataset, datasets_load_info),
        total=len(datasets_load_info),
    )
    # Define column names and types
    columns = {
        "n_samples": int,
        "n_features": int,
        "n_labels": int,
        "y_size": int,
        "micro_label_density": float,
        "macro_label_density": float,
        "samples_label_density": float,
    }

    # Create empty DataFrame with column names and types
    metadata = pd.DataFrame(columns=columns.keys())

    for dataset in tqdm(datasets):
        name, X, y = dataset["name"], dataset["X"], dataset["y"]
        metadata.loc[name, :] = (
            X.shape[0],
            X.shape[1],
            y.shape[1],
            y.size,
            y.mean(),
            y.mean(0).mean(),
            y.mean(1).mean(),
        )
    metadata = metadata.astype(columns).sort_values("y_size")

    print(metadata)
    metadata.to_csv(output_file, sep="\t")
    print(f"Saved dataset metadata to {output_file}.")


def main():
    argparser = ArgumentParser()
    argparser.add_argument(
        "--config",
        "-c",
        type=Path,
        default=Path("config.yml"),
        help="Path to run_experiments.py's config file.",
    )
    argparser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("dataset_info.tsv"),
        help="Path to tabular output file.",
    )
    argparser.add_argument(
        "--unsafe-yaml",
        action="store_true",
        help="Use PyYAML unsafe loader.",
    )
    args = argparser.parse_args()
    collect_dataset_metadata(args.config, args.output, args.unsafe_yaml)


if __name__ == "__main__":
    main()
