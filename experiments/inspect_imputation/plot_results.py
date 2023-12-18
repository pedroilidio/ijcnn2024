#!/usr/bin/env python
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


def plot_results(input_tables_dir: Path, out_dir: Path):
    data = [
        pd.read_csv(f, sep="\t").assign(
            dataset=f.stem.split("__")[0],
            estimator=f.stem.split("__")[1],
        )
        for f in input_tables_dir.glob("*.tsv")
    ]
    data = pd.concat(data) if len(data) > 1 else data[0]
    metrics = ["tp", "fn", "fp"]
    # FIXME: some metrics, such as precision_micro and f1_micro have equal values?

    if data.dataset.nunique() > 1:
        # Combine all datasets to also calculate the metrics for them together
        allsets = data.copy()
        allsets[metrics + ["tn"]] = allsets[metrics + ["tn"]].div(
            allsets[metrics + ["tn"]].sum(axis=1),
            axis=0,
        )
        allsets = (
            allsets
            .groupby(['level', 'estimator', 'dataset'])
            .mean()
            .reset_index()
        )
        allsets = allsets.assign(dataset="all", fold=allsets.dataset)
        data = pd.concat([data, allsets])

    out_dir.mkdir(exist_ok=True, parents=True)

    for gname, group in tqdm(list(data.groupby(["dataset", "estimator"]))):
        lines = group.drop(columns="fold").groupby("level").mean(numeric_only=True)
        lines['precision_missing'] = lines.tp / (lines.tp + lines.fp)
        # Just the blue area border:
        # lines['recall_missing'] = lines.tp / (lines.tp + lines.fn)

        sns.set_theme()
        ax = lines.plot.area(y=metrics)
        for col, style in (
            ('precision_missing', '-'),
            ('precision_micro', '--'),
            ('average_precision_micro', ':'),
            ('matthews_corrcoef_micro', '-.'),
        ):
            ax = lines.plot(
                ax=ax,
                y=col,
                color='black',
                linewidth=2,
                linestyle=style,
                secondary_y=True,
                ylim=(0, 1),
            )
        plt.title(f"Dataset: {gname[0]} | Estimator: {gname[1]}")
        plt.tight_layout()

        out_path = out_dir / f"{gname[0]}__{gname[1]}.png"
        plt.savefig(out_path, dpi=300)
        plt.savefig(
            out_path.with_suffix(".pdf"),
            bbox_inches="tight",
            transparent=True,
        )
        plt.close()


def main():
    parser = ArgumentParser()
    parser.add_argument("--input_tables_dir", type=Path, default="results")
    parser.add_argument("--out_dir", type=Path, default="plots")
    args = parser.parse_args()

    plot_results(
        input_tables_dir=args.input_tables_dir,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
