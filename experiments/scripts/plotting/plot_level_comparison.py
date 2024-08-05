#!/usr/bin/env python3
"""
Plot performances per level.

Not explored in the paper, but can be used to generate plots comparing the
performance of the estimators under study at different levels of the
cascade. The resulting plots show how the training, test, and OOB performance
varies across the levels of the deep forests.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


# TODO: refactoring. modularize and parametrize this script
def main():
    idx = pd.IndexSlice
    outdir = Path(__file__).parents[2] / "results/level_comparison"
    outdir_conf_matrix = outdir / "confusion_matrix_plots"
    outdir_imputation = outdir / "imputation_plots"
    print("Reading data")
    data = pd.read_table(outdir.parents[1] / 'results/results_renamed.tsv')
    label_freq_columns = (
        data.loc[:, data.columns.str.endswith("label_frequency_estimates_")]
        .map(lambda x: np.fromstring(x[1:-1], sep=",") if isinstance(x, str) else x)
    )

    data["label_freq"] = label_freq_columns.apply(
        lambda s: np.nan if s.isna().all() else s.loc[s.first_valid_index()],
        axis=1,
    )
    data["avg_label_freq"] = data["label_freq"].copy()
    data.loc[~data.avg_label_freq.isna(), "avg_label_freq"] = (
        data.loc[~data.avg_label_freq.isna(), "avg_label_freq"].map(lambda x: x.mean())
    )
    data["avg_label_freq"].iloc[40:60]
    data = data.loc[data.level.astype(str).str.match(r"^\d+$")]
    data.loc[:, "level"] = data.loc[:, "level"].astype(int)
    data["estimator.name"] = data["original_estimator_name"] + "__" + data["wrapper.name"].astype(str)
    data = (
        data.set_index([
            "cv.fold", "original_estimator_name", "dataset.name",
            "avg_label_freq", "level",
        ])
        .rename_axis(["fold", "estimator", "dataset", "avg_label_freq", "level"])
    )
    score_columns_mask = data.columns.str.match(r"results\.(test|train).*")
    data = data.loc[:, score_columns_mask]
    metric = (
        data.columns#[score_columns_mask]
        .str.removeprefix("results.")
        .str.removeprefix("train_")
        .str.removeprefix("test_")
        .str.removesuffix("_internal")
        .str.removesuffix("_oob")
        .str.removesuffix("_masked")
        .rename("metric")
        # .to_series()
        # .set_axis(expected_best.index)
    )

    data.columns = pd.MultiIndex.from_tuples(
        ((m, s.replace("_" + m, "").removeprefix("results."))
        for m, s in zip(metric, data.columns)),
        names=("metric", "scorer"),
    )
    data = data.sort_index(axis=1)
    filtered_data = data.loc[:, pd.IndexSlice[["neg_hamming_loss_micro", "average_precision_micro", "roc_auc_micro"], :]]
    # filtered_data = filtered_data.filter(like="cascade_proba", axis=0)
    filtered_data.reset_index().estimator.value_counts()
    print("Building allsets")
    allsets = (
        filtered_data.stack(["metric", "scorer"])
        .droplevel("avg_label_freq", axis=0)
        .groupby(level=["estimator", "dataset", "fold", "metric", "scorer"])
        .rank(pct=True)  # Rank by cascade level
        .groupby(level=["estimator", "dataset", "metric", "scorer", "level"])
        .mean()  # Average over folds
    )
    grouped = allsets.groupby(level=["estimator", "metric"])#, "scorer"]):
    for gname, g in tqdm(grouped, total=grouped.ngroups):
        g = g.reset_index()
        g = g[g["scorer"].isin(("test", "train_internal", "train_oob_internal"))]
        sns.boxplot(g, x="level", y=0, hue="scorer", showfliers=False)
        # sns.swarmplot(g, x="level", y=0, hue="scorer", palette="dark:k", dodge=True, size=2, legend=False)
        sns.stripplot(g, x="level", y=0, hue="scorer", palette="dark:k", dodge=True, size=2, legend=False)
        plt.title(f"{' | '.join(gname)}")
        outpath = outdir / ("boxplots/" + "/".join(gname) + ".png")
        outpath.parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(outpath)
        plt.close()


    # =============================================================================
    # Imputation plots
    # =============================================================================

    imputation_scores = data.loc[
        data.index.get_level_values("level") > 0,
        idx[
            [
                "tp_micro", "tn_micro", "fp_micro", "fn_micro", "recall_micro",
                "precision_micro", "average_precision_micro", "roc_auc_micro",
            ],
            ["train", "train_internal", "train_masked"],
        ]
    ]
    # imputation_scores = imputation_scores.filter(like="flaforest", axis=0)
    imputation_scores = imputation_scores.dropna(how="any", axis=0)

    imputation_scores[("expected_p", "expected_p")] = (
        imputation_scores.loc[:, idx[["tp_micro", "fn_micro"], "train_internal"]].abs().sum(axis=1)
        / imputation_scores.reset_index("avg_label_freq")["avg_label_freq"]
    ).values

    # Train internal scores are computed after dropping labels to simulate missing
    # data.
    p = imputation_scores.loc[:, idx[["tp_micro", "fn_micro"], "train"]].abs().sum(axis=1)
    # p = imputation_scores.loc[:, idx[["tp_micro", "fn_micro"], "train_internal"]].abs().sum(axis=1)
    # imputation_scores["positives"] = p.values

    allsets_imputation = (
        imputation_scores.abs()
        .div(p, axis=0)  # Normalize by number of labels
        .groupby(level=["estimator", "dataset", "level"])
        .mean()  # Average over folds
        .groupby(level=["estimator", "level"])
        .mean()  # Average over datasets
    )

    print("Generating imputation plots")
    grouped = allsets_imputation.groupby(level="estimator")
    for gname, g in tqdm(grouped, total=grouped.ngroups):
        g = (
            g.loc[:, idx[:, ["train", "expected_p"]]]
            .droplevel("scorer", axis=1)
            .reset_index("level")
        )
        ax = g.plot.area(
            x="level",
            y=["tp_micro", "fp_micro"],
            title=gname,
            stacked=True,
            alpha=0.5,
        )
        g.plot(
            ax=ax,
            x="level",
            y="expected_p",
            color="k",
            linestyle="--",
        )
        ax.axhline(1, color="k", linestyle="-")
        ax.set_ylim()
        outdir_conf_matrix.mkdir(exist_ok=True)
        plt.savefig(outdir_conf_matrix / f"{gname}.png")
        plt.close()


    # Scores considering only imputted labels

    imputation_scores[("expected_p", "expected_p")] = (
        imputation_scores.loc[:, idx[["tp_micro", "fn_micro"], "train_internal"]].abs().sum(axis=1)
        * (1 / imputation_scores.reset_index("avg_label_freq")["avg_label_freq"] - 1)
    ).values

    imputation_scores[("expected_p", "expected_p")]  /= (
        imputation_scores[("expected_p", "expected_p")]
        + imputation_scores.loc[:, idx[["tn_micro", "fp_micro"], "train_internal"]].abs().sum(axis=1)
    )

    # Train internal scores are computed after dropping labels to simulate missing
    # data.
    p = imputation_scores.loc[:, idx[["tp_micro", "fn_micro"], "train_masked"]].abs().sum(axis=1)
    # p = imputation_scores.loc[:, idx[["tp_micro", "fn_micro"], "train_internal"]].abs().sum(axis=1)
    # imputation_scores["positives"] = p.values

    allsets_imputation = (
        imputation_scores.abs()
        .div(p, axis=0)  # Normalize by number of labels
        .groupby(level=["estimator", "dataset", "level"])
        .mean()  # Average over folds
        .groupby(level=["estimator", "level"])
        .mean()  # Average over datasets
    )

    print("Generating imputation plots")
    grouped = allsets_imputation.groupby(level="estimator")
    for gname, g in tqdm(grouped, total=grouped.ngroups):
        g = (
            g.loc[:, idx[:, ["train_masked", "expected_p"]]]
            .droplevel("scorer", axis=1)
            .reset_index("level")
        )
        ax = g.plot.area(
            x="level",
            y=["tp_micro", "fp_micro"],
            title=gname,
            stacked=True,
            alpha=0.5,
        )
        g.plot(
            ax=ax,
            x="level",
            y="expected_p",
            color="k",
            linestyle="--",
        )
        ax.axhline(1, color="k", linestyle="-")
        ax.set_ylim()
        outdir_imputation.mkdir(exist_ok=True)
        plt.savefig(outdir_imputation / f"{gname}.png")
        plt.close()


if __name__ == "__main__":
    main()
