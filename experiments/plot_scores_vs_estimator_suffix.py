from itertools import combinations
import logging
from pathlib import Path
import scipy.stats
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import scikit_posthocs as sp
from statannotations.Annotator import Annotator
from statsmodels.stats.multitest import multipletests

outdir = Path("figures")


def plot_crossbars(
    ax,
    data,
    y,
    x,
    hue,
    block_col,
    p_adjust,
    alpha=0.05,
    **kwargs,
):
    omnibus_pvalues = {
        g_name: scipy.stats.friedmanchisquare(
            *g.pivot(index=hue, columns=[block_col], values=y).values
        ).pvalue
        for g_name, g in data.groupby(x)
        if len(g[hue].unique()) >= 4
    }

    if omnibus_pvalues:
        # Correct omnibus p-values for multiple testing
        corrected_omnibus_pvalues = multipletests(
            list(omnibus_pvalues.values()),
            method=p_adjust,
        )[1]

        omnibus_pvalues = {
            g_name: pvalue
            for g_name, pvalue in zip(
                omnibus_pvalues.keys(),
                corrected_omnibus_pvalues,
            )
        }

        print(f"{corrected_omnibus_pvalues=}")

    data = data.copy()
    data["block_col"] = data[block_col]
    data["hue"] = data[hue]
    data["x"] = data[x]
    data["y"] = data[y]
    data = data.droplevel(1, axis=1)
    data["joint_group"] = data.hue.astype(str) + "__" + data.block_col.astype(str)

    pvalue_crosstables = {}
    for x_name, x_group in data.groupby("x"):
        if len(x_group.hue.unique()) < 2:
            continue
        if x_name not in omnibus_pvalues:
            try:
                pvalue_crosstables[x_name] = sp.posthoc_wilcoxon(
                    x_group,
                    val_col="y",
                    group_col="hue",  # FIXME
                    # group_col='joint_group',  # FIXME
                    p_adjust=p_adjust,
                )
            except ValueError:
                logging.warning(f"Could not compute p-values for {x_name}")
        elif omnibus_pvalues[x_name] < alpha:
            try:
                pvalue_crosstables[x_name] = sp.posthoc_conover_friedman(
                    x_group,
                    melted=True,
                    y_col="y",
                    group_col="hue",
                    block_col="block_col",
                    p_adjust=p_adjust,
                )
            except ZeroDivisionError:
                logging.warning(f"Could not compute p-values for {x_name}")

    if not pvalue_crosstables:
        return

    pairs = [
        ((xi, h1), (xi, h2))
        for xi in data.x.unique()
        for h1, h2 in combinations(
            # Skip estimators if all scores are NaN
            data.loc[(data.x == xi) & ~data.y.isna(), "hue"].unique(),
            2,
        )
        if xi in pvalue_crosstables
    ]
    pvalues = [
        pvalue_crosstables[pair[0][0]].loc[pair[0][1], pair[1][1]] for pair in pairs
    ]

    # Filter out non-significant pairs
    pairs = [pair for pair, p in zip(pairs, pvalues) if p < alpha]
    if not pairs:
        return
    pvalues = [p for p in pvalues if p < alpha]

    annotator = Annotator(
        ax=ax,
        pairs=pairs,
        data=data,
        x="x",
        y="y",
        hue="hue",
    )

    annotator.set_pvalues_and_annotate(pvalues)


def plot_comparisons(data, x, hue, outdir):
    for dataset, dataset_data in data.groupby(("dataset", "name")):
        for metric in dataset_data.results.columns:
            print(f"\n[main] Processing {dataset=} {metric=}\n")

            plt.figure(
                figsize=(
                    len(dataset_data[x].unique())
                    * len(dataset_data[hue].unique())
                    * 0.3
                    + 1,
                    3,
                )
            )

            ax = sns.boxplot(
                data=dataset_data,
                x=x,
                hue=hue,
                y=("results", metric),
            )
            sns.stripplot(
                ax=ax,
                data=dataset_data,
                x=x,
                hue=hue,
                y=("results", metric),
                dodge=True,
                palette="dark:k",
                legend=False,
                marker="o",
                size=3,
            )
            plot_crossbars(
                ax=ax,
                data=dataset_data,
                x=x,
                hue=hue,
                y=("results", metric),
                block_col=("cv", "fold"),
                p_adjust="holm",
            )

            ax.legend(loc="lower left", bbox_to_anchor=(1, 0))
            plt.xticks(rotation=45, ha="right")
            plt.title(dataset)
            plt.ylabel(metric)
            plt.tight_layout()
            plt.savefig(outdir / f"{dataset}__{metric}.png")
            plt.savefig(
                outdir / f"{dataset}__{metric}.svg",
                transparent=True,
                bbox_inches="tight",
            )
            plt.close()


def main(outdir=outdir):
    data = pd.read_table("results.tsv", header=[0, 1])
    data[["estimator_name", "estimator_suffix"]] = data.estimator.name.str.rsplit(
        "__", n=1
    ).apply(pd.Series)

    outdir_name = outdir / "estimator_name"
    outdir_suffix = outdir / "estimator_suffix"

    outdir_name.mkdir(parents=True, exist_ok=True)
    outdir_suffix.mkdir(exist_ok=True)

    joined_datasets = data.copy()
    joined_datasets[("dataset", "name")] = "all_datasets"
    joined_datasets[("cv", "fold")] = data.groupby(
        [("dataset", "name"), ("cv", "fold")]
    ).ngroup()
    joined_datasets["results"] = (
        data["results"].groupby(joined_datasets.cv.fold).rank(pct=True)
    )

    plot_comparisons(
        data,
        x="estimator_suffix",
        hue="estimator_name",
        outdir=outdir_suffix,
    )
    plot_comparisons(
        data,
        x="estimator_name",
        hue="estimator_suffix",
        outdir=outdir_name,
    )

    plot_comparisons(
        joined_datasets,
        x="estimator_suffix",
        hue="estimator_name",
        outdir=outdir_suffix,
    )
    plot_comparisons(
        joined_datasets,
        x="estimator_name",
        hue="estimator_suffix",
        outdir=outdir_name,
    )


if __name__ == "__main__":
    main()
