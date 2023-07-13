import argparse
import itertools
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

from scipy import stats
from statsmodels.stats.multitest import multipletests
import scikit_posthocs as sp

import matplotlib.pyplot as plt
import seaborn as sns
from critical_difference_diagrams import (
    plot_critical_difference_diagram,
    _find_maximal_cliques,
)


def plot_insignificance_bars(positions, sig_matrix, ax=None, **kwargs):
    ax = ax or plt.gca()
    ylim = ax.get_ylim()
    ystart = ylim[1]
    crossbars = []
    crossbar_props = {"marker": ".", "color": "k"} | kwargs
    bar_margin = 0.1 * (ylim[1] - ylim[0])

    positions = pd.Series(positions)  # Standardize if ranks is dict

    # True if pairwise comparison is NOT significant
    adj_matrix = pd.DataFrame(
        1 - sp.sign_array(sig_matrix),
        index=sig_matrix.index,
        columns=sig_matrix.columns,
        dtype=bool,
    )
    crossbar_sets = _find_maximal_cliques(adj_matrix)

    # Sort by lowest rank and filter single-valued sets
    crossbar_sets = sorted(
        (x for x in crossbar_sets if len(x) > 1), key=lambda x: positions[list(x)].min()
    )

    # Create stacking of crossbars: for each level, try to fit the crossbar,
    # so that it does not intersect with any other in the level. If it does not
    # fit in any level, create a new level for it.
    crossbar_levels: list[list[set]] = []
    for bar in crossbar_sets:
        for level, bars_in_level in enumerate(crossbar_levels):
            if not any(bool(bar & bar_in_lvl) for bar_in_lvl in bars_in_level):
                ypos = ystart + bar_margin * (level + 1)
                bars_in_level.append(bar)
                break
        else:
            ypos = ystart + bar_margin * (len(crossbar_levels) + 1)
            crossbar_levels.append([bar])

        crossbars.append(
            ax.plot(
                # Adding a separate line between each pair enables showing a
                # marker over each elbow with crossbar_props={'marker': 'o'}.
                [positions[i] for i in bar],
                [ypos] * len(bar),
                **crossbar_props,
            )
        )

    return crossbars


def make_latex_table(
    data,
    block_col,
    group_col,
    metric,
    sig_matrix,
    positions,
    round_digits=2,
    highlight_best=True,
    higher_is_better=True,
):
    # True if pairwise comparison is NOT significant
    adj_matrix = pd.DataFrame(
        1 - sp.sign_array(sig_matrix),
        index=sig_matrix.index,
        columns=sig_matrix.columns,
        dtype=bool,
    )

    table = data.set_index(block_col).groupby(group_col)[metric].agg(["mean", "std"]).T
    text_table = (
        table.round(round_digits).astype(str).apply(lambda r: "{} ({})".format(*r))
    )

    if not highlight_best:
        return text_table

    crossbar_sets = _find_maximal_cliques(adj_matrix)

    # Get top-ranked set and filter single-valued sets
    # TODO: no sorting needed.
    top_crossbar_set = list(
        list(
            sorted(
                (x for x in crossbar_sets if len(x) > 1),
                key=lambda x: positions[list(x)].min(),
            )
        )[-1 if higher_is_better else 0]
    )

    # Highlight top-ranked set, if it is not the only set
    if len(top_crossbar_set) < len(text_table):
        text_table[top_crossbar_set] = text_table[top_crossbar_set].apply(
            lambda s: f"\\textbf{{{s}}}"
        )

    return text_table


def iter_posthoc_comparisons(
    data,
    *,
    y_cols,
    group_col,
    block_col,
    p_adjust,
):
    all_blocks = set(data[block_col].unique())
    missing_blocks = (
        data.groupby(group_col)[block_col].unique().apply(lambda x: all_blocks - set(x))
    )
    missing_blocks = missing_blocks.loc[missing_blocks.apply(len) != 0]

    if not missing_blocks.empty:
        warnings.warn(
            "The following groups have missing blocks and will be removed"
            f" from the comparison analysis:\n{missing_blocks}"
        )
        data = data[~data[group_col].isin(missing_blocks.index)]

    for metric in y_cols:
        print("- Processing metric:", metric)
        pvalue_crosstable = sp.posthoc_conover_friedman(
            data,
            melted=True,
            y_col=metric,
            group_col=group_col,
            block_col=block_col,
            p_adjust=p_adjust,
        )
        mean_ranks = (
            data.set_index([block_col, group_col])[metric]
            .groupby(level=0)
            .rank(pct=True)
            .groupby(level=1)
            .mean()
        )
        yield metric, pvalue_crosstable, mean_ranks


def make_visualizations(
    data,
    group_col,
    pvalue_crosstable,
    mean_ranks,
    outdir,
    prefix,
    metric,
    omnibus_pvalue,
):
    pvalue_crosstable.to_csv(outdir / (f"{prefix}__{metric}.tsv"), sep="\t")

    n_groups = pvalue_crosstable.shape[0]

    plt.figure(figsize=[(n_groups + 2) / 2.54] * 2)
    plt.title(f"{metric} (p = {omnibus_pvalue:.2e})", wrap=True)
    ax, cbar = sp.sign_plot(
        pvalue_crosstable,
        annot=sp.sign_table(pvalue_crosstable),
        fmt="s",
        square=True,
    )
    cbar.remove()
    plt.tight_layout()
    plt.savefig(
        outdir / (f"{prefix}__{metric}.png"),
        transparent=True,
        # bbox_inches='tight',  # Allow for title wrapping
    )
    plt.close()

    plt.figure(figsize=(6, 0.5 * n_groups / 2.54 + 1))
    plot_critical_difference_diagram(
        mean_ranks,
        pvalue_crosstable,
        crossbar_props={"marker": "."},
    )
    plt.title(f"{metric} (p = {omnibus_pvalue:.2e})", wrap=True)
    plt.tight_layout()
    plt.savefig(
        outdir / (f"{prefix}__cdd__{metric}.png"),
        transparent=True,
        # bbox_inches='tight',  # Allow for title wrapping
    )
    plt.close()

    plt.figure(figsize=(0.3 * n_groups + 1, 3))
    ax = sns.boxplot(
        data=data,
        x=group_col,
        y=metric,
        order=mean_ranks.sort_values().index,
    )
    sns.stripplot(
        ax=ax,
        data=data,
        x=group_col,
        y=metric,
        order=mean_ranks.sort_values().index,
        color="k",
        marker="o",
        size=3,
    )
    plot_insignificance_bars(
        positions=mean_ranks.rank(method="first") - 1,
        sig_matrix=pvalue_crosstable,
    )
    plt.xticks(rotation=45, ha="right")
    plt.title(f"{metric} (p = {omnibus_pvalue:.2e})", wrap=True)
    plt.tight_layout()
    plt.savefig(
        outdir / (f"{prefix}__boxplot__{metric}.png"),
        transparent=True,
        # bbox_inches='tight',  # Allow for title wrapping
    )
    plt.close()


def friedman_melted(data, *, index, columns, values):
    # Expand ("unmelt") to 1 fold per column on level 2, metrics on level 1
    pivot = data.pivot(index=index, columns=columns, values=values)

    # Apply Friedman's test for each result metric
    result = (
        pivot.groupby(level=1, axis=1)
        .apply(lambda x: stats.friedmanchisquare(*(x.values))._asdict())
        .apply(pd.Series)
    )  # Convert dicts to rows

    return result


def plot_comparison_matrix(comparison_data: pd.DataFrame):
    comparison_table = comparison_data.unstack()
    order = comparison_table.mean(1).sort_values(ascending=False).index

    comparison_table = comparison_table.loc[:, (slice(None), order)]
    comparison_table = comparison_table.loc[order]
    # comparison_table = comparison_table.loc[comparison_table.isna().sum(1).sort_values().index]
    # comparison_table = comparison_table.loc[:, comparison_table.isna().sum(0).sort_values().index]
    sns.heatmap(comparison_table.effect_size, annot=True)


def main(estimator_subset=None, dataset_subset=None, metric_subset=None):
    df = pd.read_table("results.tsv", header=[0, 1])

    if estimator_subset is not None:
        df = df[df[("estimator", "name")].isin(estimator_subset)]
    if dataset_subset is not None:
        df = df[df[("dataset", "name")].isin(dataset_subset)]
    if metric_subset is not None:
        # FIXME
        df.results = df.results.loc[:, 
            df.results.columns
            .str.removeprefix("train_")
            .str.removeprefix("test_")
            .isin(metric_subset)
        ]

    df["dataset_fold"] = (
        df[("dataset", "name")] + "_fold" + df[("cv", "fold")].astype(str)
    )
    df = df.dropna(axis=1, how="all")

    average_rank = (
        df.set_index(["dataset_fold", ("estimator", "name")])
        .results.groupby(level=0)  # by dataset and CV fold
        .rank(pct=True)
        .groupby(level=1)  # by estimator
        .mean()
        .T
    )
    average_rank.index.name = "score"
    average_rank.to_csv("average_rank.tsv", sep="\t")

    # Plot average ranks
    plt.figure(
        figsize=(
            average_rank.shape[1] * 2 / 2.54 + 1,
            average_rank.shape[0] / 2.54 + 1,
        )
    )
    sns.heatmap(average_rank, annot=True)
    plt.tight_layout()
    plt.savefig(
        "average_rank.png",
        bbox_inches="tight",
        transparent=True,
    )
    plt.close()

    average_rank_of_median = (
        df.set_index([("dataset", "name"), ("estimator", "name")])
        .results.groupby(level=(0, 1))  # by dataset and estimator
        .median()
        .groupby(level=0)  # by dataset
        .rank(pct=True)
        .groupby(level=1)  # by estimator
        .mean()
        .T
    )
    average_rank_of_median.index.name = "score"
    average_rank_of_median.to_csv("average_rank_of_median.tsv", sep="\t")

    # Plot average median ranks
    plt.figure(
        figsize=(
            average_rank_of_median.shape[1] * 2 / 2.54 + 1,
            average_rank_of_median.shape[0] / 2.54 + 1,
        )
    )
    sns.heatmap(average_rank_of_median, annot=True)
    plt.tight_layout()
    plt.savefig(
        "average_rank_of_median.png",
        bbox_inches="tight",
        transparent=True,
    )
    plt.close()

    # Calculate omnibus Friedman statistics
    allsets_friedman_statistics = friedman_melted(
        df,
        index=[("estimator", "name")],
        columns=["dataset_fold"],
        values=df.loc[:, ("results", slice(None))].columns,
    )
    # Set new level and level names to match the tables per training set that
    # we will generate later.
    allsets_friedman_statistics = pd.concat(
        {"all_datasets": allsets_friedman_statistics},
        names=["dataset", "score"],
    )

    # Calculate omnibus Friedman statistics per dataset
    friedman_statistics = df.groupby(("dataset", "name")).apply(
        friedman_melted,
        index=[("estimator", "name")],
        columns=[("cv", "fold")],
        values=df.loc[:, ("results", slice(None))].columns,
    )
    friedman_statistics.index = friedman_statistics.index.rename(["dataset", "score"])

    friedman_statistics = pd.concat([allsets_friedman_statistics, friedman_statistics])
    friedman_statistics["corrected_p"] = multipletests(
        friedman_statistics.pvalue.values,
        method="holm",
    )[1]

    friedman_statistics.to_csv("test_statistics.tsv", sep="\t")

    outdir = Path("post_hoc")
    outdir.mkdir(exist_ok=True)

    df2 = df.results.copy()
    df2["estimator"] = df[("estimator", "name")]
    df2["dataset_fold"] = df[("dataset_fold", "")]
    df2["dataset"] = df[("dataset", "name")]
    df2["fold"] = df[("cv", "fold")]

    print("Processing joined datasets statistics")
    for metric, pvalue_crosstable, mean_ranks in iter_posthoc_comparisons(
        df2,
        y_cols=df.results.columns,
        group_col="estimator",
        block_col="dataset",
        # block_col="dataset_fold",
        p_adjust="holm",
    ):
        make_visualizations(
            data=df2,
            metric=metric,
            pvalue_crosstable=pvalue_crosstable,
            mean_ranks=mean_ranks,
            group_col="estimator",
            outdir=outdir,
            prefix="all_datasets",
            omnibus_pvalue=(
                allsets_friedman_statistics.loc["all_datasets", metric].pvalue
            ),
        )

    latex_lines = []

    # Make visualizations of pairwise estimator comparisons.
    for dataset_name, dataset_group in df2.groupby("dataset"):
        print("Processing", dataset_name)
        for metric, pvalue_crosstable, mean_ranks in iter_posthoc_comparisons(
            dataset_group,
            y_cols=df.results.columns,
            group_col="estimator",
            block_col="fold",  # different from the above will all sets
            p_adjust="holm",
        ):
            omnibus_pvalue = friedman_statistics.loc[dataset_name, metric].pvalue
            make_visualizations(
                data=dataset_group,
                metric=metric,
                pvalue_crosstable=pvalue_crosstable,
                mean_ranks=mean_ranks,
                group_col="estimator",
                outdir=outdir,
                prefix=dataset_name,
                omnibus_pvalue=omnibus_pvalue,
            )
            latex_line = make_latex_table(
                data=dataset_group,
                block_col="fold",
                group_col="estimator",
                metric=metric,
                sig_matrix=pvalue_crosstable,
                positions=mean_ranks,
                round_digits=2,
                highlight_best=(omnibus_pvalue < 0.05),
                higher_is_better=not metric.endswith("time"),
            ).rename(metric)
            latex_line["dataset"] = dataset_name
            latex_lines.append(latex_line)

    latex_table = pd.concat(latex_lines, axis=1).T.pivot(columns="dataset").T
    latex_table.to_csv("latex_table.tsv", sep="\t")


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--estimators",
        nargs="+",
        help="Estimator names to include in the analysis",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        help="Dataset names to include in the analysis",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        help="Metrics to include in the analysis",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    main(
        estimator_subset=args.estimators,
        dataset_subset=args.datasets,
        metric_subset=args.metrics,
    )
