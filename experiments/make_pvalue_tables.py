import argparse
import itertools
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
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

    percentile_ranks = data.groupby(block_col)[metric].rank(pct=True)
    is_victory = percentile_ranks == 1

    percentile_ranks_stats = (
        percentile_ranks.groupby(data[group_col]).agg(["mean", "std"]).T
    )
    is_victory_stats = (
        is_victory.groupby(data[group_col]).agg(["mean", "std"]).T
    )  # How many times was this estimator the best?

    text_table = {}
    for row, row_name in (
        (table, metric),
        (percentile_ranks_stats, metric + "_rank"),
        (is_victory_stats, metric + "_victories"),
    ):
        text_table[row_name] = (
            row.round(round_digits).astype(str).apply(lambda r: "{} ({})".format(*r))
        )
    text_table = pd.concat(text_table).reorder_levels([1, 0]).sort_index()

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
    if len(top_crossbar_set) < len(adj_matrix):
        ### latex
        # text_table.loc[(top_crossbar_set, slice(None))] = text_table[
        #     top_crossbar_set
        # ].apply(lambda s: f"\\textbf{{{s}}}")

        ### html
        text_table.loc[(top_crossbar_set, slice(None))] = text_table[
            top_crossbar_set
        ].apply(lambda s: f"<b>{s}</b>")

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

    estimators_per_fold = data.groupby(block_col)[group_col].count()
    folds_to_drop = estimators_per_fold[estimators_per_fold < estimators_per_fold.max()].index
    if not folds_to_drop.empty:  # FIXME: explain
        warnings.warn(
            "The following groups have missing blocks and will be removed"
            f" from the comparison analysis:\n{folds_to_drop}"
        )
        data = data[~data[block_col].isin(folds_to_drop)]

    # missing_blocks = (
    #     data.groupby(group_col)[block_col].unique().apply(lambda x: all_blocks - set(x))
    # )
    # missing_blocks = missing_blocks.loc[missing_blocks.apply(len) != 0]

    # if not missing_blocks.empty:
    #     warnings.warn(
    #         "The following groups have missing blocks and will be removed"
    #         f" from the comparison analysis:\n{missing_blocks}"
    #     )
    #     data = data[~data[group_col].isin(missing_blocks.index)]

    groups = data[group_col].unique()
    n_groups = len(groups)

    for metric in y_cols:
        print("- Processing metric:", metric)
        if n_groups <= 1:
            warnings.warn(
                f"Skipping {metric} because there are not enough groups "
                f"({n_groups}) to perform a test statistic."
            )
            continue
        pvalue_crosstable = sp.posthoc_nemenyi_friedman(
        # pvalue_crosstable = sp.posthoc_conover_friedman(
            data,
            melted=True,
            y_col=metric,
            group_col=group_col,
            block_col=block_col,
            # p_adjust=p_adjust,
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
    plt.title(f"{metric}\np = {omnibus_pvalue:.2e}", wrap=True)
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
        # transparent=True,
        # bbox_inches='tight',  # Allow for title wrapping
    )
    plt.savefig(
        outdir / (f"{prefix}__{metric}.pdf"),
        transparent=True,
        bbox_inches="tight",
    )
    plt.close()

    plt.figure(figsize=(6, 0.5 * n_groups / 2.54 + 1))
    plot_critical_difference_diagram(
        mean_ranks,
        pvalue_crosstable,
        crossbar_props={"marker": "."},
    )
    plt.title(f"{metric}\np = {omnibus_pvalue:.2e}", wrap=True)
    plt.tight_layout()
    plt.savefig(
        outdir / (f"{prefix}__cdd__{metric}.png"),
    )
    plt.savefig(
        outdir / (f"{prefix}__cdd__{metric}.pdf"),
        transparent=True,
        bbox_inches="tight",
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
    plt.title(f"{metric}\np = {omnibus_pvalue:.2e}", wrap=True)
    plt.tight_layout()
    plt.savefig(
        outdir / (f"{prefix}__boxplot__{metric}.png"),
    )
    plt.savefig(
        outdir / (f"{prefix}__boxplot__{metric}.pdf"),
        transparent=True,
        bbox_inches="tight",
    )
    plt.close()


def friedman_melted(data, *, index, columns, values):
    # Expand ("unmelt") to 1 fold per column on level 2, metrics on level 1
    pivot = data.pivot(index=index, columns=columns, values=values)

    if pivot.shape[0] < 3:
        warnings.warn(
            f"Dataset {data.name} has only {pivot.shape[0]} estimators, "
            "which is not enough for a Friedman test."
        )
        result = pd.DataFrame(
            index=np.unique(pivot.columns.get_level_values(0)),
            columns=["statistic", "pvalue"],
            dtype=float,
        )
        result["statistic"] = np.nan
        result["pvalue"] = 1.0
        return result

    # Apply Friedman's test for each result metric
    result = (
        pivot.groupby(level=0, axis=1)
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


def plot_everything(
    estimator_subset=None,
    dataset_subset=None,
    metric_subset=None,
):
    df = pd.read_table("results.tsv", header=[0, 1])

    if estimator_subset is not None:
        df = df[df[("estimator", "name")].isin(estimator_subset)]
    if dataset_subset is not None:
        df = df[df[("dataset", "name")].isin(dataset_subset)]
    if metric_subset is not None:
        df = df.loc[:, [c[0] != "results" or c[1] in metric_subset for c in df.columns]]

    df2 = df.results.copy().dropna(axis=1, how="all")
    df2["estimator"] = df[("estimator", "name")]
    df2["dataset"] = df[("dataset", "name")]
    df2["fold"] = df[("cv", "fold")]
    # FIXME: SPOILS THE ALL_DATASETS DATAFRAME
    # HACK: Add a suffix to the dataset name to distinguish between different
    #       positive dropout ratios. The dropout is originally in the estimator
    #       name, e.g. "random_forest__drop20".
    # df2.loc[:, "dataset"] = (
    #     df2.dataset + "__" + df2.estimator.str.split("__").str[1].fillna("drop0")
    # )
    # df2.loc[:, "estimator"] = df2.estimator.str.split("__").str[0]

    # Not in one line to enable str_accessor[1] even if all splits generate a
    # single element
    str_accessor = df2.estimator.str.split("__").str
    df2["estimator"] = str_accessor[0]
    df2["dropout"] = str_accessor[1].fillna("drop0")  

    df2 = df2.loc[df2.dropout == "drop0"]  # FIXME: consider dropouts

    max_estimators_per_dataset = df2.groupby("dataset").estimator.nunique().max()
    max_folds_per_estimator = df2.groupby(["dataset", "estimator"]).fold.nunique().max()

    allsets_data = (
        df2
        # Consider only datasets with all the estimators
        .groupby("dataset")
        .filter(lambda x: x.estimator.nunique() == max_estimators_per_dataset)
    )
    discarded_datasets = set(df2.dataset.unique()) - set(allsets_data.dataset.unique())

    if discarded_datasets:
        print(
            "The following datasets were not present for all estimators and"
            " will not be considered for rankings across all datasets:"
            f" {discarded_datasets}"
        )

    allsets_data = (
        allsets_data
        # Consider only estimators with all the CV folds
        .groupby(["dataset", "estimator"])
        .filter(lambda x: x.fold.nunique() == max_folds_per_estimator)
    )
    discarded_runs = (
        set(df2[["dataset", "estimator"]])
        - set(allsets_data[["dataset", "estimator"]])
    )

    if discarded_runs:
        print(
            "The following runs were not present for all CV folds and"
            " will not be considered for rankings across all datasets:"
            f" {discarded_runs}"
        )

    allsets_data = (
        allsets_data
        .set_index(["dataset", "fold", "estimator"])  # Keep columns
        .groupby(level=[0, 1])  # groupby(["dataset", "fold"])
        .rank(pct=True)  # Rank estimators per fold
        .groupby(level=[0, 2])  # groupby(["dataset", "estimator"])
        .mean()  # Average ranks across folds for each estimator
        .rename_axis(index=["fold", "estimator"])  # 'dataset' -> 'fold'
        .reset_index()
        .assign(dataset="all_datasets")
    )

    df2 = pd.concat([allsets_data, df2], ignore_index=True, sort=False)

    # Calculate omnibus Friedman statistics per dataset
    friedman_statistics = df2.groupby("dataset").apply(
        friedman_melted,
        columns="fold",
        index="estimator",
        values=df.results.columns,
    )
    friedman_statistics["corrected_p"] = multipletests(
        friedman_statistics.pvalue.values,
        method="holm",
    )[1]

    friedman_statistics.to_csv("test_statistics.tsv", sep="\t")

    outdir = Path("post_hoc")
    outdir.mkdir(exist_ok=True)

    df2 = df2.dropna(axis=1, how="all")  # FIXME: something is bringing nans back
    metric_names = df.results.columns.intersection(df2.columns)

    latex_lines = []

    # Make visualizations of pairwise estimator comparisons.
    for dataset_name, dataset_group in df2.groupby("dataset"):
        print("Processing", dataset_name)
        for metric, pvalue_crosstable, mean_ranks in iter_posthoc_comparisons(
            dataset_group,
            y_cols=metric_names,
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
            )

            # latex_lines[(dataset_name, metric)] = latex_line
            latex_line = pd.concat({dataset_name: latex_line}, names=["dataset"])
            latex_lines.append(latex_line)

    latex_table = (
        pd.concat(latex_lines)
        .rename_axis(["dataset", "estimator", "score"])
        .unstack(level=2)
    )  # Set metrics as columns
    latex_table.to_csv("latex_table.tsv", sep="\t")
    latex_table.to_html("latex_table.html", escape=False)


def main():
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
    args = parser.parse_args()

    plot_everything(
        estimator_subset=args.estimators,
        dataset_subset=args.datasets,
        metric_subset=args.metrics,
    )


if __name__ == "__main__":
    main()
