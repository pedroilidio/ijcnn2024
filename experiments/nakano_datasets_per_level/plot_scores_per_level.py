from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_scores_per_level(path_results_table: Path, outdir: Path):
    data = pd.read_table(path_results_table).set_index(
        ["estimator.name", "estimator.level", "cv.fold", "dataset.name", "wrapper.name"]
    )
    data = data.loc[:, data.columns.str.startswith("results.")]
    data = data.dropna(axis=1, how="all")

    data.columns = data.columns.str.extract(
        r"results\.(\w+?)_+((?:\w(?!_oob))+\w)_?(oob)?",
    )
    # Captured groups examples:
    #     results.(test)_(average_precision_macro)
    #     results.(test)__(average_precision_macro)_(oob)
    #     results.(train)_(average_precision_micro)
    #     results.(internal)__(average_precision_micro)
    #     results.(test)_(f1_micro)_(oob)

    data.columns = pd.MultiIndex.from_tuples(
        data.columns, names=["set", "metric", "oob"],
    )
    data = data.reorder_levels(["metric", "set", "oob"], axis=1)

    for metric, metric_group in data.T.groupby(level="metric", as_index=False):
        for est, est_group in metric_group.T.groupby(
            level="estimator.name", as_index=False,
        ):
            if "test" not in est_group.columns.get_level_values("set"):
                print(
                    f"Skipping {metric} for {est} because it has no results for test"
                    " set."
                )
                continue

            # TODO: shouldn't as_index have dropped them already?
            est_group = est_group.droplevel("metric", axis=1)
            est_group = est_group.droplevel("estimator.name", axis=0)

            fold_means = est_group.groupby(level=["dataset.name", "estimator.level"]).mean()
            norm_fold_means = (
                fold_means
                .groupby(level="dataset.name", group_keys=False)
                # .apply(lambda g: g.dropna().div(g.mean()))
                .apply(lambda g: g.dropna().div(g.test.mean().mean()))
            )

            # Average over datasets
            harry_plotter = (
                norm_fold_means
                .groupby(level="estimator.level")
                .mean()
                .reset_index()
            )

            plt.figure(1)
            (
                fold_means
                .groupby(level="estimator.level")
                .mean()
                .reset_index()
                .plot(
                    ax=plt.gca(),
                    x=("estimator.level", ""),
                    y=("test", np.nan),
                    label=est,
                )
            )

            plt.figure(2)
            ax_relative = None
            ax_relative = harry_plotter.plot(
                ax=ax_relative,
                x=("estimator.level", ""),
                y=("test", np.nan),
                label=est + "_test",
                linestyle="-",
            )
            ax_relative = harry_plotter.plot(
                ax=ax_relative,
                x=("estimator.level", ""),
                y=("internal", np.nan),
                label=est + "_train",
                linestyle=":",
            )
            ax_relative = harry_plotter.plot(
                ax=ax_relative,
                x=("estimator.level", ""),
                y=("internal", "oob"),
                label=est + "_oob",
                linestyle="--",
            )

            title = f"{metric}_{est}"
            plt.title(title)

            outpath = outdir / f"relative_metrics/{title}.png"
            outpath.parent.mkdir(exist_ok=True, parents=True)
            plt.savefig(outpath)
            print(f"Saved plot to '{outpath}'.")
            plt.close()

        plt.figure(1)
        outpath = outdir / f"estimator_comparison/{metric}.png"
        outpath.parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(outpath)
        print(f"Saved plot to '{outpath}'.")
        plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('results', type=Path, help='Path to results table')
    parser.add_argument(
        '--outdir', '-o', type=Path, default=Path.cwd(),
        help='Directory to save plots to',
    )
    args = parser.parse_args()

    plot_scores_per_level(
        path_results_table=args.results,
        outdir=args.outdir,
    )


if __name__ == "__main__":
    main()
