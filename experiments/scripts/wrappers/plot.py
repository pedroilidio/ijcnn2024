#!/usr/bin/env python3
"""Generate plots comparing the performance of the estimators under study."""

import argparse
from pathlib import Path
import sys

# HACK
sys.path.insert(0, str(Path(__file__).parents[2]))
from scripts.plotting.make_statistical_comparisons import make_statistical_comparisons


OUTDIR = "results/statistical_comparisons"
INPUT_RESULTS = "results/results_renamed_final.tsv"
METRICS = (
    "test_roc_auc_micro",
    "test_average_precision_micro",
    "test_neg_hamming_loss_micro",
    "test_neg_label_ranking_loss",
    "test_f1_micro",
    "test_matthews_corrcoef_micro",
)
ESTIMATORS = (
    "RF+ET",
    "gcForest",
    "LCForest",
    "SLCForest",
    "FLAForest",
    "CaFE",
    "CaFE-OS",
    "CaFE-SLC",
    "CaFE-FLA",
)
DATASETS = (
    "VirusGO",
    "VirusPseAAC",
    "flags",
    "GrampositivePseAAC",
    "CHD_49",
    "emotions",
    "Gram_negative",
    "PlantGO",
    "birds",
    "scene",
    "yeast",
    "medical",
    "enron",
    # "Gram_positive"
    # "genbase"
    # "CAL500"  # No wang
    # "LLOG"  # No wang
)


def plot(
    results_table_path=INPUT_RESULTS,
    outdir=OUTDIR,
    datasets=DATASETS,
    estimators=ESTIMATORS,
    metrics=METRICS,
):
    """Generate plots comparing the performance of the estimators under study.

    Parameters
    ----------
    results_table_path : str or Path
        Path to the results table.
    outdir : str or Path
        Output directory for the statistical comparisons.
    datasets : list of str
        Dataset names to include in the analysis.
    estimators : list of str
        Estimator names to include in the analysis.
    metrics : list of str
        Metrics to include in the analysis.
    """
    results_table_path = Path(results_table_path)
    outdir = Path(outdir)

    # Turn possible tuples into lists and remove quotes from the estimator
    # names (issues with MLFlow otherwise).
    estimators = [x.strip('"') for x in estimators]
    metrics = [x.strip('"') for x in metrics]
    datasets = [x.strip('"') for x in datasets]

    for dirname, suffix in (
        ("drop30", "__30"),
        ("drop50", "__50"),
        ("drop70", "__70"),
        ("no_drop", ""),
    ):
        print("\n* Processing:", dirname)

        make_statistical_comparisons(
            results_table_path=results_table_path,
            main_outdir=outdir / dirname,
            estimator_subset=[est + suffix for est in estimators],
            metric_subset=metrics,
            dataset_subset=datasets,
        )


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Process results and generate visualizations.",
    )
    parser.add_argument(
        "--outdir",
        "-o",
        default=OUTDIR,
        help="Output directory for the statistical comparisons.",
    )
    parser.add_argument(
        "--results-table",
        default=INPUT_RESULTS,
        help="Path to the results table.",
    )
    parser.add_argument(
        "--estimators",
        nargs="+",
        default=ESTIMATORS,
        help="Estimator names to include in the analysis",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DATASETS,
        help="Dataset names to include in the analysis",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=METRICS,
        help="Metrics to include in the analysis",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    plot(
        results_table_path=args.results_table,
        outdir=args.outdir,
        datasets=args.datasets,
        estimators=args.estimators,
        metrics=args.metrics,
    )


if __name__ == "__main__":
    main()
