#!/usr/bin/env bash

set -e
BASEDIR="."

METRICS=(
    "test_roc_auc_micro"
    "test_average_precision_micro"
    "test_neg_hamming_loss_micro"
    "test_neg_label_ranking_loss"
    "test_f1_micro"
    "test_matthews_corrcoef_micro"
)
ESTIMATORS=(
    "RF+ET"
    "gcForest"
    "LCForest"
    "SLCForest"
    "FLAForest"
    "CaFE"
    "CaFE-OS"
    "CaFE-SLC"
    "CaFE-FLA"
)
DATASETS=(
    "VirusGO"
    "VirusPseAAC"
    "flags"
    "GrampositivePseAAC"
    "CHD_49"
    "emotions"
    "Gram_negative"
    "PlantGO"
    "birds"
    "scene"
    "yeast"
    "medical"
    "enron"
    # "Gram_positive"
    # "genbase"
    # "CAL500"  # No wang
    # "LLOG"  # No wang
)

DIRNAME=("drop30" "drop50" "drop70" "no_drop")
SUFFIX=("__30" "__50" "__70" "")

for i in "${!DIRNAME[@]}"
do
    printf "\n\n* Processing: %s\n\n\n" "${DIRNAME[$i]}"

    python $BASEDIR/scripts/plotting/make_statistical_comparisons.py \
        --results-table $BASEDIR/results/results_renamed_final.tsv \
        --outdir $BASEDIR/results/statistical_comparisons/${DIRNAME[$i]}/ \
        --estimators \
            $(
                for E in "${ESTIMATORS[@]}"
                do echo "$E${SUFFIX[$i]}"
                done
            ) \
        --metrics "${METRICS[@]}" \
        --datasets "${DATASETS[@]}"
done
