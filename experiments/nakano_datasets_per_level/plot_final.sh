#!/usr/bin/env bash

set -e
BASEDIR=$(dirname "$0")

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
    "VirusGO" #
    "VirusPseAAC"
    "flags"
    # "Gram_positive"  #  FIXME: Should work!
    "GrampositivePseAAC"
    "CHD_49"
    "emotions"
    "Gram_negative"  #
    "PlantGO"  #
    "birds"
    "scene"  #
    # "genbase"  #
    "yeast"
    "medical"
    # "CAL500"  # No wang
    "enron"
    # "LLOG"  # No wang
)

# OUTER_DIRNAME=("no_drop" "drop30" "drop50" "drop70")
# OUTER_SUFFIX=("" "__30" "__50" "__70")
OUTER_DIRNAME=("drop30" "drop50" "drop70" "no_drop")
OUTER_SUFFIX=("__30" "__50" "__70" "")
INNER_DIRNAME="final"

for i in "${!OUTER_DIRNAME[@]}"
do
    printf "\n\n* Processing: %s\n\n\n" "${OUTER_DIRNAME[$i]}"

    python $BASEDIR/make_statistical_comparisons.py \
        --results-table $BASEDIR/results_renamed_final.tsv \
        --outdir $BASEDIR/statistical_comparisons/${OUTER_DIRNAME[$i]}/$INNER_DIRNAME \
        --estimators \
            $(
                for E in "${ESTIMATORS[@]}"
                do echo "$E${OUTER_SUFFIX[$i]}"
                done
            ) \
        --metrics "${METRICS[@]}" \
        --datasets "${DATASETS[@]}"
done
