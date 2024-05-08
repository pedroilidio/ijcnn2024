#!/usr/bin/env bash

set -e
BASEDIR=$(dirname "$0")

METRICS=(
    "test_roc_auc_micro"
    "test_average_precision_micro"
    "test_neg_hamming_loss_micro"
    "test_neg_label_ranking_loss"
)
ESTIMATORS=(
    "cascade_wang"
    "cascade_zhou"
    "cascade_proba"
    "cascade_tree_embedder"
    "cascade_tree_embedder_proba"
    "cascade_lc_proba"
    "cascade_lc_tree_embedder"
    "cascade_lc_tree_embedder_proba"
    "cascade_scar_proba"
    "cascade_scar_tree_embedder_proba"
)

ORIGINAL_LEVEL_SELECTION=(
    "__finished_imputation"
    "__max_ascending_oob"
    "__best_train"
    "__best_train"
    "__best_train"
    "__best_oob"
    "__best_oob"
    "__best_oob"
    "__best_oob"
    "__best_oob"
)

DIR_ORIGINAL_LEVEL_SELECTION="original_level_selection"
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

OUTER_DIRNAME=("no_drop" "drop30" "drop50" "drop70")  # "drop90")
OUTER_SUFFIX=("" "__30" "__50" "__70")  # "__90")

INNER_DIRNAME=(
    "best_level"
    "level_with_best_oob_score"
    "level_with_best_training_score"
)
INNER_SUFFIX=("__b" "__bo" "__bt")

for i in "${!OUTER_DIRNAME[@]}"; do
    printf "\n\n* Processing: %s\n\n\n" "${OUTER_DIRNAME[$i]}"

    python make_statistical_comparisons.py \
        --results-table $BASEDIR/results_renamed.tsv \
        --outdir $BASEDIR/statistical_comparisons/${OUTER_DIRNAME[$i]}/$DIR_ORIGINAL_LEVEL_SELECTION \
        --estimators \
            "cascade_tree_embedder${OUTER_SUFFIX[$i]}__level0" \
            $(
                for iE in "${!ESTIMATORS[@]}"
                do echo "${ESTIMATORS[$iE]}${OUTER_SUFFIX[$i]}${ORIGINAL_LEVEL_SELECTION[$iE]}"
                done
            ) \
        --metrics "${METRICS[@]}" \
        --datasets "${DATASETS[@]}"

    for j in "${!INNER_DIRNAME[@]}"; do
        python make_statistical_comparisons.py \
            --results-table $BASEDIR/results_renamed.tsv \
            --outdir $BASEDIR/statistical_comparisons/${OUTER_DIRNAME[$i]}/${INNER_DIRNAME[$j]} \
            --estimators \
                "cascade_tree_embedder${OUTER_SUFFIX[$i]}__level0" \
                $(
                    for E in "${ESTIMATORS[@]}"
                    do echo "${E}${OUTER_SUFFIX[$i]}${INNER_SUFFIX[$j]}"
                    done
                ) \
            --metrics "${METRICS[@]}" \
            --datasets "${DATASETS[@]}"

    done
done
