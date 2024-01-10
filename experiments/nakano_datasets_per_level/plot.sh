#!/usr/bin/env bash

set -e
BASEDIR=$(dirname "$0")

# Non-abbreviated version
# ESTIMATORS=(
#     "cascade_proba"
#     "cascade_tree_embedder"
#     "cascade_tree_embedder_proba"
#     "cascade_tree_embedder_chi2"
#     "cascade_lc_proba"
#     "cascade_lc_tree_embedder"
#     "cascade_lc_tree_embedder_proba"
#     "cascade_lc_tree_embedder_chi2"
#     "cascade_scar_proba"
#     "cascade_scar_tree_embedder"
#     "cascade_scar_tree_embedder_proba"
#     "cascade_scar_tree_embedder_chi2"
# )
# $(for E in "${ESTIMATORS[@]}" ; do echo "${E}__expected_best" ; done) \

ESTIMATORS=(
    "os"
    "te"
    "te_os"
    # "te_chi2"
    "lc_os"
    "lc_te"
    "lc_te_os"
    # "lc_te_chi2"
    # "scar_os"
    # "scar_te"
    # "scar_te_os"
    # "scar_te_chi2"
)

# OUTER_DIRNAME=("drop50" "drop70" "drop90" "no_drop")
# OUTER_SUFFIX=("__50" "__70" "__90" "")
OUTER_DIRNAME=("no_drop" "drop50" "drop70" "drop90")
OUTER_SUFFIX=("" "__50" "__70" "__90")

INNER_DIRNAME=(
    "best_level"
    "level_with_best_oob_score"
    "level_with_best_training_score"
)
INNER_SUFFIX=("__b" "__bo" "__bt")

for i in "${!OUTER_DIRNAME[@]}"; do
    for j in "${!INNER_DIRNAME[@]}"; do
        python make_statistical_comparisons.py \
            --results-table $BASEDIR/results_renamed.tsv \
            --outdir $BASEDIR/statistical_comparisons/${OUTER_DIRNAME[$i]}/${INNER_DIRNAME[$j]} \
            --estimators \
                "os${OUTER_SUFFIX[$i]}__0" \
                "te${OUTER_SUFFIX[$i]}__0" \
                $(
                    for E in "${ESTIMATORS[@]}"
                    do echo "${E}${OUTER_SUFFIX[$i]}${INNER_SUFFIX[$j]}"
                    done
                ) \
            --metrics test_roc_auc_micro test_average_precision_micro
    done
done
