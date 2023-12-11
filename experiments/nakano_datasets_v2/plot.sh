#!/bin/bash
set -e

python make_statistical_comparisons.py \
    --results-table nakano_datasets_v2/results_renamed.tsv \
    --outdir nakano_datasets_v2/statistical_comparisons/no_drop \
    --estimators \
        proba \
        embedding \
        embedding_proba \
        embedding_chi2 \
        proba_weak \
        embedding_weak \
        embedding_weak_chi2

python make_statistical_comparisons.py \
    --results-table nakano_datasets_v2/results_renamed.tsv \
    --outdir nakano_datasets_v2/statistical_comparisons/drop50 \
    --estimators \
        proba__50 \
        embedding__50 \
        embedding_proba__50 \
        embedding_chi2__50 \
        proba_weak__50 \
        embedding_weak__50 \
        embedding_weak_chi2__50

python make_statistical_comparisons.py \
    --results-table nakano_datasets_v2/results_renamed.tsv \
    --outdir nakano_datasets_v2/statistical_comparisons/drop70 \
    --estimators \
        proba__70 \
        embedding__70 \
        embedding_proba__70 \
        embedding_chi2__70 \
        proba_weak__70 \
        embedding_weak__70 \
        embedding_weak_chi2__70

python make_statistical_comparisons.py \
    --results-table nakano_datasets_v2/results_renamed.tsv \
    --outdir nakano_datasets_v2/statistical_comparisons/drop90 \
    --estimators \
        proba__90 \
        embedding__90 \
        embedding_proba__90 \
        embedding_chi2__90 \
        proba_weak__90 \
        embedding_weak__90 \
        embedding_weak_chi2__90
