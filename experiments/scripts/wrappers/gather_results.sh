#!/bin/bash
set -e

BASEDIR="."
mkdir $BASEDIR/results -p

python $BASEDIR/scripts/results_processing/make_runs_table.py \
    --out $BASEDIR/results/results.tsv \
    --runs $BASEDIR/paper_runs

python $BASEDIR/scripts/results_processing/rename_estimators.py \
    $BASEDIR/results/results.tsv

python $BASEDIR/scripts/results_processing/final_rename.py \
    $BASEDIR/results/results_renamed.tsv \
    -o $BASEDIR/results/results_renamed_final.tsv
