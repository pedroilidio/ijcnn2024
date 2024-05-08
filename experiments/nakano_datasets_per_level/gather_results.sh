#!/bin/bash
set -e

BASEDIR=$(dirname "$0")

python make_runs_table.py \
    --out $BASEDIR/results.tsv \
    --runs $BASEDIR/runs

python $BASEDIR/rename_estimators.py $BASEDIR/results.tsv
python $BASEDIR/final_rename.py $BASEDIR/results_renamed.tsv -o $BASEDIR/results_renamed_final.tsv
