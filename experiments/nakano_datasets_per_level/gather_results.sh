#!/bin/bash
set -e

BASEDIR=$(dirname "$0")

python make_runs_table.py \
    --out $BASEDIR/results.tsv \
    --runs $BASEDIR/runs

python $BASEDIR/rename_estimators.py $BASEDIR/results.tsv
