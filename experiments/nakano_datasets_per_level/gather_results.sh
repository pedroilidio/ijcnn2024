#!/bin/bash
set -e

# This script's parent directory
BASEDIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

python make_runs_table.py \
    --out $BASEDIR/results.tsv \
    --runs $BASEDIR/runs

# $BASEDIR/runs \
python $BASEDIR/parse_level_scores.py \
    nakano_datasets_v2/runs \
    --out $BASEDIR/level_scores.tsv

python $BASEDIR/rename_estimators.py $BASEDIR/results.tsv $BASEDIR/level_scores.tsv

