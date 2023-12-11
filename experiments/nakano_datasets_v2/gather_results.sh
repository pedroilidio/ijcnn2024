#!/bin/bash
set -e

python make_runs_table.py \
    --out nakano_datasets_v2/results.tsv \
    --runs nakano_datasets_v2/runs

python nakano_datasets_v2/rename_estimators.py nakano_datasets_v2/results.tsv
