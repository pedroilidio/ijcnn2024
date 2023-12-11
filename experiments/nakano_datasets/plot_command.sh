python make_statistical_comparisons.py \
    --results-table nakano_datasets/results.tsv \
    --outdir nakano_datasets/statistical_comparisons/best \
    --estimators proba__10 embedding__10 embedding_proba__10 proba__6 proba__2 embedding__5 embedding_proba__5 embedding_chi2__10 embedding_chi2__5 final_estimator \
&& python make_statistical_comparisons.py \
    --results-table nakano_datasets/results.tsv \
    --outdir nakano_datasets/statistical_comparisons/embedding_proba \
    --estimators embedding_proba__1 embedding_proba__2 embedding_proba__3 embedding_proba__4 embedding_proba__5 embedding_proba__6 embedding_proba__7 embedding_proba__8 embedding_proba__9 embedding_proba__10 final_estimator \
&& python make_statistical_comparisons.py \
    --results-table nakano_datasets/results.tsv \
    --outdir nakano_datasets/statistical_comparisons/proba \
    --estimators proba__1 proba__2 proba__3 proba__4 proba__5 proba__6 proba__7 proba__8 proba__9 proba__10 \
&& python make_statistical_comparisons.py final_estimator \
    --results-table nakano_datasets/results.tsv \
    --outdir nakano_datasets/statistical_comparisons/embedding_chi2 \
    --estimators embedding_chi2__1 embedding_chi2__2 embedding_chi2__3 embedding_chi2__4 embedding_chi2__5 embedding_chi2__6 embedding_chi2__7 embedding_chi2__8 embedding_chi2__9 embedding_chi2__10 final_estimator \
&& python make_statistical_comparisons.py \
    --results-table nakano_datasets/results.tsv \
    --outdir nakano_datasets/statistical_comparisons/embedding \
    --estimators embedding__1 embedding__2 embedding__3 embedding__4 embedding__5 embedding__6 embedding__7 embedding__8 embedding__9 embedding__10 final_estimator \
