#!/env/bin bash
# set -e

printf -v FILES "%s " \
    statistical_comparisons/no_drop/final/all_datasets/boxplots/*.pdf \
    statistical_comparisons/drop30/final/all_datasets/boxplots/*.pdf \
    statistical_comparisons/drop50/final/all_datasets/boxplots/*.pdf \
    statistical_comparisons/drop70/final/all_datasets/boxplots/*.pdf \
    \

    # level_comparison/cascade_lc_tree_embedder_proba__50/average_precision_micro.png \
    # level_comparison/cascade_lc_tree_embedder_proba__50/roc_auc_micro.png \
    # level_comparison/cascade_scar_tree_embedder_proba__50/average_precision_micro.png \
    # level_comparison/cascade_scar_tree_embedder_proba__50/roc_auc_micro.png \
    # level_comparison/cascade_tree_embedder_proba__50/average_precision_micro.png \
    # level_comparison/cascade_tree_embedder_proba__50/roc_auc_micro.png \

    # \
    # level_comparison/cascade_tree_embedder/average_precision_micro.png \
    # level_comparison/cascade_tree_embedder/roc_auc_micro.png \
    # level_comparison/cascade_tree_embedder__30/average_precision_micro.png \
    # level_comparison/cascade_tree_embedder__30/roc_auc_micro.png \
    # level_comparison/cascade_tree_embedder__50/average_precision_micro.png \
    # level_comparison/cascade_tree_embedder__50/roc_auc_micro.png \
    # level_comparison/cascade_tree_embedder__70/average_precision_micro.png \
    # level_comparison/cascade_tree_embedder__70/roc_auc_micro.png \
    # level_comparison/cascade_tree_embedder__90/average_precision_micro.png \
    # level_comparison/cascade_tree_embedder__90/roc_auc_micro.png \

    # \
    # imputation_plots/cascade_lc_tree_embedder_proba__30.png \
    # imputation_plots/cascade_lc_tree_embedder_proba__50.png \
    # imputation_plots/cascade_lc_tree_embedder_proba__70.png \
    # imputation_plots/cascade_lc_tree_embedder_proba__90.png \
    # imputation_plots/cascade_scar_tree_embedder_proba__50.png \
    # imputation_plots/cascade_scar_tree_embedder_proba__30.png \
    # imputation_plots/cascade_scar_tree_embedder_proba__70.png \
    # imputation_plots/cascade_scar_tree_embedder_proba__90.png \


for FILE in ${FILES[@]}
    do ln -sr $FILE  selected_figures/$(echo $FILE | sed "s/\.\.\///g" | sed "s/\//__/g")
done
