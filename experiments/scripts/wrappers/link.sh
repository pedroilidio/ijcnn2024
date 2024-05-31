#!/env/bin bash
# set -e

printf -v FILES "%s " \
    statistical_comparisons/no_drop/final/all_datasets/boxplots/*.pdf \
    statistical_comparisons/drop30/final/all_datasets/boxplots/*.pdf \
    statistical_comparisons/drop50/final/all_datasets/boxplots/*.pdf \
    statistical_comparisons/drop70/final/all_datasets/boxplots/*.pdf \
    \

    # level_comparison/cafe_slc__50/average_precision_micro.png \
    # level_comparison/cafe_slc__50/roc_auc_micro.png \
    # level_comparison/cafe_fla__50/average_precision_micro.png \
    # level_comparison/cafe_fla__50/roc_auc_micro.png \
    # level_comparison/cafe_os__50/average_precision_micro.png \
    # level_comparison/cafe_os__50/roc_auc_micro.png \

    # \
    # level_comparison/cafe/average_precision_micro.png \
    # level_comparison/cafe/roc_auc_micro.png \
    # level_comparison/cafe__30/average_precision_micro.png \
    # level_comparison/cafe__30/roc_auc_micro.png \
    # level_comparison/cafe__50/average_precision_micro.png \
    # level_comparison/cafe__50/roc_auc_micro.png \
    # level_comparison/cafe__70/average_precision_micro.png \
    # level_comparison/cafe__70/roc_auc_micro.png \
    # level_comparison/cafe__90/average_precision_micro.png \
    # level_comparison/cafe__90/roc_auc_micro.png \

    # \
    # imputation_plots/cafe_slc__30.png \
    # imputation_plots/cafe_slc__50.png \
    # imputation_plots/cafe_slc__70.png \
    # imputation_plots/cafe_slc__90.png \
    # imputation_plots/cafe_fla__50.png \
    # imputation_plots/cafe_fla__30.png \
    # imputation_plots/cafe_fla__70.png \
    # imputation_plots/cafe_fla__90.png \


for FILE in ${FILES[@]}
    do ln -sr $FILE  selected_figures/$(echo $FILE | sed "s/\.\.\///g" | sed "s/\//__/g")
done
