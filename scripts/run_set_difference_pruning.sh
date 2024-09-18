#!/bin/shd
# This script runs the alignment-attribution code
eval "$(conda shell.bash hook)"
conda activate unlearning

method="wandg"

ratio_vals=($(seq 0 44)) #45 total configs

for config_id in ${ratio_vals[@]}
do 
    CUDA_VISIBLE_DEVICES=1 python -m src.set_difference_pruning.alignment-attribution-code.run_gridsearch --model cais/Zephyr_RMU --prune_method $method -i $config_id --prune_data wmdp_bio-forget-corpus_abstract --util_data wikitext &
    sleep 2
    CUDA_VISIBLE_DEVICES=2 python -m src.set_difference_pruning.alignment-attribution-code.run_gridsearch --model J4Q8/zephyr-dpo-bio --prune_method $method -i $config_id --prune_data generations-zephyr-dpo-bio-wmdp-bio-prune --util_data mmlu_abcd &
    sleep 2
    CUDA_VISIBLE_DEVICES=3 python -m src.set_difference_pruning.alignment-attribution-code.run_gridsearch --model J4Q8/zephyr-npo-bio --prune_method $method -i $config_id --prune_data generations-zephyr-npo-bio-wmdp-bio-prune --util_data mmlu_abcd
done

echo "Process done"
# conda deactivate