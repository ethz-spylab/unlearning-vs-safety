#!/bin/shd
eval "$(conda shell.bash hook)"
conda activate unlearning

CUDA_VISIBLE_DEVICES=4,5 python -m src.perturbations.evaluate_perturbations \
    --model_name_or_path J4Q8/zephyr-npo-bio --tokenizer_name_or_path HuggingFaceH4/zephyr-7b-beta --perturbations_path results/perturbations/informed/Zephyr_RMU/wmdp-bio

CUDA_VISIBLE_DEVICES=4,5 python -m src.perturbations.evaluate_perturbations \
    --model_name_or_path J4Q8/zephyr-dpo-bio --tokenizer_name_or_path HuggingFaceH4/zephyr-7b-beta --perturbations_path results/perturbations/informed/Zephyr_RMU/wmdp-bio