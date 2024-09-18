#!/bin/shd
eval "$(conda shell.bash hook)"
conda activate unlearning

CUDA_VISIBLE_DEVICES=6,7 python -m src.enhanced_gcg.test_adv_prefixes \
    --model_name_or_path cais/Zephyr_RMU \
    --original_model_name_or_path HuggingFaceH4/zephyr-7b-beta \
    --wmdp_subset wmdp-cyber