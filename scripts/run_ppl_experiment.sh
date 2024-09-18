#!/bin/shd
eval "$(conda shell.bash hook)"
conda activate unlearning

CUDA_VISIBLE_DEVICES=6,7 python -m src.enhanced_gcg.measure_ppl --task wmdp-bio --unlearned_model cais/Zephyr_RMU --original_model HuggingFaceH4/zephyr-7b-beta