#!/bin/shd
eval "$(conda shell.bash hook)"
conda activate unlearning

CUDA_VISIBLE_DEVICES=5 python -m src.perturbations.measure_ppl --task wmdp-bio --unlearned_model cais/Zephyr_RMU --original_model HuggingFaceH4/zephyr-7b-beta --perturbations_dir results/perturbations/informed/Zephyr_RMU/wmdp-bio