#!/bin/shd
eval "$(conda shell.bash hook)"
conda activate unlearning

CUDA_VISIBLE_DEVICES=2 python -m src.benchmarking.benchmark --model_name_or_path HuggingFaceH4/zephyr-7b-beta
CUDA_VISIBLE_DEVICES=2 python -m src.benchmarking.benchmark --model_name_or_path cais/Zephyr_RMU