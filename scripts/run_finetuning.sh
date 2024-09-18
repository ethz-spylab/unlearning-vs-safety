#!/bin/shd
eval "$(conda shell.bash hook)"
conda activate finetuning

CUDA_VISIBLE_DEVICES=6 python -m src.finetuning.finetune \
            --model cais/Zephyr_RMU \
            --dataset wikitext \
            --n_samples 5000 &

CUDA_VISIBLE_DEVICES=7 python -m src.finetuning.finetune \
    --model cais/Zephyr_RMU \
    --dataset wikitext \
    --n_samples 10000