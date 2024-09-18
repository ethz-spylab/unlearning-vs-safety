#!/bin/shd
eval "$(conda shell.bash hook)"
conda activate flrt

subset=wmdp-cyber
prompt_ids=0,2,3,4,5
model_name=J4Q8/zephyr-npo-cyber


CUDA_VISIBLE_DEVICES=4,5 python -m src.enhanced_gcg.flrt_repo.demo --model_name_or_path $model_name --optimize_prompts $prompt_ids  --wmdp_subset $subset --use_static_representations --dont_clamp_loss --attack_layers 20 --use_init rmu-best &
sleep 2
CUDA_VISIBLE_DEVICES=6,7 python -m src.enhanced_gcg.flrt_repo.demo --model_name_or_path $model_name --optimize_prompts $prompt_ids  --wmdp_subset $subset --use_static_representations --dont_clamp_loss --attack_layers 20 --use_init rmu-best2 &