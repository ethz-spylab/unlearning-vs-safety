#!/bin/shd
eval "$(conda shell.bash hook)"
conda activate unlearning

CUDA_VISIBLE_DEVICES=0,1 python -m src.set_difference_pruning.generate_datasets \
    --unlearned_model cais/Zephyr_RMU --original_model HuggingFaceH4/zephyr-7b-beta --task wmdp-bio

CUDA_VISIBLE_DEVICES=0,1 python -m src.set_difference_pruning.generate_datasets \
    --unlearned_model J4Q8/zephyr-dpo-bio --unlearned_tokenizer HuggingFaceH4/zephyr-7b-beta --original_model HuggingFaceH4/zephyr-7b-beta --task wmdp-bio

CUDA_VISIBLE_DEVICES=0,1 python -m src.set_difference_pruning.generate_datasets \
    --unlearned_model J4Q8/zephyr-npo-bio --unlearned_tokenizer HuggingFaceH4/zephyr-7b-beta --original_model HuggingFaceH4/zephyr-7b-beta --task wmdp-bio

CUDA_VISIBLE_DEVICES=0,1 python -m src.set_difference_pruning.generate_datasets \
    --unlearned_model cais/Zephyr_RMU  --original_model HuggingFaceH4/zephyr-7b-beta --task wmdp-cyber

CUDA_VISIBLE_DEVICES=0,1 python -m src.set_difference_pruning.generate_datasets \
    --unlearned_model J4Q8/zephyr-dpo-cyber --unlearned_tokenizer HuggingFaceH4/zephyr-7b-beta --original_model HuggingFaceH4/zephyr-7b-beta --task wmdp-cyber

CUDA_VISIBLE_DEVICES=0,1 python -m src.set_difference_pruning.generate_datasets \
    --unlearned_model J4Q8/zephyr-npo-cyber --unlearned_tokenizer HuggingFaceH4/zephyr-7b-beta --original_model HuggingFaceH4/zephyr-7b-beta --task wmdp-cyber