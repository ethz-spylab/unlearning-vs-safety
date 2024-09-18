#!/bin/shd
eval "$(conda shell.bash hook)"
conda activate unlearning

CUDA_VISIBLE_DEVICES=0 python -m src.logit_lens.logit_lens --task wmdp-bio --model_name_or_path HuggingFaceH4/zephyr-7b-beta 
CUDA_VISIBLE_DEVICES=0 python -m src.logit_lens.logit_lens --task wmdp-bio --model_name_or_path cais/Zephyr_RMU
CUDA_VISIBLE_DEVICES=0 python -m src.logit_lens.logit_lens --task wmdp-bio --model_name_or_path J4Q8/zephyr-dpo-bio --tokenizer_name_or_path HuggingFaceH4/zephyr-7b-beta 
CUDA_VISIBLE_DEVICES=0 python -m src.logit_lens.logit_lens --task wmdp-bio --model_name_or_path J4Q8/zephyr-npo-bio --tokenizer_name_or_path HuggingFaceH4/zephyr-7b-beta 

python -m src.logit_lens.plot_logit_lens --task wmdp-bio

CUDA_VISIBLE_DEVICES=0 python -m src.logit_lens.logit_lens --task wmdp-cyber --model_name_or_path HuggingFaceH4/zephyr-7b-beta 
CUDA_VISIBLE_DEVICES=0 python -m src.logit_lens.logit_lens --task wmdp-cyber --model_name_or_path cais/Zephyr_RMU
CUDA_VISIBLE_DEVICES=0 python -m src.logit_lens.logit_lens --task wmdp-cyber --model_name_or_path J4Q8/zephyr-dpo-cyber --tokenizer_name_or_path HuggingFaceH4/zephyr-7b-beta 
CUDA_VISIBLE_DEVICES=0 python -m src.logit_lens.logit_lens --task wmdp-cyber --model_name_or_path J4Q8/zephyr-npo-cyber --tokenizer_name_or_path HuggingFaceH4/zephyr-7b-beta 

python -m src.logit_lens.plot_logit_lens --task wmdp-cyber

CUDA_VISIBLE_DEVICES=0 python -m src.logit_lens.logit_lens --task wmdp-bio --model_name_or_path HuggingFaceH4/zephyr-7b-beta --apply_chat_template
CUDA_VISIBLE_DEVICES=0 python -m src.logit_lens.logit_lens --task wmdp-bio --model_name_or_path cais/Zephyr_RMU --apply_chat_template
CUDA_VISIBLE_DEVICES=0 python -m src.logit_lens.logit_lens --task wmdp-bio --model_name_or_path J4Q8/zephyr-dpo-bio --tokenizer_name_or_path HuggingFaceH4/zephyr-7b-beta --apply_chat_template
CUDA_VISIBLE_DEVICES=0 python -m src.logit_lens.logit_lens --task wmdp-bio --model_name_or_path J4Q8/zephyr-npo-bio --tokenizer_name_or_path HuggingFaceH4/zephyr-7b-beta --apply_chat_template

python -m src.logit_lens.plot_logit_lens --task wmdp-bio --apply_chat_template

CUDA_VISIBLE_DEVICES=0 python -m src.logit_lens.logit_lens --task wmdp-cyber --model_name_or_path HuggingFaceH4/zephyr-7b-beta --apply_chat_template
CUDA_VISIBLE_DEVICES=0 python -m src.logit_lens.logit_lens --task wmdp-cyber --model_name_or_path cais/Zephyr_RMU --apply_chat_template
CUDA_VISIBLE_DEVICES=0 python -m src.logit_lens.logit_lens --task wmdp-cyber --model_name_or_path J4Q8/zephyr-dpo-cyber --tokenizer_name_or_path HuggingFaceH4/zephyr-7b-beta --apply_chat_template
CUDA_VISIBLE_DEVICES=0 python -m src.logit_lens.logit_lens --task wmdp-cyber --model_name_or_path J4Q8/zephyr-npo-cyber --tokenizer_name_or_path HuggingFaceH4/zephyr-7b-beta --apply_chat_template

python -m src.logit_lens.plot_logit_lens --task wmdp-cyber --apply_chat_template