#!/bin/shd
eval "$(conda shell.bash hook)"
conda activate unlearning

model=HuggingFaceH4/zephyr-7b-beta 

    # '3,0.05,1e-5,0.5,0.5,bio'
    # '3,0.05,5e-5,0.5,0.5,bio'
    # '3,0.1,5e-5,0.5,2,bio'
    # '3,0.1,1e-4,0.5,2,bio'
    # '3,0.1,5e-4,0.5,2,bio'
    # '3,0.2,1e-5,0.5,1,bio'
    # '3,0.2,5e-5,0.5,1,bio'
    # '3,0.2,1e-4,0.5,1,bio'
    # '3,0.2,5e-4,0.5,1,bio'
    # '3,0.5,1e-4,0.5,5,bio'
experimentConfigs=(
    '3,0.05,1e-5,0.5,0.5,cyber'
    '3,0.05,5e-5,0.5,0.5,cyber'
    '3,0.1,1e-5,0.5,2,cyber'
    '3,0.1,5e-5,0.5,2,cyber'
    '3,0.1,1e-4,0.5,2,cyber'
    '3,0.1,5e-4,0.5,2,cyber'
    '3,0.2,1e-5,0.5,1,cyber'
    '3,0.2,5e-5,0.5,1,cyber'
    '3,0.2,1e-4,0.5,1,cyber'
    '3,0.2,5e-4,0.5,1,cyber'
    '3,0.5,1e-4,0.5,5,cyber'
)

for config in "${experimentConfigs[@]}"; do

    while IFS=',' read -r n_epochs beta lr use_chat_temp retain_mult subset; do

        accelerate launch --config_file=src/npo/accelerate/deepspeed3.yaml --num_processes 8 \
            -m src.npo.npo \
                --model_name_or_path $model \
                --per_device_train_batch_size 3 \
                --max_length 1024 \
                --max_prompt_length 1024 \
                --max_target_length 1024 \
                --num_train_epochs $n_epochs \
                --retain_mult $retain_mult\
                --beta $beta \
                --learning_rate $lr \
                --gradient_accumulation_steps 3 \
                --logging_steps 10 \
                --eval_steps 100 \
                --output_dir="" \
                --warmup_steps 150 \
                --report_to wandb \
                --bf16 \
                --logging_first_step \
                --no_remove_unused_columns \
                --gradient_checkpointing 1 \
                --use_chat_template_p $use_chat_temp \
                --subset $subset
                # --attn_implementation=flash_attention_2 \

    done <<< "$config"

done

echo "Processing has finished"

conda deactivate