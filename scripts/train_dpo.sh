#!/bin/shd
eval "$(conda shell.bash hook)"
conda activate unlearning

model=HuggingFaceH4/zephyr-7b-beta 

experimentConfigs=(
    '3,0.1,1e-6,0.5,0.5,0.25,0.25,cyber'
    '3,0.1,5e-6,0.5,0.5,0.25,0.25,cyber'
    '3,0.05,1e-6,0.5,0.5,0.25,0.25,cyber'
    '3,0.05,5e-6,0.5,0.5,0.25,0.25,cyber'
    '3,0.5,1e-6,0.5,0.5,0.25,0.25,cyber'
    '3,0.5,5e-6,0.5,0.5,0.25,0.25,cyber'
)

for config in "${experimentConfigs[@]}"; do

    while IFS=',' read -r n_epochs beta lr use_chat_temp p1 p2 p3 subset; do

        accelerate launch --config_file=src/dpo/accelerate/deepspeed3.yaml --num_processes 8 \
            -m src.dpo.dpo \
                --dataset_proportions="$p1,$p2,$p3" \
                --model_name_or_path $model \
                --per_device_train_batch_size 4 \
                --max_length 1024 \
                --max_prompt_length 1024 \
                --max_target_length 1024 \
                --num_train_epochs $n_epochs \
                --beta $beta \
                --learning_rate $lr \
                --gradient_accumulation_steps 1 \
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

    done <<< "$config"

done

echo "Processing has finished"

conda deactivate