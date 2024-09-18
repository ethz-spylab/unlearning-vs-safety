#!/bin/shd
eval "$(conda shell.bash hook)"
conda activate finetuning

# finetune bio
for n_samples in 5 10 50 100 500 1000
do
    for dataset in wmdp_bio-forget-corpus-mc wmdp_bio-retain-corpus-mc wikitext-bio-mc
    do 
        CUDA_VISIBLE_DEVICES=5 python -m src.finetuning.finetune \
            --model cais/Zephyr_RMU \
            --dataset $dataset \
            --n_samples $n_samples &
        
        CUDA_VISIBLE_DEVICES=6 python -m src.finetuning.finetune \
            --model J4Q8/zephyr-dpo-bio-priv1 \
            --dataset $dataset \
            --n_samples $n_samples &
        
        CUDA_VISIBLE_DEVICES=7 python -m src.finetuning.finetune \
            --model J4Q8/zephyr-npo-bio-priv1 \
            --dataset $dataset \
            --n_samples $n_samples
        
        wait
    done
done

# finetune cyber
for n_samples in 5 10 50 100 500 1000
do
    for dataset in wmdp_cyber-forget-corpus-mc wmdp_cyber-retain-corpus-mc wikitext-cyber-mc
    do 
        CUDA_VISIBLE_DEVICES=5 python -m src.finetuning.finetune \
            --model cais/Zephyr_RMU \
            --dataset $dataset \
            --n_samples $n_samples &
        
        CUDA_VISIBLE_DEVICES=6 python -m src.finetuning.finetune \
            --model J4Q8/zephyr-dpo-cyber-priv1 \
            --dataset $dataset \
            --n_samples $n_samples &
        
        CUDA_VISIBLE_DEVICES=7 python -m src.finetuning.finetune \
            --model J4Q8/zephyr-npo-cyber-priv1 \
            --dataset $dataset \
            --n_samples $n_samples
        
        wait
    done
done