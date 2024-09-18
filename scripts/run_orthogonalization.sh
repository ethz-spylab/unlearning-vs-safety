#!/bin/shd
eval "$(conda shell.bash hook)"
conda activate unlearning

for subset in cyber bio
    do
    # first compute all layers so that they don't have to be recomputed later
    CUDA_VISIBLE_DEVICES=0,1 python -m src.orthogonalization.orthogonalization \
        --task wmdp-${subset} --layers all --unlearned_model cais/Zephyr_RMU --unlearned_tokenizer HuggingFaceH4/zephyr-7b-beta --original_model HuggingFaceH4/zephyr-7b-beta
    sleep 2
    CUDA_VISIBLE_DEVICES=0,1 python -m src.orthogonalization.orthogonalization \
        --task wmdp-${subset} --layers all --unlearned_model J4Q8/zephyr-dpo-${subset} --unlearned_tokenizer HuggingFaceH4/zephyr-7b-beta --original_model HuggingFaceH4/zephyr-7b-beta
    sleep 2
    CUDA_VISIBLE_DEVICES=0,1 python -m src.orthogonalization.orthogonalization \
        --task wmdp-${subset} --layers all --unlearned_model J4Q8/zephyr-npo-${subset} --unlearned_tokenizer HuggingFaceH4/zephyr-7b-beta --original_model HuggingFaceH4/zephyr-7b-beta 
    wait
    # try all unlearned layers
    CUDA_VISIBLE_DEVICES=0,1 python -m src.orthogonalization.orthogonalization \
        --task wmdp-${subset} --layers 5,6,7 --unlearned_model cais/Zephyr_RMU --unlearned_tokenizer HuggingFaceH4/zephyr-7b-beta --original_model HuggingFaceH4/zephyr-7b-beta

    for layer in 0 7 15 23 31
    do
        CUDA_VISIBLE_DEVICES=0,1 python -m src.orthogonalization.orthogonalization \
            --task wmdp-${subset} --layers $layer --unlearned_model cais/Zephyr_RMU --unlearned_tokenizer HuggingFaceH4/zephyr-7b-beta --original_model HuggingFaceH4/zephyr-7b-beta
        sleep 2
        CUDA_VISIBLE_DEVICES=0,1 python -m src.orthogonalization.orthogonalization \
            --task wmdp-${subset} --layers $layer --unlearned_model J4Q8/zephyr-dpo-${subset} --unlearned_tokenizer HuggingFaceH4/zephyr-7b-beta --original_model HuggingFaceH4/zephyr-7b-beta
        sleep 2
        CUDA_VISIBLE_DEVICES=0,1 python -m src.orthogonalization.orthogonalization \
            --task wmdp-${subset} --layers $layer --unlearned_model J4Q8/zephyr-npo-${subset} --unlearned_tokenizer HuggingFaceH4/zephyr-7b-beta --original_model HuggingFaceH4/zephyr-7b-beta 
        wait
    done

done