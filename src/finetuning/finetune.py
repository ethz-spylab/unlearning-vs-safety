import os
import json
import datetime
from argparse import ArgumentParser
from tkinter import E

import wandb
import torch
from trl import SFTTrainer
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from transformers import TrainingArguments, AutoTokenizer, AutoModelForCausalLM

from src.benchmarking.benchmark import Benchmark
from src.util.globals import OUTPUT_DIR
from src.util.helpers import seed_everything, create_if_not_exists, save_tokenizer, save_as_json
from src.util.data import *

DATASET_REGISTRY = {
    "wmdp_bio-forget-corpus" : get_bio_forget_dataset,
    "wmdp_bio-retain-corpus" : get_bio_retain_dataset,
    "wmdp_cyber-retain-corpus" : get_cyber_retain_dataset,
    "wmdp_cyber-forget-corpus" : get_cyber_forget_dataset,
    "wikitext": get_wikitext_dataset,
    "wmdp_bio-forget-corpus-mc": get_bio_forget_mc_dataset,
    "wmdp_bio-retain-corpus-mc": get_bio_retain_mc_dataset,
    "wmdp_cyber-forget-corpus-mc": get_cyber_forget_mc_dataset,
    "wmdp_cyber-retain-corpus-mc": get_cyber_retain_mc_dataset,
    "wikitext-bio-mc": get_wikitext_bio_mc_dataset,
    "wikitext-cyber-mc": get_wikitext_cyber_mc_dataset,
}

# Parameters

SEED = 42
MAX_SEQ_LEN = 1024
DTYPE = torch.bfloat16
LORA_RANK = 128
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj","gate_proj", "up_proj", "down_proj",]
LORA_ALPHA = 16
LORA_DROPOUT = 0
BATCH_SIZE = 1 #old: 2
GRAD_ACCUM_STEPS = 1 #old: 4
EPOCHS = 3
LR = 2e-4
WARMUP_RATIO = 0.05
WEIGHT_DECAY = 0.01

if __name__ == "__main__":
    # Ensure reproducibility
    seed_everything(SEED)
    
    # Parse input
    parser = ArgumentParser()
    parser.add_argument("--model", "-m", 
                        required=True, 
                        help="Name of the model to finetune")
    parser.add_argument("--tokenizer", 
                        default="HuggingFaceH4/zephyr-7b-beta",)
    parser.add_argument("--dataset", "-d",
                        required=True,
                        choices=list(DATASET_REGISTRY.keys()),
                        help="Name of the dataset to use for finetuning")
    parser.add_argument("--n_samples", "-n", 
                        type=int,
                        default=100,
                        help="Number of samples in a dataset to use")
    parser.add_argument("--system_prompt", "-s", default="You are a helpful assistant.")
    args = parser.parse_args()
    
    # Get destination subdir
    date = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    model_str = args.model.split("/")[-1]
    run_name = f"{model_str}__{args.dataset}__{args.n_samples}__{date}"
    output_dest = os.path.join(OUTPUT_DIR, "finetuning", run_name)
    create_if_not_exists(output_dest)
    
    # Initialize wandb
    wandb.init(project="finetuning", dir=output_dest, name = run_name)
    
    # Get dataset
    dataset = DATASET_REGISTRY[args.dataset](limit=args.n_samples, system_prompt=args.system_prompt)
    
    # Get model
    model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=DTYPE,
            device_map="auto",
            trust_remote_code=True,
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    model.gradient_checkpointing_enable()
    
    # Get tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    tokenizer.padding_side = 'right'
    tokenizer.pad_token = tokenizer.eos_token
    
    # Set up LORA
    peft_config = LoraConfig(
        model,
        r = LORA_RANK, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules = LORA_TARGET_MODULES,
        lora_alpha = LORA_ALPHA,
        lora_dropout = LORA_DROPOUT, # Supports any, but = 0 is optimized
        bias = "none",    # Supports any, but = "none" is optimized
    )
    model = get_peft_model(model, peft_config)
    
    # initialize trainer
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        max_seq_length = MAX_SEQ_LEN,
        dataset_num_proc = 2,
        peft_config=peft_config,
        packing = False, # Can make training 5x faster for short sequences.
        args = TrainingArguments( #these can be retrieved later on so no need to make them into Global constants for saving purposes
            per_device_train_batch_size = BATCH_SIZE,
            gradient_accumulation_steps = GRAD_ACCUM_STEPS,
            warmup_ratio = WARMUP_RATIO,
            num_train_epochs = EPOCHS,
            learning_rate = LR,
            logging_steps = 1,
            optim = "adamw_torch",
            weight_decay = WEIGHT_DECAY,
            lr_scheduler_type = "linear",
            seed = SEED,
            output_dir = output_dest,
            report_to="wandb",
            run_name=run_name,
            save_strategy="no",
        ),
    )
    # hack to make trainer return eval loss for peft model
    trainer.can_return_loss = True
    
    # GPU memory usage monitoring - current stats
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")
    
    ### Finetune
    trainer_stats = trainer.train()
    
    # GPU memory usage monitoring - final stats
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory         /max_memory*100, 3)
    lora_percentage = round(used_memory_for_lora/max_memory*100, 3)
    print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
    print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
    print(f"Peak reserved memory = {used_memory} GB.")
    print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
    print(f"Peak reserved memory % of max memory = {used_percentage} %.")
    print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")
    
    ### Log parameters
    
    # Get user-defined global variables (for reproducibility)
    # assumption = user defined globals consist only uppercase letter
    global_params = {key : globals()[key] for key in list(globals().keys()) if key.isupper()}
    save_as_json(output_dest, "global_params.json", global_params)
    
    # Get training arguments
    train_params = vars(trainer.args)
    save_as_json(output_dest, "train_params.json", train_params)
    
    ### Evaluate model
    wandb.finish() # benchmark has its own wandb.init
    merged_model = trainer.model.merge_and_unload()
    bench = Benchmark(
        output_dest,
        tasks=["mmlu", "wmdp-bio", "wmdp-chem", "wmdp-cyber"],
        wandb_project="finetuning_results",
        run_name=run_name,
        save_requests=False,
        upload_requests_to_hf=False,
    )
    bench.run(merged_model, tokenizer)