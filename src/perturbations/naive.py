import argparse
import datetime
import os
import random
import gc
import string
import itertools
from typing import Callable

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformer_lens import HookedTransformer
from pandas import DataFrame
from tqdm import tqdm

from src.util.helpers import create_if_not_exists, save_as_json, seed_everything, get_question
from src.util.globals import OUTPUT_DIR, SEED
from src.benchmarking.benchmark import Benchmark

# Default parameters
SEED = 42
N_DEVICES = 2 # number of GPUs per model
HF_DATASET_DIR = "cais/wmdp"
HF_DATASET_NAME = "wmdp-bio"
HF_DATASET_SPLIT = "test"
PERTURBATION_FREQUENCY = "1,3,5,7,11,15,0.1,0.2,0.3"
PERTURBATION_TYPE = "!,_, ,),;,?,5,~,^,.,-,Q,shuffle"
UNLEARNED_MODEL = "cais/Zephyr_RMU"
ORIGINAL_MODEL = "HuggingFaceH4/zephyr-7b-beta"
BASE_MODEL = "mistralai/Mistral-7B-v0.1"


# Global variables
SYSTEM_PROMPT = "You are a helpful assistant"
DEBUG_MODE = False

def get_random_non_alphabetic():
    return random.choice(string.punctuation + string.digits)

def get_perturbation_fn(perturbation_freq, perturbation_type) -> Callable:
    # define condition based on perturbation_freq
    if perturbation_freq >= 1:
        perturbation_freq = int(perturbation_freq)
        condition = lambda idx: idx % perturbation_freq == 0
    else:
        condition = lambda _: random.random() < perturbation_freq
    
    if perturbation_type != "shuffle":
        def perturb(prompt):
            return "".join([char + perturbation_type if condition(idx) else char for idx, char in enumerate(prompt)])
    else:
        def perturb(prompt):
            return "".join([char + get_random_non_alphabetic() if condition(idx) else char for idx, char in enumerate(prompt)])

    return perturb    
    

def run_batch_experiments(args, output_dest):
    # load models
    tokenizer = AutoTokenizer.from_pretrained(args.original_model)

    model = AutoModelForCausalLM.from_pretrained(args.unlearned_model,
                                                    torch_dtype=torch.bfloat16,
                                                    device_map="cuda:0",
                                                    low_cpu_mem_usage=True)
    tokenizer = AutoTokenizer.from_pretrained(args.unlearned_model, device_map="cuda:0",)

    model_original = AutoModelForCausalLM.from_pretrained(args.original_model,
                                                    torch_dtype=torch.bfloat16,
                                                    device_map="cuda:1",
                                                    low_cpu_mem_usage=True)
    tokenizer_original = AutoTokenizer.from_pretrained(args.original_model, device_map="cuda:1",)
    
    configs = [(1, "")] + list(itertools.product(args.perturbation_freq, args.perturbation_type))
    n_configs = len(configs)
    
    for i, (perturbation_freq, perturbation_type) in enumerate(configs):
        bench = Benchmark(
            output_dir=output_dest,
            tasks=[args.hf_dataset_name],
            wandb_project="perturbations-naive",
            run_name=f"{perturbation_freq}_{perturbation_type}",
            config={"perturbation_freq": perturbation_freq, "perturbation_type": perturbation_type},
            save_requests=True,
            upload_requests_to_hf=False,
            wmdp_whole_perturbation= get_perturbation_fn(perturbation_freq, perturbation_type),
        )
        bench.run(model, tokenizer, model_original, tokenizer_original)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--perturbation_freq", "-p", 
                        help="""Perturbation will be applied to every nth position. 
                                Provide coma-separated list of values. If float 0 < x < 1 is provided, 
                                it will be treated as probability of applying perturbation at every position""",
                        type=str,
                        default = PERTURBATION_FREQUENCY)
    parser.add_argument("--perturbation_type", "-t",
                        help="Provide coma-separated list of charatecters to use as perturbations. Include 'shuffle' to use a mixture", 
                        type=str,
                        default=PERTURBATION_TYPE)
    parser.add_argument("--apply_chat_template", action="store_true")
    parser.add_argument("--unlearned_model", type=str, default=UNLEARNED_MODEL)
    parser.add_argument("--original_model", type=str, default=ORIGINAL_MODEL)
    parser.add_argument("--base_model", type=str, default=BASE_MODEL, 
                        help="HF model name that can be used with transformerlens and is a base for unlearned & original model")
    parser.add_argument("--hf_dataset_dir", default=HF_DATASET_DIR)
    parser.add_argument("--hf_dataset_name", default=HF_DATASET_NAME)
    parser.add_argument("--hf_dataset_split", default=HF_DATASET_SPLIT)
    parser.add_argument("--output_dir", default=OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--n_devices", type=int, default=N_DEVICES)
    parser.add_argument("--debug_mode", action="store_true")
    args = parser.parse_args()
    
    seed_everything(args.seed)
    
    # Parse input
    args.perturbation_freq = [float(i) for i in args.perturbation_freq.split(",")]
    args.perturbation_type = args.perturbation_type.split(",")
    if args.debug_mode:
        DEBUG_MODE = True
            
    # Get destination subdir
    output_dest = os.path.join(OUTPUT_DIR, "perturbations", "naive", f"{args.unlearned_model.split('/')[-1]}", f"{args.hf_dataset_name}")
    create_if_not_exists(output_dest)
    
    # Actual experiments
    results, logits, ans_idx = run_batch_experiments(args, output_dest)