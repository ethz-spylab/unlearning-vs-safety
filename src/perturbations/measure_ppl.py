import os
import argparse
import json
import pathlib
import re

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.benchmarking.benchmark import Benchmark
from src.util.globals import OUTPUT_DIR, SEED
from src.util.helpers import create_if_not_exists, seed_everything


def to_chat(prompt, tokenizer) -> str:
    messages = [{"role": "system", "content": ""},
                {"role": "user", "content": prompt}]
    chat = tokenizer.apply_chat_template(messages, add_generation_token=True, tokenize=False)
    return chat

def prepare_requests(request_file) -> dict:
    with open(request_file, "r") as f:
        requests = json.load(f)
    return requests

def prepare_original_requests(args, output_dir) -> dict:
    benchmark = Benchmark(output_dir,
                        tasks=[args.task],
                        upload_requests_to_hf=False,
                        save_requests=False,)
    context = benchmark.get_context(args.task)
    dataset = load_dataset("cais/wmdp", args.task, split="test")
    requests = benchmark.generate_wmdp_requests(dataset, context)
    return requests

def get_ppl(prompt_ids, generation_ids, model):
    input_ids = torch.cat([prompt_ids, generation_ids], dim=1)
    target_ids = input_ids.clone()
    target_ids[:, :-generation_ids.size(1)] = -100

    with torch.no_grad():
        outputs = model(input_ids, labels=target_ids)

        # loss is calculated using CrossEntropyLoss which averages over valid labels
        # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
        # to the left by 1.
        neg_log_likelihood = outputs.loss
    
    return torch.exp(neg_log_likelihood)

def measure_ppl(args, model, tokenizer, model_original, tokenizer_original, output_dest, apply_chat_template, request_file):
    clean_req = prepare_original_requests(args, output_dest)
    perturbed_req = prepare_requests(request_file)
    
    ppl = {
        "original_gen_original_model": [],
        "original_gen_rmu": [],
        "pertubed_q_original_gen_rmu": [],
        "rmu_gen_original_model": [],
        "perturbed_q_gen_original_model": [],
    }
    for i in tqdm(range(len(clean_req))):
        pure_question = clean_req[i]["question"]
        pure_question = to_chat(pure_question, tokenizer) if apply_chat_template else pure_question
        
        # get original generation
        pure_question_ids = tokenizer_original(pure_question, return_tensors="pt")["input_ids"].to(model.device)
        original_generation = model_original.generate(pure_question_ids, max_new_tokens=args.n_tokens, min_new_tokens=args.n_tokens)
        original_generation = original_generation[:, -args.n_tokens:]
        
        # get ppl for original generation
        ppl["original_gen_original_model"].append(get_ppl(pure_question_ids, original_generation, model_original).item())
        
        # get ppl of the original generation using unlearned model
        ppl["original_gen_rmu"].append(get_ppl(pure_question_ids, original_generation, model).item())
        
        # get ppl of the original generation given the perturbed question
        perturbed_question = perturbed_req[str(i)]["question"]
        perturbed_question = to_chat(perturbed_question, tokenizer) if apply_chat_template else perturbed_question
        perturbed_question_ids = tokenizer(perturbed_question, return_tensors="pt")["input_ids"].to(model.device)
        ppl["pertubed_q_original_gen_rmu"].append(get_ppl(perturbed_question_ids, original_generation, model).item())
        
        # use original model to get perplexity of RMU's generation
        rmu_gen = model.generate(pure_question_ids, max_new_tokens=args.n_tokens, min_new_tokens=args.n_tokens)
        rmu_gen = rmu_gen[:, -args.n_tokens:]
        ppl["rmu_gen_original_model"].append(get_ppl(pure_question_ids, rmu_gen, model_original).item())
        
        # use original model to get perplexity of RMU's generation given the perturbed question
        perturbed_generation = model.generate(perturbed_question_ids, max_new_tokens=args.n_tokens, min_new_tokens=args.n_tokens)
        perturbed_generation = perturbed_generation[:, -args.n_tokens:]
        # we measure perplexity as if it came from the original model with pure question
        ppl["perturbed_q_gen_original_model"].append(get_ppl(pure_question_ids, perturbed_generation, model_original).item())
        
    means = {key: np.mean(val) for key, val in ppl.items()}
    return means, ppl


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--unlearned_model", type=str, required=True, help="Name or path of the unlearned model")
    parser.add_argument("--unlearned_tokenizer", type=str, help="Name or path of the unlearned tokenizer")
    parser.add_argument("--original_model", type=str, required=True, help="Name or path of the original model")
    parser.add_argument("--original_tokenizer", type=str, help="Name or path of the unlearned tokenizer")
    parser.add_argument("--task", choices=["wmdp-bio", "wmdp-cyber"], help="The domain of the unlearned model")
    parser.add_argument("--n_tokens", type=int, default=50, help="Number of tokens to generate over which perplexity is calculated")
    parser.add_argument("--perturbations_dir", type=str, required=True, help="Directory containing the perturbations")
    parser.add_argument("--apply_chat_template", action="store_true", help="Whether to apply the chat template to the promtp")
    args = parser.parse_args()
    
    # process arguments
    if args.unlearned_tokenizer is None:
        args.unlearned_tokenizer = args.unlearned_model
    
    if args.original_tokenizer is None:
        args.original_tokenizer = args.original_model
    
    # create output dir
    output_dest = os.path.join(OUTPUT_DIR, "perturbations", "informed", f"{args.unlearned_model.split("/")[-1]}", args.task, "perplexity")
    create_if_not_exists(output_dest)
    
    # prepare seed and set torch to no grad
    seed_everything(SEED)
    torch.set_grad_enabled(False)
    
    # load models and tokenizers
    model = AutoModelForCausalLM.from_pretrained(args.unlearned_model,
                                                    torch_dtype=torch.bfloat16,
                                                    device_map="auto",
                                                    low_cpu_mem_usage=True)
    tokenizer = AutoTokenizer.from_pretrained(args.unlearned_tokenizer, device_map=model.device)
    
    model_original = AutoModelForCausalLM.from_pretrained(args.original_model,
                                                    torch_dtype=torch.bfloat16,
                                                    device_map="auto",
                                                    low_cpu_mem_usage=True)
    tokenizer_original = AutoTokenizer.from_pretrained(args.original_tokenizer, device_map=model_original.device)
    
    # get request files
    files = os.listdir(args.perturbations_dir)
    json_files = [file for file in files if file.endswith(".json")]
    
    for file in json_files:            
        partial, total = re.search(r"(\d.\d)_(\d.\d).json", file).groups((1,2))
    
        mean_ppl, ppl_raw = measure_ppl(args, model, tokenizer, model_original, tokenizer_original, output_dest, False, os.path.join(args.perturbations_dir, file))
        mean_ppl_chat, ppl_raw_chat = measure_ppl(args, model, tokenizer, model_original, tokenizer_original, output_dest, True, os.path.join(args.perturbations_dir, file))
            
        mean_ppl.update({f"{key}_chat": value for key, value in mean_ppl_chat.items()})
        mean_ppl.update({"partial": partial, "total": total})
        mean_ppl.update(vars(args))
        print("Results saved to:", output_dest)
        with open(os.path.join(output_dest, "results.jsonl"), "a") as f:
            json.dump(mean_ppl, f)
            f.write("\n")