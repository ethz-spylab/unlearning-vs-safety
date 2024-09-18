import os
import argparse
import json
import pathlib

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.benchmarking.benchmark import Benchmark
from src.util.globals import OUTPUT_DIR, SEED
from src.util.helpers import create_if_not_exists, seed_everything

ADV_PREFIX_CSV = "adv_prefixes.csv"

def to_chat(prompt, tokenizer) -> str:
    messages = [{"role": "system", "content": ""},
                {"role": "user", "content": prompt}]
    chat = tokenizer.apply_chat_template(messages, add_generation_token=True, tokenize=False)
    return chat

def prepare_requests(args, adv_prefix, output_dir) -> dict:
    benchmark = Benchmark(output_dir,
                        wmdp_adv_prefix=adv_prefix,
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

def measure_ppl(args, model, tokenizer, model_original, tokenizer_original, output_dest, adv_prefix, apply_chat_template):
    req_with_adv_prefix = prepare_requests(args, adv_prefix, output_dest)
    req_no_adv_prefix = prepare_requests(args, "", output_dest)
    
    ppl = {
        "original": [],
        "original_no_adv_prefix": [],
        "original_with_adv_prefix": [],
        "no_pref_gen_zephyr": [],
        "pref_gen_zephyr": [],
    }
    for i in tqdm(range(len(req_no_adv_prefix))):
        pure_question = req_no_adv_prefix[i]["question"]
        pure_question = to_chat(pure_question, tokenizer) if apply_chat_template else pure_question
        
        # get original generation
        pure_question_ids = tokenizer_original(pure_question, return_tensors="pt")["input_ids"].to(model.device)
        original_generation = model_original.generate(pure_question_ids, max_new_tokens=args.n_tokens, min_new_tokens=args.n_tokens)
        original_generation = original_generation[:, -args.n_tokens:]
        
        # get ppl for original generation
        ppl["original"].append(get_ppl(pure_question_ids, original_generation, model_original).item())
        
        # get ppl using unlearned model without adv prefix
        ppl["original_no_adv_prefix"].append(get_ppl(pure_question_ids, original_generation, model).item())
        
        # get ppl using unlearned model with adv prefix
        question_with_adv = req_with_adv_prefix[i]["question"]
        question_with_adv = to_chat(question_with_adv, tokenizer) if apply_chat_template else question_with_adv
        question_with_adv_ids = tokenizer(question_with_adv, return_tensors="pt")["input_ids"].to(model.device)
        ppl["original_with_adv_prefix"].append(get_ppl(question_with_adv_ids, original_generation, model).item())
        
        # use original model to get perplexity of RMU's generation without adv prefix
        no_pref_generation = model.generate(pure_question_ids, max_new_tokens=args.n_tokens, min_new_tokens=args.n_tokens)
        no_pref_generation = no_pref_generation[:, -args.n_tokens:]
        ppl["no_pref_gen_zephyr"].append(get_ppl(pure_question_ids, no_pref_generation, model_original).item())
        
        # use original model to get perplexity of RMU's generation with adv prefix
        pref_generation = model.generate(question_with_adv_ids, max_new_tokens=args.n_tokens, min_new_tokens=args.n_tokens)
        pref_generation = pref_generation[:, -args.n_tokens:]
        # me measure perplexity as if it came from the original model with pure question
        ppl["pref_gen_zephyr"].append(get_ppl(pure_question_ids, pref_generation, model_original).item())
        
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
    parser.add_argument("--apply_chat_template", action="store_true", help="Whether to apply the chat template to the promtp")
    args = parser.parse_args()
    
    # process arguments
    if args.unlearned_tokenizer is None:
        args.unlearned_tokenizer = args.unlearned_model
    
    if args.original_tokenizer is None:
        args.original_tokenizer = args.original_model
    
    # create output dir
    output_dest = os.path.join(OUTPUT_DIR, "adv_prefixes", f"{args.unlearned_model.replace("/", "_")}", "perplexity")
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
    
    file = pathlib.Path(__file__).parent.resolve() / ADV_PREFIX_CSV
    adv_prefix_df = pd.read_csv(file)
    
    for index, row in adv_prefix_df.iterrows():
        mean_ppl, ppl_raw = measure_ppl(args, model, tokenizer, model_original, tokenizer_original, output_dest, row["prefix"], False)
        mean_ppl_chat, ppl_raw_chat = measure_ppl(args, model, tokenizer, model_original, tokenizer_original, output_dest, row["prefix"], True)
        
        mean_ppl.update({f"{key}_chat": value for key, value in mean_ppl_chat.items()})
        mean_ppl.update(vars(args))
        mean_ppl.update(row.to_dict())
        print("Results saved to:", output_dest)
        with open(os.path.join(output_dest, "results.jsonl"), "a") as f:
            json.dump(mean_ppl, f)
            f.write("\n")