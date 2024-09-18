import os
import json
from pdb import run
import random
import string
import argparse

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from src.util.helpers import create_if_not_exists, save_as_json, seed_everything, get_question
from src.util.globals import OUTPUT_DIR, SEED, WMDP_OPTIONS
from src.benchmarking.benchmark import Benchmark

# Default parameters
PARTIAL_MAX_ITER = 500
TOTAL_MAX_ITER = 500
HF_DATASET_DIR = "cais/wmdp"
HF_DATASET_NAME = "wmdp-bio"
HF_DATASET_SPLIT = "test"
UNLEARNED_MODEL = "cais/Zephyr_RMU"
ORIGINAL_MODEL = "HuggingFaceH4/zephyr-7b-beta"


def _informed_perturbation(prompt, cos, encodings, threshold, only_first):
    def get_random_non_alphabetic():
        return random.choice(string.punctuation + string.digits)
    
    def perturb(string):
        if len(string) == 0:
            print("Weird stuff going on, string has length 0. You shouldn't be here")
            return ""
        elif len(string) == 1:
            default_replacement_tokens = ["~", "^"]
            if  string in default_replacement_tokens:
                return get_random_non_alphabetic()
            else:
                return random.choice(default_replacement_tokens) + string
        else:
            # by default, we insert a space at a random position
            split_idx = random.randint(1, len(string)-1)
            
            # print(repr(string))
            return string[:split_idx] + " " + string[split_idx:]

    positions = encodings['offset_mapping'][0]
    
    if only_first:
        perturb_token_id = [i for i, cos_val in enumerate(cos) if cos_val > threshold]
        # stop condition
        if len(perturb_token_id) == 0:
            return prompt
        else:
            perturb_token_id = perturb_token_id[0]
            
        perturbed_prompt = prompt[:positions[perturb_token_id][0]] \
            + perturb(prompt[positions[perturb_token_id][0]:positions[perturb_token_id][1]]) \
            + prompt[positions[perturb_token_id][1]:]
        # print("AAAA", perturb_token_id, cos[perturb_token_id], repr(perturbed_prompt))
    else:            
        perturbed_prompt = "".join([perturb(prompt[positions[i][0]:positions[i][1]]) 
                                    if cos_val > threshold else prompt[positions[i][0]:positions[i][1]] 
                                    for i, cos_val in enumerate(cos)])
    
    return perturbed_prompt

def informed_perturbation(prompt, threshold, tokenizer, model, ablation_direction, max_iter = 500, only_first=False):
    old_prompt = ""
    new_prompt = prompt
    
    for _ in tqdm(range(max_iter), disable=True):
        
        if old_prompt == new_prompt:
            break
        
        old_prompt = new_prompt
        
        encodings = tokenizer(old_prompt, return_offsets_mapping=True, return_tensors="pt")
        model(encodings["input_ids"].to("cuda:0"))
        layer_7_out = model.layer_7_out.float().cpu().detach().numpy()
        cos = torch.nn.CosineSimilarity(dim=-1)(torch.tensor(ablation_direction), torch.tensor(layer_7_out)).numpy().squeeze()
    
        new_prompt = _informed_perturbation(old_prompt, cos, encodings, threshold, only_first)
        # print(repr(new_prompt))

    # print(f"Result: {repr(new_prompt)}")
    encodings = tokenizer(new_prompt, return_offsets_mapping=True, return_tensors="pt")
    model(encodings["input_ids"].to("cuda:0"))
    layer_7_out = model.layer_7_out.float().cpu().detach().numpy()
    cos = torch.nn.CosineSimilarity(dim=-1)(torch.tensor(ablation_direction), torch.tensor(layer_7_out)).numpy().squeeze()
    # for i, (token_id, pos) in enumerate(zip(encodings['input_ids'][0], encodings['offset_mapping'][0])):
    #     print(repr(tokenizer.decode(token_id)), pos, repr(new_prompt[pos[0]:pos[1]]), cos[i])
    
    return new_prompt

def get_noise_direction() -> np.ndarray:
    noise_data = np.load("tmp/noise_vectors.npy")
    print(noise_data.shape)

    ground_truth = np.load("tmp/ground_truth_vectors.npy")
    print(ground_truth.shape)

    clean_data = np.load("tmp/clean_wikitext_vectors.npy")
    print(clean_data.shape)
    clean_tokens = np.load("tmp/wikitext_tokens.npy")
    print(clean_tokens.shape)
    noise_tokens = np.load("tmp/noise_vectors_token_list.npy")
    print(noise_tokens.shape)

    clean_filter = np.logical_and(clean_tokens != '\n', clean_tokens != '<s>')
    noise_filter = np.logical_and(noise_tokens != '\n', noise_tokens != '<s>')
    ablation_direction = np.mean(noise_data, axis=0) - np.mean(ground_truth, axis=0)
    return ablation_direction


def main_experiment(args, noise_direction, output_dest) -> tuple[float, float, dict]:

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


    def hook(_, __, output):
        model.layer_7_out = output[0]

    model.layer_7_out = torch.empty(0)
    model.model.layers[7].register_forward_hook(hook)
    pass
    
    bench = Benchmark(
        output_dir=output_dest,
        tasks=[args.hf_dataset_name],
        wandb_project="perturbations-informed",
        run_name=f"{args.thresh_partial}_{args.thresh_total}",
        save_requests=True,
        upload_requests_to_hf=False,
        wmdp_element_perturbation= lambda x: informed_perturbation(x,args.thresh_partial, tokenizer, model, noise_direction, only_first=True, max_iter=args.max_iter_partial),
        wmdp_whole_perturbation= lambda x: informed_perturbation(x, args.thresh_total, tokenizer, model, noise_direction, only_first=True, max_iter=args.max_iter_total)
    )
    
    bench.run(model, tokenizer, model_original, tokenizer_original)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--thresh_partial", type=float)
    parser.add_argument("--max_iter_partial", type=int, default=PARTIAL_MAX_ITER)
    parser.add_argument("--thresh_total", type=float)
    parser.add_argument("--max_iter_total", type=int, default=TOTAL_MAX_ITER)
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--unlearned_model", type=str, default=UNLEARNED_MODEL)
    parser.add_argument("--original_model", type=str, default=ORIGINAL_MODEL)
    parser.add_argument("--hf_dataset_dir", default=HF_DATASET_DIR)
    parser.add_argument("--hf_dataset_name", default=HF_DATASET_NAME)
    parser.add_argument("--hf_dataset_split", default=HF_DATASET_SPLIT)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    
    seed_everything(args.seed)
    
    # Get destination subdir
    output_dest = os.path.join(OUTPUT_DIR, "perturbations", "informed", f"{args.unlearned_model.split('/')[-1]}", f"{args.hf_dataset_name}")
    create_if_not_exists(output_dest)
    
    # Run experiment
    noise_direction = np.zeros(4096)#get_noise_direction()
    unlearned_acc, original_acc, perturbed_prompts = main_experiment(args, noise_direction, output_dest)