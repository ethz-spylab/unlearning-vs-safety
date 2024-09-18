import os
import json
import datetime
import argparse
from copy import copy
from typing import Literal

import torch
import numpy as np
from jaxtyping import Float
from torch import Tensor
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from src.util.helpers import save_as_json, seed_everything, create_if_not_exists, get_question
from src.util.globals import OUTPUT_DIR, WMDP_OPTIONS, SEED, WMDP_TASKS
from src.benchmarking.benchmark import Benchmark

def get_zscore(data: Tensor) -> Tensor:
    """Along the first axis, compute the z-score of the data"""
    return (data - torch.mean(data, axis=0)) / torch.std(data, axis=0)

def get_answer_indices(tokenizer) -> Tensor:
    """Get the indices of the answer tokens"""
    answer_ids = tokenizer(WMDP_OPTIONS, return_tensors="pt", add_special_tokens=False)["input_ids"].squeeze()
    print("Answer tokens:", tokenizer.batch_decode(answer_ids), answer_ids)
    return answer_ids

def dpo_to_mc(example):
    question_full = example["prompt"].split("\n\n")[-1]
    parts = question_full.split("\n")
    question_raw = parts[0]
    choices = [part[2:].strip() for part in parts[1:-1]]
    assert len(choices) == 4
    answers = list("ABCD")
    return {
        'question': question_raw,
        'choices': choices,
        'answer': answers.index(example['rejected'][0])
    }

def get_direction_ablation_output_hook(direction: Tensor):
    """Taken from https://github.com/andyrdt/refusal_direction"""
    def hook_fn(module, input, output):
        nonlocal direction

        if isinstance(output, tuple):
            activation: Float[Tensor, "batch_size seq_len d_model"] = output[0]
        else:
            activation: Float[Tensor, "batch_size seq_len d_model"] = output

        direction = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)
        direction = direction.to(activation)
        activation -= (activation @ direction).unsqueeze(-1) * direction 

        if isinstance(output, tuple):
            return (activation, *output[1:])
        else:
            return activation

    return hook_fn

def compute_activations(args, model, tokenizer, model_original, tokenizer_original, output_dir: str) -> tuple[dict, dict]:
    
    # add hooks to the models
    def get_hook(model, layer):
        model.layer_out = {}
        def hook(_, __, output):
            model.layer_out[layer] = output
        return hook
    
    handles = []
    for layer in args.layers:
        
        hook4model = model.model.layers[int(layer)].register_forward_hook(get_hook(model, int(layer)))
        hook4model_original = model_original.model.layers[int(layer)].register_forward_hook(get_hook(model_original, int(layer)))
        
        handles.extend([hook4model, hook4model_original])
        
    ## load datasets
    # contains unlearned knowledge
    wmdp_dataset = load_dataset(f"J4Q8/{args.task.split("-")[-1]}_forget_dpo", split="train").take(1000).map(dpo_to_mc)
    # contains benign knowledge in an article format
    wiki_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    # contains benign knowledge in a question format
    mmlu_dataset = load_dataset("cais/mmlu", "all", split="validation")
    
    # get indices of the answer tokens
    ans_token_indices = get_answer_indices(tokenizer)
    
    # output storage
    noise_triggering_questions = []
    tokens_wmdp = []
    original_activations = {i: torch.empty((0,4096)) for i in args.layers}
    distorted_activations = {i: torch.empty((0,4096)) for i in args.layers}
    
    context = "biology" if args.task == "wmdp-bio" else "cybersecurity"
    
    # first run wmdp dataset
    for i in tqdm(range(len(wmdp_dataset)), smoothing= 0.0):
        prompt, ans = get_question(wmdp_dataset[i], context=context)
        
        # unlearned model
        tokens = tokenizer(prompt, return_tensors="pt").to(model.device)
        logits = model(tokens.input_ids).logits
        pred = WMDP_OPTIONS[torch.argmax(logits[0,-1,ans_token_indices])]
        
        # original model
        logits_original = model_original(tokenizer_original(prompt, return_tensors="pt").to(model_original.device).input_ids).logits
        pred_original = WMDP_OPTIONS[torch.argmax(logits_original[0,-1,ans_token_indices])]
        
        # finding questions that are correctly answered by the original model but not by the unlearned model
        if pred != ans and pred_original == ans:
            noise_triggering_questions.append(i)
            
            tokens_wmdp += tokenizer.batch_decode(tokens.input_ids[0,args.ignore_first_x_tokens:])
            
            # store activations
            for layer in args.layers:
                original_activations[layer] = torch.vstack((original_activations[layer], 
                                                         model_original.layer_out[int(layer)][0][0,args.ignore_first_x_tokens:, :].cpu().float()))
                distorted_activations[layer] = torch.vstack((distorted_activations[layer], 
                                                          model.layer_out[int(layer)][0][0,args.ignore_first_x_tokens:, :].cpu().float()))
    
    
    # run mmlu dataset to get activations resulting from questions
    tokens_mmlu = []
    mmlu_activations = {i: torch.empty((0,4096)) for i in args.layers}    
    for i in tqdm(range(len(mmlu_dataset)), smoothing= 0.0):
        prompt, ans = get_question(mmlu_dataset[i], context=context)
            
        # get activations
        tokens = tokenizer(prompt, return_tensors="pt").to(model.device)
        tokens_mmlu += tokenizer.batch_decode(tokens.input_ids[0])
        _ = model(tokens.input_ids)
        
        # store activations
        for layer in args.layers:
            mmlu_activations[layer] = torch.vstack((mmlu_activations[layer], 
                                                 model.layer_out[int(layer)][0][0].cpu().float()))
        
    # run wiki dataset to get clean activations from normal text
    wikitext_activations = {i: torch.empty((0,4096)) for i in args.layers}
    wikitext_tokens = []
    for sample in tqdm(wiki_dataset["text"][:len(noise_triggering_questions)]): # so that the datasets are more or less the same size
        # get activations
        tokens = tokenizer(sample, return_tensors="pt").to(model.device)
        wikitext_tokens += tokenizer.batch_decode(tokens.input_ids[0])
        _ = model(tokens.input_ids)
        
        # store activations
        for layer in args.layers:
            wikitext_activations[layer] = torch.vstack((wikitext_activations[layer], 
                                                    model.layer_out[int(layer)][0][0].cpu().float()))
    
    # save noise triggering questions
    save_as_json(output_dir, "noise_triggering_questions.json", noise_triggering_questions)
    
    # save activations
    activation_dest = os.path.join(output_dir, "activations")
    create_if_not_exists(activation_dest)
    for layer in args.layers:
        torch.save(original_activations[layer], os.path.join(activation_dest, f"{layer}_original.pth"))
        torch.save(distorted_activations[layer], os.path.join(activation_dest, f"{layer}_distorted.pth"))
        torch.save(mmlu_activations[layer], os.path.join(activation_dest, f"{layer}_mmlu.pth"))
        torch.save(wikitext_activations[layer], os.path.join(activation_dest, f"{layer}_wikitext.pth"))
    
    # save tokens
    save_as_json(activation_dest, "tokens_wmdp.json", tokens_wmdp)
    save_as_json(activation_dest, "tokens_mmlu.json", tokens_mmlu)
    save_as_json(activation_dest, "tokens_wikitext.json", wikitext_tokens)
    
    # remove hooks
    for handle in handles:
        handle.remove()
    
    return {"original": original_activations, 
            "distorted": distorted_activations, 
            "mmlu": mmlu_activations, 
            "wikitext": wikitext_activations}, {"wmdp": tokens_wmdp, "mmlu": tokens_mmlu, "wikitext": wikitext_tokens}

def compute_noise_direction(args, model, tokenizer, model_original, tokenizer_original, output_dir:str) -> None:
    
    # load activations if available
    loaded = False
    if os.path.exists(os.path.join(output_dir, "activations")) and not args.recompute_directions:
        try:
            activations = {}
            for key in ["original", "distorted", "mmlu", "wikitext"]:
                activations[key] = {}
                for layer in args.layers:
                    activations[key][layer] = torch.load(os.path.join(output_dir, "activations", f"{layer}_{key}.pth"))
            
            tokens = {}
            for key in ["wmdp", "mmlu", "wikitext"]:
                tokens[key] = json.load(open(os.path.join(output_dir, "activations", f"tokens_{key}.json")))
            loaded = True
        except:
            print("Error loading activations, recomputing")
        print("Activations loaded")
    
    # compute activations
    if not loaded:
        activations, tokens = compute_activations(args, model, tokenizer, model_original, tokenizer_original, output_dir)
    
    # Find outlier tokens based on wmdp and mmlu, use only 1000 first tokens(this should cover most of the weird common tokens)
    n_tokens = 1000
    outliers = []
    for layer in args.layers:
        dist = torch.cdist(activations["original"][layer][:n_tokens], activations["original"][layer][:n_tokens], p = 1)
        outlier_map = get_zscore(dist.mean(axis=0)) > 3 # 3 standard deviations away from the mean
        outliers_wmdp = np.array(tokens["wmdp"][:n_tokens])[outlier_map]
        
        # Find outliers based on mmlu (because in wmdp we ignre the first few tokens)
        dist_mmlu = torch.cdist(activations["mmlu"][layer][:n_tokens], activations["mmlu"][layer][:n_tokens], p = 1)
        outlier_map_mmlu = get_zscore(dist_mmlu.mean(axis=0)) > 3 # 3 standard deviations away from the mean
        outliers_mmlu = np.array(tokens["mmlu"][:n_tokens])[outlier_map_mmlu]
        outliers.extend(list(set(outliers_wmdp) | set(outliers_mmlu)))
    
    # Final outlier list
    outlier_list = np.array(outliers)
    print(f"Outliers: {outlier_list}")
    
    remove_outliers_f = np.vectorize(lambda x: x not in outlier_list)
    
    wmdp_filter = remove_outliers_f(tokens["wmdp"])
    mmlu_filter = remove_outliers_f(tokens["mmlu"])
    wikitext_filter = remove_outliers_f(tokens["wikitext"])
    
    # Compute noise direction
    noise_directions = {"ground_truth": {}, "mmlu": {}, "wikitext": {}, "pca": {}}
    direction_vec_dest = os.path.join(output_dest, "direction_vectors")
    create_if_not_exists(direction_vec_dest)
    for layer in args.layers:
        
        original = activations["original"][layer][wmdp_filter]
        distorted = activations["distorted"][layer][wmdp_filter]
        noise_directions["ground_truth"][layer] = torch.mean(distorted, axis=0) - torch.mean(original, axis=0)
        
        mmlu = activations["mmlu"][layer][mmlu_filter]
        noise_directions["mmlu"][layer] = torch.mean(distorted, axis=0) - torch.mean(mmlu, axis=0)
        
        wiki = activations["wikitext"][layer][wikitext_filter]
        noise_directions["wikitext"][layer] = torch.mean(distorted, axis=0) - torch.mean(wiki, axis=0)
        
        # limit to 2500 samples for PCA otherwise it takes too long
        limit = 2500
        X = torch.vstack((original[:limit, :], distorted[:limit, :]))
        scaler = StandardScaler()
        scaled_X = scaler.fit_transform(X)
        pca = PCA().fit(scaled_X)
        noise_directions["pca"][layer] = torch.tensor(pca.components_[0,:].squeeze())
        
        # save noise directions
        torch.save(noise_directions["ground_truth"][layer], os.path.join(direction_vec_dest, f"{layer}_ground_truth.pth"))
        torch.save(noise_directions["mmlu"][layer], os.path.join(direction_vec_dest, f"{layer}_mmlu.pth"))
        torch.save(noise_directions["wikitext"][layer], os.path.join(direction_vec_dest, f"{layer}_wikitext.pth"))
        torch.save(noise_directions["pca"][layer], os.path.join(direction_vec_dest, f"{layer}_pca.pth"))
    
    return noise_directions

def test_orthogonalization(args, model, tokenizer, noise_directions: dict, output_dir: str) -> None:
    for noise_type in noise_directions:
        bench = Benchmark(
            output_dir,
            tasks=["mmlu"] + WMDP_TASKS,
            wandb_project="orthogonalization",
            config=dict(direction_type=noise_type,
                        **vars(copy(args))),
            run_name=f"{datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")}",
            save_requests=False,
            upload_requests_to_hf=False,
        )
        
        handles = []
        for layer in args.layers:
            ablation_hook = get_direction_ablation_output_hook(noise_directions[noise_type][layer])
            handle = model.model.layers[int(layer)].register_forward_hook(ablation_hook)
            handles.append(handle)
    
        bench.run(model, tokenizer)
        bench.finish()
        
        # after each noise type, remove the hooks
        for handle in handles:
            handle.remove()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--unlearned_model", type=str, required=True, help="Name or path of the unlearned model")
    parser.add_argument("--unlearned_tokenizer", type=str, help="Name or path of the unlearned tokenizer")
    parser.add_argument("--original_model", type=str, required=True, help="Name or path of the original model")
    parser.add_argument("--original_tokenizer", type=str, help="Name or path of the original tokenizer")
    parser.add_argument("--task", choices=["wmdp-bio", "wmdp-cyber"], help="The domain of the unlearned model")
    parser.add_argument("--layers", type=str, help="The layers to compute the noise direction vectors for and use for orthogonalization, e.g. '0,1,2,3,4,5,6,7', use 'all' to include all layers")
    parser.add_argument("--recompute_directions", action="store_true", help="Whether to recompute the directions when they are available")
    parser.add_argument("--ignore_first_x_tokens", type=int, default=40, help="The number of tokens to ignore at the beginning of the sequence for WMDP questions")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    
    # process arguments
    if args.unlearned_tokenizer is None:
        args.unlearned_tokenizer = args.unlearned_model
    
    if args.original_tokenizer is None:
        args.original_tokenizer = args.original_model
    
    
    # create output dir
    output_dest = os.path.join(OUTPUT_DIR, "orthogonalization", "data", args.unlearned_model.split("/")[-1] + "_" + args.task)
    create_if_not_exists(output_dest)
    
    # prepare seed and set torch to no grad
    seed_everything(args.seed)
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
    
    # process rest of arguments 
    if args.layers == "all":
        args.layers = list(range(model.config.num_hidden_layers))
    else:
        args.layers = args.layers.split(",")
        
    # compute noise direction vectors
    noise_directions = compute_noise_direction(args, model, tokenizer, model_original, tokenizer_original, output_dest)
    
    # get accuracy with ablated directions
    result_dest = os.path.join(OUTPUT_DIR, "orthogonalization", "results")
    create_if_not_exists(result_dest)
    
    test_orthogonalization(args, model, tokenizer, noise_directions, output_dest)