import os
from argparse import ArgumentParser
import token

import torch
from tqdm import tqdm
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.util.globals import OUTPUT_DIR, SEED, CUSTOM_DATASET_DIR, HF_USERNAME
from src.util.helpers import seed_everything


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--unlearned_model", type=str, required=True, help="Name or path of the unlearned model")
    parser.add_argument("--unlearned_tokenizer", type=str, help="Name or path of the unlearned tokenizer")
    parser.add_argument("--original_model", type=str, required=True, help="Name or path of the original model")
    parser.add_argument("--original_tokenizer", type=str, help="Name or path of the original tokenizer")
    parser.add_argument("--task", choices=["wmdp-bio", "wmdp-cyber"], help="The domain of the unlearned model")
    parser.add_argument("--limit", type=int, default=512, help="The number of samples to generate")
    parser.add_argument("--n_tokens", type=int, default=50, help="Number of tokens to generate")
    args = parser.parse_args()
    
    # process arguments
    if args.unlearned_tokenizer is None:
        args.unlearned_tokenizer = args.unlearned_model
    
    if args.original_tokenizer is None:
        args.original_tokenizer = args.original_model
        
        
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
    
    if args.task == "wmdp-bio":
        dataset = load_dataset('J4Q8/bio_forget_dpo', split='train').shuffle(SEED).take(args.limit)
    elif args.task == "wmdp-cyber":
        dataset = load_dataset('J4Q8/cyber_forget_dpo', split='train').shuffle(SEED).take(args.limit)
    else:
        raise ValueError("Invalid task")
    
    hf_dataset = {"prompt":[],
                  "original_generation":[], 
                  "unlearned_generation":[],}
    for i, example in enumerate(tqdm(dataset)):
        hf_dataset["prompt"].append(example["prompt"])
        
        # get original generation
        model_inputs = tokenizer_original(example["prompt"], return_tensors="pt")
        model_inputs = {k: v.to(model_original.device) for k, v in model_inputs.items()}
        original_generation = model_original.generate(**model_inputs, max_new_tokens=args.n_tokens, min_new_tokens=args.n_tokens)
        original_generation = original_generation[:, -args.n_tokens:]
        hf_dataset["original_generation"].append(tokenizer.decode(original_generation[0]))
        
        # get unlearned generation
        model_inputs = tokenizer(example["prompt"], return_tensors="pt")
        model_inputs = {k: v.to(model.device) for k, v in model_inputs.items()}
        unlearned_generation = model.generate(**model_inputs, max_new_tokens=args.n_tokens, min_new_tokens=args.n_tokens)
        unlearned_generation = unlearned_generation[:, -args.n_tokens:]
        hf_dataset["unlearned_generation"].append(tokenizer.decode(unlearned_generation[0]))
    
    # upload to huggingface
    formatted_dataset = Dataset.from_dict(hf_dataset)
    formatted_dataset.push_to_hub(f"{HF_USERNAME}/generations-{args.unlearned_model.split("/")[-1]}-{args.task}", private=True)