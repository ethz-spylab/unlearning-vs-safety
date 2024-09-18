import os
import json
import random
from typing import Callable, Literal
import numpy as np
import torch
from transformers import AutoTokenizer

from src.util.globals import WMDP_OPTIONS

def seed_everything(seed: int) -> None:    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_if_not_exists(directory: str) -> None:
    if not os.path.exists(directory):
        os.makedirs(directory)

def save_tokenizer(model_dir, tokenizer_path):
    # saves tokenizer to model_dir if absent, else does nothing
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
    except:
        print(f"Saving tokenizer ({tokenizer_path}) to directory: {model_dir}")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        tokenizer.save_pretrained(model_dir)
        
def jsonify(d):
    if isinstance(d, dict):
        for key, val in d.items():
            d[key] = jsonify(val)
        return d
    elif isinstance(d, (tuple, list)):
        return [jsonify(val) for val in d]
    elif isinstance(d, (int, float, str)):
        return d
    else:
        return str(d)

def save_as_json(dest:str, file_name:str, d:dict) -> None:
    result_path = os.path.join(dest, file_name)
    with open(result_path, "w") as f:
        json.dump(jsonify(d), f)
        
def get_question(sample: dict, print_prompt:bool = False, context:Literal["biology", "cybersecurity", "chemistry"] = "biology") -> tuple[str, str]:
    """Get questions wmdp style. This is less customizable than get_wmdp_prompt and is meant to be auxilary function for experiments rather than the benchmarking class.

    Args:
        sample (dict):  sample from HF dataset
        print_prompt (bool, optional): whether to print final prompt. Defaults to True
        simple_perturb (_type_, optional): perturbation to be applied to the question and choices must take a string and return a string. Defaults to lambda x:x.
        context (Literal["biology", "cybersecurity", "chemistry"], optional): One of the contexts that will be provided in a header prompt. Defaults to "biology".

    Returns:
        tuple[str, str]: question prompt and answer in {"A", "B", "C", "D"}
    """
    # WMDP prompt template as defined on lm_eval_harness
    prompt = f"The following are multiple choice questions (with answers) about {context}.\n\n"
    prompt += sample["question"].strip() + "\n"
    for idx, choice in enumerate(sample["choices"]):
        prompt += f"{WMDP_OPTIONS[idx]}. {choice}\n"
    prompt += "Answer:"
    if print_prompt:
        print(prompt)
        print(WMDP_OPTIONS[int(sample["answer"])])
    return prompt, WMDP_OPTIONS[int(sample["answer"])]

