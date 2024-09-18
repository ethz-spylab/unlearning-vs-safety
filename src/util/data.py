from os import system
from random import shuffle
from typing import Optional

from datasets import load_dataset, Dataset

from src.util.globals import BIO_FORGET_CORPUS_PATH, HF_USERNAME

REQUEST_PROMPT_BIO = "Write a research article in the field of biology."
GENERATION_PREFIX_BIO = "Of course, here is a reseach article in the field of biology. "

REQUEST_PROMPT_CYBER = "Write a github page about cybersecurity."
GENERATION_PREFIX_CYBER = "Of course, here is a github page about cybersecurity. "

REQUEST_PROMPT_WIKI = "Write a wikipedia article."
GENERATION_PREFIX_WIKI = "Of course, here is a wikipedia article. "


def text_to_dialogue(text, request_prompt, generation_prefix):
        return [
            request_prompt, 
            generation_prefix + text["text"],
        ]
        
def dialogue_to_chat(dialogue, system_prompt):
    out = [{'role': 'system', 'content': system_prompt}]
    for idx, message in enumerate(dialogue):
        role = 'user' if idx%2==0 else 'assistant'
        out.append({'role': role, 'content': message})
    return out

def process_example(example, request_prompt, generation_prefix, system_prompt):
    out = text_to_dialogue(example, request_prompt, generation_prefix)
    out = dialogue_to_chat(out, system_prompt)
    example["messages"] = out
    return example

def process_text_dataset(dataset:Dataset, limit:Optional[int], train_val_ratio:Optional[float], request_prompt:str, generation_prefix:str, system_prompt:str):
    if limit is not None:
        dataset = dataset.select(range(limit))
        
    def one_argument_process_example(example):
        return process_example(example, request_prompt, generation_prefix, system_prompt)
    
    dataset = dataset.map(one_argument_process_example)
    dataset = dataset.select_columns("messages")
    dataset = dataset.shuffle()
    if train_val_ratio is not None:
        dataset = dataset.train_test_split(test_size=train_val_ratio)
    
    return dataset

def process_dpo_dataset(dataset:Dataset, limit:Optional[int], train_val_ratio:Optional[float], request_prompt:str, generation_prefix:str, system_prompt:str, key_with_answer:str):
    if limit is not None:
        dataset = dataset.select(range(limit))
        
    def one_argument_process_example(example):
        example["messages"] = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': example["prompt"]},
            {'role': 'assistant', 'content': example[key_with_answer]},
        ]
        return example
    
    dataset = dataset.map(one_argument_process_example)
    dataset = dataset.select_columns("messages")
    dataset = dataset.shuffle()
    if train_val_ratio is not None:
        dataset = dataset.train_test_split(test_size=train_val_ratio)
    
    return dataset

def get_bio_retain_dataset(limit:Optional[int] = None, train_val_ratio:Optional[float]=None, system_prompt:str=""):
    dataset = load_dataset("cais/wmdp-corpora", "bio-retain-corpus", split="train")
    return process_text_dataset(dataset, limit, train_val_ratio, REQUEST_PROMPT_BIO, GENERATION_PREFIX_BIO, system_prompt)

def get_cyber_retain_dataset(limit:Optional[int] = None, train_val_ratio:Optional[float]=None, system_prompt:str=""):
    dataset = load_dataset("cais/wmdp-corpora", "cyber-retain-corpus", split="train")
    return process_text_dataset(dataset, limit, train_val_ratio, REQUEST_PROMPT_CYBER, GENERATION_PREFIX_CYBER, system_prompt)

def get_bio_forget_dataset(limit:Optional[int] = None, train_val_ratio:Optional[float]=None, system_prompt:str=""):
    dataset = load_dataset("json", data_dir=str(BIO_FORGET_CORPUS_PATH), split="train")
    return process_text_dataset(dataset, limit, train_val_ratio, REQUEST_PROMPT_BIO, GENERATION_PREFIX_BIO, system_prompt)

def get_cyber_forget_dataset(limit:Optional[int] = None, train_val_ratio:Optional[float]=None, system_prompt:str=""):
    dataset = load_dataset("cais/wmdp-corpora", "cyber-forget-corpus", split="train")
    return process_text_dataset(dataset, limit, train_val_ratio, REQUEST_PROMPT_CYBER, GENERATION_PREFIX_CYBER, system_prompt)

def get_wikitext_dataset(limit:Optional[int] = None, train_val_ratio:Optional[float]=None, system_prompt:str=""):
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    return process_text_dataset(dataset, limit, train_val_ratio, REQUEST_PROMPT_WIKI, GENERATION_PREFIX_WIKI, system_prompt)

## Multiple choice datasets

def get_bio_retain_mc_dataset(limit:Optional[int] = None, train_val_ratio:Optional[float]=None, system_prompt:str=""):
    dataset = load_dataset(f"J4Q8/bio_retain_dpo", split="train")
    return process_dpo_dataset(dataset, limit, train_val_ratio, REQUEST_PROMPT_BIO, GENERATION_PREFIX_BIO, system_prompt, "chosen")

def get_cyber_retain_mc_dataset(limit:Optional[int] = None, train_val_ratio:Optional[float]=None, system_prompt:str=""):
    dataset = load_dataset(f"J4Q8/cyber_retain_dpo", split="train")
    return process_dpo_dataset(dataset, limit, train_val_ratio, REQUEST_PROMPT_CYBER, GENERATION_PREFIX_CYBER, system_prompt, "chosen")

def get_bio_forget_mc_dataset(limit:Optional[int] = None, train_val_ratio:Optional[float]=None, system_prompt:str=""):
    dataset = load_dataset(f"J4Q8/bio_forget_dpo", split="train")
    return process_dpo_dataset(dataset, limit, train_val_ratio, REQUEST_PROMPT_BIO, GENERATION_PREFIX_BIO, system_prompt, "rejected")

def get_cyber_forget_mc_dataset(limit:Optional[int] = None, train_val_ratio:Optional[float]=None, system_prompt:str=""):
    dataset = load_dataset(f"J4Q8/cyber_forget_dpo", split="train")
    return process_dpo_dataset(dataset, limit, train_val_ratio, REQUEST_PROMPT_CYBER, GENERATION_PREFIX_CYBER, system_prompt, "rejected")

def get_wikitext_bio_mc_dataset(limit:Optional[int] = None, train_val_ratio:Optional[float]=None, system_prompt:str=""):
    dataset = load_dataset(f"J4Q8/wikitext_bio_dpo", split="train")
    return process_dpo_dataset(dataset, limit, train_val_ratio, REQUEST_PROMPT_WIKI, GENERATION_PREFIX_WIKI, system_prompt, "chosen")

def get_wikitext_cyber_mc_dataset(limit:Optional[int] = None, train_val_ratio:Optional[float]=None, system_prompt:str=""):
    dataset = load_dataset(f"J4Q8/wikitext_cyber_dpo", split="train")
    return process_dpo_dataset(dataset, limit, train_val_ratio, REQUEST_PROMPT_WIKI, GENERATION_PREFIX_WIKI, system_prompt, "chosen")


# for debuging
if __name__ == "__main__":
    get_bio_forget_dataset(100)