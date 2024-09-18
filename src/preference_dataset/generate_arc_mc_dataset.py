import json
import random
import os
from typing import Optional

from tqdm import tqdm
from datasets import load_dataset, Dataset
from pydantic import BaseModel

from .refusal_list import REFUSAL_LIST
from ..util.helpers import create_if_not_exists
from ..util.globals import OPENAI_ORG_ID, BIO_FORGET_CORPUS_PATH, CUSTOM_DATASET_DIR, HF_USERNAME

SEED = 42

def format_prompt(sample: str, contex: str) -> str:
    prompt = f"The following are multiple choice questions (with answers) about {contex}.\n\n"
    prompt += f"{sample['question']}\n"
    ABCDEF = list("ABCDEF")
    for i, option in enumerate(sample["choices"]["text"]):
        prompt += f"{ABCDEF[i]}. {option}\n"
    prompt += "Answer:"
    
    answer_id = sample["choices"]["label"].index(sample["answerKey"])
    answer = f"{ABCDEF[answer_id]}. {sample["choices"]["text"][answer_id]}\n"
    return prompt, answer

if __name__ == "__main__":
    
    random.seed(SEED)
    ds1 = load_dataset("allenai/ai2_arc", "ARC-Easy", split="all") # we do not care about train test split
    ds2 = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="all") # we do not care about train test split

    mc_bio_retain_dpo = {
            "prompt": [],
            "chosen": [],
            "rejected": [],
        }

    mc_cyber_retain_dpo = {
            "prompt": [],
            "chosen": [],
            "rejected": [],
        }
    for ds in [ds1, ds2]:
        for sample in ds:
            # generate one batch using "biology" as a context
            prompt, answer = format_prompt(sample, "biology")
            
            mc_bio_retain_dpo["prompt"].append(prompt)
            mc_bio_retain_dpo["chosen"].append(answer)
            mc_bio_retain_dpo["rejected"].append(random.choice(REFUSAL_LIST))
            
            # and genereate another batch using "cybersecurity" as a context
            prompt, answer = format_prompt(sample, "cybersecurity")
            
            mc_cyber_retain_dpo["prompt"].append(prompt)
            mc_cyber_retain_dpo["chosen"].append(answer)
            mc_cyber_retain_dpo["rejected"].append(random.choice(REFUSAL_LIST))


    with open(os.path.join(CUSTOM_DATASET_DIR, "mc_bio_retain_dpo.json"), "w") as f:
        json.dump(mc_bio_retain_dpo, f)
        
    with open(os.path.join(CUSTOM_DATASET_DIR, "mc_cyber_retain_dpo.json"), "w") as f:
        json.dump(mc_bio_retain_dpo, f)
    
    # upload to huggingface
    final_bio = Dataset.from_dict(mc_bio_retain_dpo)
    final_bio.push_to_hub(f"{HF_USERNAME}/mc_bio_retain_dpo", private=True) 
    
    final_cyber = Dataset.from_dict(mc_cyber_retain_dpo)
    final_cyber.push_to_hub(f"{HF_USERNAME}/mc_cyber_retain_dpo", private=True) 