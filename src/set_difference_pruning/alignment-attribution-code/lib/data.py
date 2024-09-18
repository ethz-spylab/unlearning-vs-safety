# Code adapted from https://github.com/IST-DASLab/sparsegpt/blob/master/datautils.py

from httpx import get
import numpy as np
import random
import torch
from datasets import load_dataset

from src.util.globals import BIO_FORGET_CORPUS_PATH


# Set seed for reproducibility
def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)


# Wrapper for tokenized input IDs
class TokenizerWrapper:
    def __init__(self, input_ids):
        self.input_ids = input_ids


# Load and process aligned dataset
def get_align(nsamples, seed, seqlen, tokenizer, disentangle=False, mode="base"):
    print(f"Using align: {mode}")
    # Load train and test datasets
    if mode == "short":
        data_files = {"train": "./data/SFT_aligned_llama2-7b-chat-hf_train_short.csv"}
    else:
        data_files = {"train": "./data/SFT_aligned_llama2-7b-chat-hf_train.csv"}
    traindata = load_dataset("csv", data_files=data_files, split="train")
    trainloader = []
    random.seed(seed)
    if disentangle:
        traindata_sampled = traindata.shuffle(seed=seed).select(range(nsamples))
        for i in range(nsamples):
            trainenc_prompt = tokenizer(
                traindata_sampled["prompt"][i], return_tensors="pt"
            )
            trainenc_response = tokenizer(
                traindata_sampled["response"][i], return_tensors="pt"
            )
            inp = torch.cat(
                (trainenc_prompt.input_ids, trainenc_response.input_ids[:, 1:]), dim=1
            )
            tar = inp.clone()
            trainenc_prompt_len = trainenc_prompt.input_ids.shape[1]
            tar[:, :trainenc_prompt_len] = -100
            trainloader.append((inp, tar))
    else:
        # Encode datasets
        trainenc = tokenizer(" ".join(traindata["text"]), return_tensors="pt")

        # Generate samples from training set
        for _ in range(nsamples):
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))
    return trainloader, None

def get_mmlu(nsamples, seed, seqlen, tokenizer, disentangle):
    # replace part of the data with mmlu dev to retain relevant info
    
    trainloader = []
    mmlu = load_dataset("cais/mmlu", "all", split="dev")
    for line in mmlu:
        enc_prompt = tokenizer(line["question"], return_tensors="pt")
        enc_label = tokenizer(line["choices"][line["answer"]], return_tensors="pt")
        inp = torch.cat((enc_prompt.input_ids, enc_label.input_ids), dim=1)
        tar = inp.clone()
        if inp.shape[1] > seqlen:
            # to make sure that the model doesn't crash
            continue
        enc_prompt_len = enc_prompt.input_ids.shape[1]
        tar[:, :enc_prompt_len] = -100
        trainloader.append((inp, tar))
        
    return trainloader, None

def get_mmlu_abcd(nsamples, seed, seqlen, tokenizer, disentangle):
    # replace part of the data with mmlu dev to retain relevant info
    
    def apply_question_template(sample):
        prompt = sample["question"].strip() + "\n"
        options = ["A", "B", "C", "D"]
        for idx, choice in enumerate(sample["choices"]):
            prompt += f"{options[idx]}. {choice}\n"
        prompt += "Answer:"
        return prompt
    
    def apply_response_template(sample):
        options = ["A", "B", "C", "D"]
        prompt = options[sample["answer"]] + ". " + sample["choices"][sample["answer"]]
        return prompt
    
    trainloader = []
    mmlu = load_dataset("cais/mmlu", "all", split="dev")
    for line in mmlu:
        enc_prompt = tokenizer(apply_question_template(line), return_tensors="pt")
        enc_label = tokenizer(apply_response_template(line), return_tensors="pt")
        inp = torch.cat((enc_prompt.input_ids, enc_label.input_ids), dim=1)
        tar = inp.clone()
        if inp.shape[1] > seqlen:
            # to make sure that the model doesn't crash
            continue
        enc_prompt_len = enc_prompt.input_ids.shape[1]
        tar[:, :enc_prompt_len] = -100
        trainloader.append((inp, tar))
        
    return trainloader, None

def get_math(nsamples, seed, seqlen, tokenizer, dataset, disentangle):
    # replace part of the data with mmlu dev to retain relevant info
    
    trainloader = []
    if "add_sub" in dataset:
        data = load_dataset("deepmind/math_dataset", "arithmetic__add_sub_multiple", split="train")
    elif "mul_div" in dataset:
        data = load_dataset("deepmind/math_dataset", "arithmetic__mul_div_multiple", split="train")
    else:
        raise NotImplementedError("So far we support only 'add_sub' and 'mul_div' options")
    
    data = data.shuffle(seed=seed).select(range(nsamples))
    
    for line in data:
        enc_prompt = tokenizer(line["question"], return_tensors="pt")
        enc_label = tokenizer(line["answer"], return_tensors="pt")
        inp = torch.cat((enc_prompt.input_ids, enc_label.input_ids), dim=1)
        tar = inp.clone()
        if inp.shape[1] > seqlen:
            # to make sure that the model doesn't crash
            continue
        enc_prompt_len = enc_prompt.input_ids.shape[1]
        tar[:, :enc_prompt_len] = -100
        trainloader.append((inp, tar))
        
    return trainloader, None

# Load and process wikitext2 dataset
def get_wikitext2(nsamples, seed, seqlen, tokenizer, disentangle, name):
    print("Using wikitext2")
    # Load train and test datasets
    testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")

    trainloader = []
    random.seed(seed)
    traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    trainenc = tokenizer(" ".join(traindata["text"]), return_tensors="pt") # ~10.9 mil characters
    # Generate samples from training set
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        fake_delimiter = -1 if not disentangle else -50
        tar[:, :fake_delimiter] = -100
        trainloader.append((inp, tar))
        
    if "mmlu" in name:
        train_extra, _ = get_mmlu(nsamples, seed, seqlen, tokenizer, disentangle)
        trainloader.extend(train_extra) 
        
    return trainloader, testenc


def get_alpaca(nsamples, seed, seqlen, tokenizer, disentangle=False, dataset="alpaca"):
    print(f"Using alpaca: {dataset}")
    if dataset == "alpaca":
        data_files = {"train": "./data/alpaca_train.csv"}
    elif dataset == "alpaca_cleaned":
        data_files = {"train": "./data/alpaca_cleaned_train.csv"}
    elif dataset == "alpaca_cleaned_no_safety":
        data_files = {"train": "./data/alpaca_cleaned_no_safety_train.csv"}
    else:
        raise ValueError("Dataset not supported")
    traindata = load_dataset("csv", data_files=data_files, split="train")
    random.seed(seed)
    # Encode datasets
    trainloader = []
    if disentangle:
        traindata_sampled = traindata.shuffle(seed=seed).select(range(nsamples))
        for i in range(nsamples):
            trainenc_prompt = tokenizer(
                traindata_sampled["prompt"][i], return_tensors="pt"
            )
            trainenc_response = tokenizer(
                traindata_sampled["response"][i], return_tensors="pt"
            )
            inp = torch.cat(
                (trainenc_prompt.input_ids, trainenc_response.input_ids[:, 1:]), dim=1
            )  # to remove the first token of the response ('1')
            tar = inp.clone()
            trainenc_prompt_len = trainenc_prompt.input_ids.shape[1]
            tar[:, :trainenc_prompt_len] = -100
            trainloader.append((inp, tar))
    else:
        trainenc = tokenizer(" ".join(traindata["text"]), return_tensors="pt")
        # Generate samples from training set
        for _ in range(nsamples):
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))
    return trainloader, None

def get_wmdp(nsamples, seed, seqlen, tokenizer, dataset:str, disentangle):
    print(f"Using wmdp: {dataset}")
    
    dataset_args = dataset.split("_")
    subset = dataset_args[1]
    
    # Load train and test datasets
    if subset in "wmdp_bio-forget-corpus":
        traindata = load_dataset("json", data_dir=str(BIO_FORGET_CORPUS_PATH), split="train")
    else:
        traindata = load_dataset("cais/wmdp-corpora", subset, split="train")
        
    random.seed(seed)
    trainloader = []
    len_limit = int(1e+7)
    key = "text"
    if "abstract" in traindata.features and "abstract" in dataset:
        print("Using abstracts")
        key = "abstract"
    joined_data = " ".join(traindata[key])
    if len(joined_data) > len_limit:
        start = random.randint(0, len(joined_data) - len_limit - 1)
        joined_data = joined_data[start : start + len_limit] #take first 10 million characters
    print("length of dataset in characters: ", len(joined_data))
    trainenc = tokenizer(joined_data, return_tensors="pt")
    # Generate samples from training set
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        fake_delimiter = -1 if not disentangle else -50
        tar[:, :fake_delimiter] = -100
        trainloader.append((inp, tar))
        
    if "mmlu" in dataset_args:
        train_extra, _ = get_mmlu(nsamples, seed, seqlen, tokenizer, disentangle)
        trainloader.extend(train_extra) 
    
    return trainloader, None

def get_generations(nsamples, seed, seqlen, tokenizer, dataset_name, disentangle):
    parts = dataset_name.split("-")
    if parts[-1] == "prune":
        target_key = "unlearned_generation"
    elif parts[-1] == "util":
        target_key = "original_generation"
    else:
        raise ValueError("Invalid dataset name")
    
    model_name = "-".join(parts[1:-3])
    wmdp_split = "-".join(parts[-3:-1])
        
    trainloader = []
    ds = load_dataset(f"J4Q8/generations-{model_name}-{wmdp_split}", split="train").take(nsamples)
    for line in ds:
        enc_prompt = tokenizer(line["prompt"], return_tensors="pt")
        enc_label = tokenizer(line[target_key], return_tensors="pt")
        inp = torch.cat((enc_prompt.input_ids, enc_label.input_ids), dim=1)
        tar = inp.clone()
        if inp.shape[1] > seqlen:
            # to make sure that the model doesn't crash
            continue
        enc_prompt_len = enc_prompt.input_ids.shape[1]
        tar[:, :enc_prompt_len] = -100
        trainloader.append((inp, tar))
        
    return trainloader, None

# Function to select the appropriate loader based on dataset name
def get_loaders(
    name, nsamples=128, seed=0, seqlen=2048, tokenizer=None, disentangle=False
):
    if name.startswith("wikitext"):
        return get_wikitext2(nsamples, seed, seqlen, tokenizer, disentangle=disentangle, name=name)
    if name in ["alpaca", "alpaca_cleaned", "alpaca_cleaned_no_safety"]:
        return get_alpaca(nsamples, seed, seqlen, tokenizer, disentangle, dataset=name)
    if name == "align":
        return get_align(nsamples, seed, seqlen, tokenizer, disentangle=disentangle)
    if name == "align_short":
        return get_align(
            nsamples, seed, seqlen, tokenizer, disentangle=disentangle, mode="short"
        )
    if name.startswith("wmdp"):
        return get_wmdp(nsamples, seed, seqlen, tokenizer, dataset=name, disentangle=disentangle)
    if name == "mmlu":
        return get_mmlu(nsamples, seed, seqlen, tokenizer, disentangle=disentangle)
    if name == "mmlu_abcd":
        return get_mmlu_abcd(nsamples, seed, seqlen, tokenizer, disentangle=disentangle)
    if name.startswith("math"):
        return get_math(nsamples, seed, seqlen, tokenizer, dataset=name, disentangle=disentangle)
    if name.startswith("generations"):
        return get_generations(nsamples, seed, seqlen, tokenizer, dataset_name=name, disentangle=disentangle)