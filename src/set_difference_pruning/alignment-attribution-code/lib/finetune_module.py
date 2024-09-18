import os
import gc
import shutil
from typing import Optional

from datasets import load_dataset, Dataset


import torch
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model
from transformers import TrainingArguments


def create_if_not_exists(directory: str) -> None:
    if not os.path.exists(directory):
        os.makedirs(directory)
        
def text_to_dialogue(text, request_prompt, generation_prefix):
    return [
        request_prompt, 
        generation_prefix + text["text"],
    ]
        
def dialogue_to_chat(dialogue):
    out = [{'role': 'system', 'content': "You are a helpful assistant."}]
    for idx, message in enumerate(dialogue):
        role = 'user' if idx%2==0 else 'assistant'
        out.append({'role': role, 'content': message})
    return out

def process_example(example, request_prompt, generation_prefix):
    out = text_to_dialogue(example, request_prompt, generation_prefix)
    out = dialogue_to_chat(out)
    example["messages"] = out
    return example

def process_text_dataset(dataset:Dataset, limit:Optional[int], request_prompt:str, generation_prefix:str, seed:int=42):
    dataset = dataset.shuffle(seed)
    
    if limit is not None:
        dataset = dataset.select(range(limit))
        
    def one_argument_process_example(example):
        return process_example(example, request_prompt, generation_prefix)
    
    dataset = dataset.map(one_argument_process_example)
    dataset = dataset.select_columns("messages")
    
    return dataset

def get_wiki_dataset(seed:int):
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    return process_text_dataset(dataset, 100, "Write a wikipedia article.", "Sure, here is a wikipedia article: ", seed)

class FinetuningModule():
    max_seq_len = 1024
    dtype = torch.bfloat16
    lora_rank = 128
    lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj","gate_proj", "up_proj", "down_proj",]
    lora_alpha = 16
    lora_dropout = 0
    
    def finetune(self, model, tokenizer, seed):
        # Get model
        model.config.use_cache = False
        model.config.pretraining_tp = 1
        model.gradient_checkpointing_enable()
        
        # Tokenizer params
        tokenizer.padding_side = 'right'
        tokenizer.pad_token = tokenizer.eos_token
        
        # get dataset 
        dataset = get_wiki_dataset(seed)
        
        # Set up LORA
        peft_config = LoraConfig(
            model,
            r = self.lora_rank, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
            target_modules = self.lora_target_modules,
            lora_alpha = self.lora_alpha,
            lora_dropout = self.lora_dropout, # Supports any, but = 0 is optimized
            bias = "none",    # Supports any, but = "none" is optimized
        )
        peft_model = get_peft_model(model, peft_config)
        
        create_if_not_exists("tmp")
        
        # initialize trainer
        trainer = SFTTrainer(
            model = peft_model,
            tokenizer = tokenizer,
            train_dataset = dataset,
            max_seq_length = self.max_seq_len,
            dataset_num_proc = 2,
            peft_config=peft_config,
            packing = False, # Can make training 5x faster for short sequences.
            args = TrainingArguments( #these can be retrieved later on so no need to make them into Global constants for saving purposes
                per_device_train_batch_size = 2,
                gradient_accumulation_steps = 4,
                warmup_ratio = 0.05,
                num_train_epochs = 3,
                learning_rate = 2e-4,
                optim = "adamw_torch",
                weight_decay = 0.01,
                logging_steps = 1,
                lr_scheduler_type = "linear",
                seed = 42,
                output_dir="tmp",
                run_name="tmp",
            ),
        )
        # hack to make trainer return eval loss for peft model
        trainer.can_return_loss = True
        
        shutil.rmtree("tmp")
        
        ### Finetune
        _ = trainer.train()
        
        merged_model = trainer.model.merge_and_unload()
        
        # remove all leftovers
        del trainer, dataset, peft_model
        gc.collect()
        torch.cuda.empty_cache()
        
        assert merged_model, "Merged model is none, ups"
        
        return merged_model